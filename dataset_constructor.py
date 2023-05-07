from .config import Config
from .utils import load_json
from .features import read_wav, get_mfb, get_bert_embedding, get_w2v2
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.utils.class_weight import compute_class_weight
import multiprocessing.dummy as mp_thread
import numpy as np
import torch
import librosa
import pickle
import os
import copy

# Iterable PyTorch dataset instance
class DatasetInstance(torch.utils.data.Dataset):
    def __init__(self, dataset_constructor, keys_to_use):
        self.dataset_id = dataset_constructor.dataset_id
        self.dataset_ids = {key: self.dataset_id for key in keys_to_use}
        dicts_to_copy = ['wav_lengths', 'wav_rms', 'labels']
        for dict_name in dicts_to_copy:
            parent_dict = getattr(dataset_constructor, dict_name)
            new_dict = {}
            for key in keys_to_use:
                new_dict[key] = copy.deepcopy(parent_dict[key])
            setattr(self, dict_name, new_dict)

        self.split_keys = keys_to_use.copy()

    def __len__(self):
        return len(self.split_keys)

    def __getitem__(self, idx):
        item_key = self.split_keys[idx]
        labels = self.labels[item_key]

        # Return a dictionary of all labels for this item + the current dataset id
        return {**labels, 'item_key': item_key, 'dataset_id': self.dataset_ids[item_key]}

    def class_weights(self):
        act_labels = [self.labels[key]['act_bin'] if 'act_bin' in self.labels[key] else self.labels[key]['self-report-act_bin'] for key in self.labels]
        val_labels = [self.labels[key]['val_bin'] if 'val_bin' in self.labels[key] else self.labels[key]['self-report-val_bin'] for key in self.labels]
        return {
            'act': compute_class_weight(class_weight='balanced', classes=np.unique(act_labels), y=np.array(act_labels)),
            'val': compute_class_weight(class_weight='balanced', classes=np.unique(val_labels), y=np.array(val_labels))
        }

# We just need a new init function for the multi domain dataset as it just combines other datasets into this 
class MultiDomainDataset(DatasetInstance):
    def __init__(self, dataset_instances, dataset_strs):
        assert all([type(dataset) == DatasetInstance for dataset in dataset_instances]), 'MultiDomainMMFusionDataset constructor expects all provided datasets to be of type MMFusionDataset'
        assert len(dataset_instances) == len(dataset_strs), f'Expected to get same number of datasets ({len(dataset_instances)}) as dataset_strs/dataset string names ({len(dataset_strs)})'
        print('Merging datasets with id', [ds.dataset_id for ds in dataset_instances])
        print('dataset sizes:', [len(ds.split_keys) for ds in dataset_instances])
        self.dataset_ids = {}
        self.split_keys = []
        dicts_to_copy = ['wav_lengths', 'wav_rms', 'labels']
        for dict_name in dicts_to_copy:
            new_dict = {}
            for i, dataset in enumerate(dataset_instances):
                dataset_str = dataset_strs[i]
                parent_dict = getattr(dataset, dict_name)
                for key in parent_dict:
                    new_key = f'{dataset_str}_{key}'
                    new_dict[new_key] = parent_dict[key]
                    self.dataset_ids[new_key] = dataset.dataset_id
                    self.split_keys.append(new_key)
            setattr(self, dict_name, new_dict)

        self.split_keys = list(set(self.split_keys))
    
    # Essentially the same as the base class, but instead of returning self.dataset_id return the stored sample id 
    def __getitem__(self, idx):
        item_key = self.split_keys[idx]
        labels = self.labels[item_key]

        # Return a dictionary of all labels for this item + the current dataset id
        return {**labels, 'item_key': item_key, 'dataset_id': self.dataset_ids[item_key]}

# This class loads all the relevant data and then returns iterable datasets for each 
# data split
class DatasetConstructor:
    # Use a singleton class style here
    # This way if the data was already loaded once the code won't waste time reading
    # from the disk again
    dataset_instances = {}
    def __new__(cls, *args, **kwargs):
        if not cls.__name__ in DatasetConstructor.dataset_instances:
            DatasetConstructor.dataset_instances[cls.__name__] = object.__new__(cls)
            DatasetConstructor.dataset_instances[cls.__name__]._initialised = False
        return DatasetConstructor.dataset_instances[cls.__name__]

    def __init__(self, dataset_id, filter_fn=None, dataset_save_location=None):
        # If this is a singleton class that was already initialised then don't load data again
        if self._initialised:
            return

        self.dataset_id = dataset_id
        self.config = Config()

        if dataset_save_location is not None:
            # Load dataset if file exists 
            if os.path.exists(dataset_save_location):
                self.load(dataset_save_location)
                return

        assert filter_fn is None or callable(filter_fn), 'If filter_fn is defined it should be a callable function\nreturning True if a key should be kept and False if it should be deleted'

        # self.labels should be of the format {'key': {label: label_value...}}
        self.labels = self.read_labels()

        self.prepare_labels()

        if filter_fn is not None:
            self.labels = {key: self.labels[key] for key in list(self.labels.keys()) if filter_fn(key)} # Filter the labels based on the filter_fn

        # Features will be added to the labels file
        self.generate_features()

        self._initialised = True

        if dataset_save_location is not None:
            # Load dataset if file exists 
            self.save(dataset_save_location)

    def load(self, load_path):
        with open(load_path, 'rb') as f:
            loaded_attributes = pickle.load(f)
            assert self.config == loaded_attributes['config'], 'Saved dataset config does not match config set in data_config.yaml'
            for key in loaded_attributes:
                setattr(self, key, loaded_attributes[key])

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def read_labels(self):
        raise NotImplementedError('Template dataset constructor called. This class should be inherited and have read_labels return a dictionary of loaded labels and a dictionary of label meta information')

    def prepare_labels(self, new_minimum=-1, new_maximum=1, items_to_scale=None):
        if items_to_scale is None:
            raise ValueError('prepare_labels should be overridden and called with items_to_scale set to a list of strings defining which items should be scaled')
        # function min-max scales continuous labels and bins categorical labels
        num_bins = self.config['num_labels']
        for value_type in items_to_scale:
            all_values = [value[value_type] for value in self.labels.values() if value_type in value]
            min_val = min(all_values)
            max_val = max(all_values)
            buckets = np.linspace(new_minimum, new_maximum, num=num_bins+1)
            for key in self.labels:
                if value_type in self.labels[key]:
                    self.labels[key][value_type] = (new_maximum-new_minimum) * (self.labels[key][value_type] - min_val)/(max_val - min_val) + new_minimum
                    self.labels[key][f'{value_type}_bin'] = min(np.searchsorted(buckets, self.labels[key][value_type], side='right')-1, num_bins-1)

    def get_dataset_splits(self, data_split_type):
        assert type(data_split_type) == str, 'data_split_type should be a string\neither describing the type of split that should be used\nor a filepath to a json file defining splits'
        if data_split_type[-5:].lower() == '.json' and os.path.exists(data_split_type):
            return load_json(data_split_type)
        return data_split_type # Whatever inherits this template should override this method and handle the string

    def build(self, data_split_type=None, **kwargs):
        split_dict = self.get_dataset_splits(data_split_type, **kwargs)
        if split_dict == 'all': # If using all keys just return the one dataset 
            return DatasetInstance(self, list(self.labels.keys()))

        dataset_split_dict = {}
        for split in split_dict:
            keys = split_dict[split]
            if split == 'train_keys':
                split = 'train'
            if split == 'val_key':
                split = 'val'
            dataset_split_dict[split] = DatasetInstance(self, keys)

        return dataset_split_dict

    def generate_features(self):
        self.wav_keys_to_use = self.get_wavs()

        # Generic datasets for testing may have no labels file
        # in this case we want to create new instances for the wav files
        self.create_new_labels = self.labels is None

        # Now remove all labels that don't have an associated wav file and vice versa
        if self.labels is not None:
            label_keys = set(self.labels.keys())
            wav_keys = set(self.wav_keys_to_use.keys())
            invalid_label_keys = label_keys - wav_keys
            print(f'Removing {len(invalid_label_keys)} labels where no corresponding wav file was found.')
            print(f'labels were: {invalid_label_keys}')
            for key in invalid_label_keys:
                del self.labels[key]
            
            invalid_wav_keys = wav_keys - label_keys
            print(f'Removing {len(invalid_wav_keys)} wavs where no corresponding label key was found.')
            print(f'wavs were: {invalid_wav_keys}')
            for key in invalid_wav_keys:
                del self.wav_keys_to_use[key]
            
        # Invalid wav keys (too long or short) can be removed while making audio features
        num_workers = 2 # TODO: This value should vary based on the settings -- wav2vec2/bert better with lower number over raw values (more processing = less workers)
        pool = mp_thread.Pool(num_workers)

        # Define some values that we want to track about the wav files
        self.wav_rms = {} # Used for SNR
        self.wav_lengths = {}
        self.removed_wavs = []

        # Create w2v2 and bert models if required by features
        if self.config['audio_feature_type'] != 'raw' and self.config['audio_feature_type'] != 'mfb' and type(self.config['audio_feature_type']) == str:
            self.wav2vec2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.config['audio_feature_type'])
            self.wav2vec2_model = Wav2Vec2Model.from_pretrained(self.config['audio_feature_type'])
            if torch.cuda.is_available():
                self.wav2vec2_model = self.wav2vec2_model.to('cuda')
            self.wav2vec2_model.eval()

        if self.config['text_feature_type'] != 'raw' and type(self.config['text_feature_type']) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['text_feature_type'])
            self.bert_model = AutoModel.from_pretrained(self.config['text_feature_type'])
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.to('cuda')

        for _ in tqdm(pool.imap_unordered(self.save_speech_features, self.wav_keys_to_use.keys()), total=len(self.wav_keys_to_use.keys()), desc='Saving and extracting speech features'):
            pass
        
        pool.close()
        pool.join()
        print(f'Removed {len(self.removed_wavs)} wav files for being too short/long')
        print(f'Wav files were {self.removed_wavs}')
        del self.removed_wavs

    def save_speech_features(self, wav_key):
        # First check if the wav key has a label
        if not self.create_new_labels and wav_key not in self.labels:
            del self.wav_keys_to_use[wav_key]
            return

        wav_path = self.wav_keys_to_use[wav_key]
        y, sr = read_wav(wav_path)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < self.config['min_len'] or duration > self.config['max_len']:
            self.removed_wavs.append(wav_key)
            del self.wav_keys_to_use[wav_key]
            if wav_key in self.labels:
                del self.labels[wav_key]
            return # Wav/label has been deleted and is invalid

        # Valid wav key
        # create label if necessary
        if self.create_new_labels and wav_key not in self.labels:
            self.labels[wav_key] = {}

        if self.config['audio_feature_type'] == 'raw':
            self.labels[wav_key]['audio'] = y
        elif self.config['audio_feature_type'] == 'mfb':
            self.labels[wav_key]['audio'] = get_mfb(y, sr)
        elif type(self.config['audio_feature_type']) == str:
            self.labels[wav_key]['audio'] = get_w2v2(y, sr, w2v2_extractor=self.wav2vec2_feature_extractor, w2v2_model=self.wav2vec2_model)
        else:
            raise IOError(f"Uknown audio feature type {self.config['audio_feature_type']} in data_config.yaml")

        # Now that we have the valid wav key, we should store the additional wav information
        # Calculate rms for snr calculations
        rms = np.mean(librosa.feature.rms(y=y))
        self.wav_rms[wav_key] = rms
        self.wav_lengths[wav_key] = duration

        # Now get text features
        # if the labels file existed then we just use the loaded transcripts
        # if not then ASR is required
        if 'transcript' not in self.labels[wav_key]:
            raise NotImplementedError('TODO: Implement ASR')
        if self.config['text_feature_type'] == 'raw':
            self.labels[wav_key]['transcript'] = self.labels[wav_key]['transcript']
        elif type(self.config['text_feature_type']) == str:
            self.labels[wav_key]['transcript'] = get_bert_embedding(self.labels[wav_key]['transcript'], bert_tokenizer=self.tokenizer, bert_model=self.bert_model)
        else:
            raise IOError(f"Uknown text feature type {self.config['audio_feature_type']} in data_config.yaml")

    def get_wavs(self):
        raise NotImplementedError('Template dataset constructor called. This class should be inherited and have get_wavs return a dictionary of key: wav_file_path. Key should match up the ID in the labels file.')
