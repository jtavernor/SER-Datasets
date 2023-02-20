from config import Config
from utils import load_json
from features import read_wav, get_mfb, get_bert_embedding, get_w2v2
from tqdm import tqdm
import multiprocessing.dummy as mp_thread
import numpy as np
import torch
import librosa

# Iterable PyTorch dataset instance
class DatasetInstance(torch.utils.data.Dataset):
    def __init__(self, dataset_constructor, keys_to_use):
        self.dataset_id = parent_dataset.dataset_id
        self.dataset_ids = {}
        dicts_to_copy = ['wav_lengths', 'wav_rms', 'labels']
        for dict_name in dicts_to_copy:
            parent_dict = getattr(parent_dataset, dict_name)
            new_dict = {key: parent_dict[key] for key in parent_dict}
            setattr(self, dict_name, new_dict)

        self.split_keys = keys_to_use.copy()

    def __len__(self):
        return len(self.split_keys)

    def __getitem__(self, idx):
        item_key = self.split_keys[idx]
        labels = self.labels[item_key]
        act_label, act_bin_label, val_label, val_bin_label, transcript_emb, audio_feats = labels['act'], labels['act_bin'], labels['val'], labels['val_bin'], self.transcript_embeddings[item_key], self.features[item_key]

        return audio_feats, transcript_emb, act_label, act_bin_label, val_label, val_bin_label, item_key, self.dataset_ids[item_key]

    def class_weights(self):
        act_labels = [self.labels[key]['act_bin'] for key in self.labels]
        val_labels = [self.labels[key]['val_bin'] for key in self.labels]
        return {
            'act': compute_class_weight(class_weight='balanced', classes=np.unique(act_labels), y=np.array(act_labels)),
            'val': compute_class_weight(class_weight='balanced', classes=np.unique(val_labels), y=np.array(val_labels))
        }

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

        if dataset_save_location is not None:
            # Load dataset if file exists 
            pass

        assert filter_fn is None or callable(filter_fn), 'If filter_fn is defined it should be a callable function\nreturning True if a key should be kept and False if it should be deleted'

        self.config = Config()

        # self.labels should be of the format {'key': {label: label_value...}}
        self.labels = self.read_labels()

        self.prepare_labels()

        if filter_fn is not None:
            self.labels = {key: self.labels[key] for key in list(self.labels.keys()) if filter_fn(key)} # Filter the labels based on the filter_fn

        # Features will be added to the labels file
        self.generate_features()

    def read_labels(self):
        raise NotImplementedError('Template dataset constructor called. This class should be inherited and have read_labels return a dictionary of loaded labels and a dictionary of label meta information')

    def prepare_labels(self, new_minimum=-1, new_maximum=1, items_to_scale=['act', 'val', 'self-report-val', 'self-report-act']):
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

    def build(self, data_split_type):
        split_dict = self.get_dataset_splits(data_split_type)
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
        num_workers = 8
        pool = mp_thread.Pool(num_workers)

        # Define some values that we want to track about the wav files
        self.wav_rms = {} # Used for SNR
        self.wav_lengths = {}

        for _ in tqdm(pool.imap_unordered(self.save_speech_features, self.wav_keys_to_use.keys()), total=len(self.wav_keys_to_use.keys()), desc='Saving audio features'):
            pass
        
        pool.close()
        pool.join()

    def save_speech_features(self, wav_key):
        # First check if the wav key has a label
        if not self.create_new_labels and wav_key not in self.labels:
            del self.wav_keys_to_use[wav_key]
            return

        wav_path = self.wav_keys_to_use[wav_key]
        y, sr = read_wav(wav_path)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < self.config['min_len'] or duration > self.config['max_len']:
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
            self.labels[wav_key]['audio'] = get_w2v2(y, sr, w2v2_model=self.config['audio_feature_type'])
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
            self.labels[wav_key]['text'] = self.labels[wav_key]['transcript']
        elif type(self.config['text_feature_type']) == str:
            self.labels[wav_key]['text'] = get_bert_embedding(self.labels[wav_key]['transcript'], bert_model=self.config['text_feature_type'])
        else:
            raise IOError(f"Uknown text feature type {self.config['audio_feature_type']} in data_config.yaml")

    def get_wavs(self):
        raise NotImplementedError('Template dataset constructor called. This class should be inherited and have get_wavs return a dictionary of key: wav_file_path. Key should match up the ID in the labels file.')
