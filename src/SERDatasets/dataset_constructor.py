from multiprocessing import Pool
from .config import Config
from .utils import load_json, load_pk
from .features import read_wav, get_bert_embedding, get_w2v2, pool_w2v2
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.utils.class_weight import compute_class_weight
from .kde_probability import kde_probability_bs
import multiprocessing.dummy as mp_thread
import numpy as np
import torch
import librosa
import pickle
import os
import copy

# Iterable PyTorch dataset instance
class DatasetInstance(torch.utils.data.Dataset):
    def __init__(self, dataset_constructor, keys_to_use, keys_to_scale, kde_size=5):
        self.config = dataset_constructor.config
        self.dataset_id = dataset_constructor.dataset_id
        self.dataset_ids = {key: self.dataset_id for key in keys_to_use}
        self.kde_size = kde_size
        dicts_to_copy = ['wav_lengths', 'wav_rms', 'labels']
        has_individual_evaluators = hasattr(dataset_constructor, 'individual_evaluators')
        if has_individual_evaluators:
            self.individual_evaluators = {}
        new_keys_to_use = copy.deepcopy(keys_to_use)
        self.split_keys = list(set(new_keys_to_use))
        for dict_name in dicts_to_copy:
            parent_dict = getattr(dataset_constructor, dict_name)
            new_dict = {}
            for key in keys_to_use:
                new_dict[key] = copy.deepcopy(parent_dict[key])
            setattr(self, dict_name, new_dict)

        if has_individual_evaluators:
            for key in self.labels:
                for evaluator in self.labels[key]['evaluators']:
                    if evaluator not in self.individual_evaluators:
                        self.individual_evaluators[evaluator] = {}
                    self.individual_evaluators[evaluator][key] = dataset_constructor.individual_evaluators[evaluator][key]
        self.has_individual_evaluators = has_individual_evaluators

        self.prepare_labels(items_to_scale=keys_to_scale)
        if Config()['calculate_kde']:
            with torch.no_grad():
                self.create_kde_labels() # Prepare labels will scale soft labels 
                for k in self.labels:
                    self.labels[k]['act_variance'] = np.var(self.labels[k]['soft_act_labels'])
                    self.labels[k]['val_variance'] = np.var(self.labels[k]['soft_val_labels'])
                if hasattr(self, 'individual_evaluators'):
                    self.variance_from_evaluator_grid()
                    self.variance_from_evaluator_knn()

    def variance_from_evaluator_grid(self):
        # If we want to use the 9 regions as variance uncomment following
        act_grid = {e: {0:{0:[],1:[],2:[]},1:{0:[],1:[],2:[]},2:{0:[],1:[],2:[]}} for e in self.individual_evaluators}
        val_grid = {e: {0:{0:[],1:[],2:[]},1:{0:[],1:[],2:[]},2:{0:[],1:[],2:[]}} for e in self.individual_evaluators}
        mapping = {e: {} for e in self.individual_evaluators}
        for e in tqdm(self.individual_evaluators, desc='calculating evaluator grid variance'):
            for k in self.individual_evaluators[e]:
                act,val = self.individual_evaluators[e][k]['act'], self.individual_evaluators[e][k]['val']
                if act < -1/3:
                    xpos = 0
                elif -1/3 <= act < 1/3:
                    xpos = 1
                else:
                    xpos = 2
                if val < -1/3:
                    ypos = 0
                elif -1/3 <= val < 1/3:
                    ypos = 1
                else:
                    ypos = 2
                act_grid[e][xpos][ypos].append(act)
                val_grid[e][xpos][ypos].append(val)
                mapping[e][k] = (xpos, ypos)
            for x in act_grid[e]:
                for y in act_grid[e][x]:
                    act_grid[e][x][y] = np.var(act_grid[e][x][y])
                    val_grid[e][x][y] = np.var(val_grid[e][x][y])
        for k in self.labels:
            acts = {}
            vals = {}
            for e in self.labels[k]['evaluators']:
                x,y = mapping[e][k]
                acts[e] = act_grid[e][x][y]
                vals[e] = val_grid[e][x][y]
            self.labels[k]['act_variance_grid'] = acts
            self.labels[k]['val_variance_grid'] = vals
            # print('Chosen variance', self.labels[k]['act_variance_knn'], self.labels[k]['val_variance_knn'])

    def variance_from_evaluator_knn(self):
        # If we want to use the k-nearest samples then uncomment the following 
        num_k = 5
        distances = {e: {k: {} for k in self.labels} for e in self.individual_evaluators}
        for e in self.individual_evaluators:
            for k in self.individual_evaluators[e]:
                for l in self.individual_evaluators[e]:
                    if l == k:
                        continue
                    distance = np.sqrt(np.square(self.individual_evaluators[e][l]['act'] - self.individual_evaluators[e][k]['act']) + np.square(self.individual_evaluators[e][l]['val'] - self.individual_evaluators[e][k]['val']))
                    if distance not in distances[e][k]:
                        distances[e][k][distance] = []
                    distances[e][k][distance].append(l)
        # Now pick the 5 nearest samples
        for k in tqdm(self.labels, desc=f'calculating variance from {num_k} nearest samples'):
            evaluator_act_vars = {}
            evaluator_val_vars = {}
            for e in self.labels[k]['evaluators']:
                smallest_distances = sorted(distances[e][k])
                for_calculation = []
                remaining = num_k
                for small_distance in smallest_distances:
                    if remaining <= 0:
                        break
                    choices = distances[e][k][small_distance]
                    num_at_this_distance = len(choices)
                    choose = min(num_at_this_distance, remaining)
                    chosen = np.random.choice(choices, replace=False, size=choose)
                    remaining -= choose
                    for_calculation.extend(chosen)
                acts = [self.individual_evaluators[e][l]['act'] for l in for_calculation]
                vals = [self.individual_evaluators[e][l]['act'] for l in for_calculation]
                acts = np.var(acts) if len(acts) else 0
                vals = np.var(vals) if len(vals) else 0 
                evaluator_act_vars[e] = acts
                evaluator_val_vars[e] = vals
            self.labels[k]['act_variance_knn'] = evaluator_act_vars
            self.labels[k]['val_variance_knn'] = evaluator_val_vars
            # print('Chosen variance', self.labels[k]['act_variance_knn'], self.labels[k]['val_variance_knn'])
    
    def create_kde_labels(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Build the large nan-filled soft_act and soft_val labels for kde calculation
        # First calculate the maximum number of evaluators in the dataset so that other samples can be padded with torch.nan
        if self.has_individual_evaluators:
            max_num_evaluators = max(len(self.labels[sample]['evaluators']) for sample in self.labels)
        else:
            max_num_evaluators = max(len(self.labels[sample]['soft_act_labels']) for sample in self.labels)
        samples = list(self.labels.keys())
        batch_size=256
        total_samples = len(self.labels)
        pbar = tqdm(total=total_samples, desc='Calculating KDE')
        num_batches = np.ceil(total_samples/batch_size)
        for i in range(int(num_batches)):
            start = i*batch_size
            end = min((i+1)*batch_size, total_samples)
            curr_bs = end-start
            batched_examples = samples[start:end]
            soft_act = torch.stack([torch.as_tensor(self.labels[sample]['soft_act_labels'] + [torch.nan]*(max_num_evaluators-len(self.labels[sample]['soft_act_labels'])), device=device).float() for sample in batched_examples], dim=0)
            soft_val = torch.stack([torch.as_tensor(self.labels[sample]['soft_val_labels'] + [torch.nan]*(max_num_evaluators-len(self.labels[sample]['soft_val_labels'])), device=device).float() for sample in batched_examples], dim=0)
            kde_2d_prob = kde_probability_bs(soft_act, soft_val, use_soft_histogram=False, prob_grid_size=self.kde_size, temperature=512, density_grid_size=512, precision=torch.float64)
            negs = kde_2d_prob < 0
            if negs.any():
                raise ValueError(f'Negative values in KDE probability. Largest negative:-{kde_2d_prob[negs].abs().max()}')
            kde_2d_prob = kde_2d_prob.view(curr_bs,-1)# - kde_2d_prob.view(curr_bs,-1).min(dim=-1).values.unsqueeze(dim=-1)
            kde_2d_prob = kde_2d_prob / kde_2d_prob.sum(dim=-1).unsqueeze(dim=-1)
            kde_2d_prob = kde_2d_prob.view(curr_bs,self.kde_size,self.kde_size).float()
            # kde_2d_prob = torch.softmax(kde_2d_prob.view(end-start, -1), dim=-1).view(kde_2d_prob.shape)
            # kde_2d_prob_4x4 = kde_probability_bs(soft_act, soft_val, use_soft_histogram=True, prob_grid_size=4, temperature=512)
            # kde_2d_prob_2x2 = kde_probability_bs(soft_act, soft_val, use_soft_histogram=True, prob_grid_size=2, temperature=512)
            for i, example in enumerate(batched_examples):
                self.labels[example]['kde_2d_probability'] = kde_2d_prob[i].cpu()
                # self.labels[example]['kde_2d_probability_4x4'] = kde_2d_prob_4x4[i].cpu()
                # self.labels[example]['kde_2d_probability_2x2'] = kde_2d_prob_2x2[i].cpu()
            pbar.update(end-start)

    def prepare_labels(self, new_minimum=-1, new_maximum=1, items_to_scale=None):
        if items_to_scale is None:
            raise ValueError('prepare_labels requires items_to_scale to be passed from parent constructor. Define prepare_labels function in constructor and return a list of string of label names that require scaling.')
        # function min-max scales continuous labels and bins categorical labels
        num_bins = self.config['num_labels']
        for value_type in items_to_scale: # Prepare labels only operates on labels since at this stage there are no memory labels stored 
            all_values = [value[value_type] for value in self.labels.values() if value_type in value]
            min_val = min(all_values)
            max_val = max(all_values)
            if value_type == 'act' or value_type == 'val':
                # Also scale soft_[type]_labels as well 
                # Also update them min/max using values found in the soft labels 
                soft_key = f'soft_{value_type}_labels'
                all_soft_values = [x for value in self.labels.values() for x in value[soft_key] if soft_key in value]
                min_val = float(min(min_val, min(all_soft_values)))
                max_val = float(max(max_val, max(all_soft_values)))
                
                if hasattr(self, 'individual_evaluators'):
                    # Also scale the individual evaluator values
                    for evaluator in self.individual_evaluators:
                        for utt_id in self.individual_evaluators[evaluator]:
                            self.individual_evaluators[evaluator][utt_id][value_type] = (new_maximum-new_minimum) * (self.individual_evaluators[evaluator][utt_id][value_type] - min_val)/(max_val - min_val) + new_minimum

            buckets = np.linspace(new_minimum, new_maximum, num=num_bins+1)
            for key in self.split_keys:
                if not key in self.labels:
                    self.labels[key] = {}
                values = self.labels[key]
                if value_type == 'act' or value_type == 'val':
                    # Also scale soft_[type]_labels as well 
                    soft_key = f'soft_{value_type}_labels'
                    if soft_key in values:
                        self.labels[key][soft_key] = [(new_maximum-new_minimum) * (soft_rating - min_val)/(max_val - min_val) + new_minimum for soft_rating in self.labels[key][soft_key]]
                    individual_evaluators_key = f'individual_evaluators_{value_type}'
                    if individual_evaluators_key in values:
                        for evaluator in self.labels[key][individual_evaluators_key]:
                            evaluator_rating = self.labels[key][individual_evaluators_key][evaluator]
                            self.labels[key][individual_evaluators_key][evaluator] = (new_maximum-new_minimum) * (evaluator_rating - min_val)/(max_val - min_val) + new_minimum

                if value_type in values:
                    self.labels[key][value_type] = (new_maximum-new_minimum) * (values[value_type] - min_val)/(max_val - min_val) + new_minimum
                    self.labels[key][f'{value_type}_bin'] = min(np.searchsorted(buckets, self.labels[key][value_type], side='right')-1, num_bins-1)

    def __len__(self):
        return len(self.split_keys)

    def get_item_by_key(self, item_key):
        labels = self.labels[item_key]
        # Return a dictionary of all labels for this item + the current dataset id
        return {**labels, 'item_key': item_key, 'dataset_id': self.dataset_ids[item_key]}

    def __getitem__(self, idx):
        item_key = self.split_keys[idx]
        return self.get_item_by_key(item_key)

    def class_weights(self):
        act_labels = [self.labels[key]['act_bin'] for key in self.split_keys if 'act_bin' in self.labels[key]]
        val_labels = [self.labels[key]['val_bin'] for key in self.split_keys if 'val_bin' in self.labels[key]]
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
        ds = DatasetConstructor.dataset_instances[cls.__name__]
        return ds

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
                self.cleanup()
                return

        assert filter_fn is None or callable(filter_fn), 'If filter_fn is defined it should be a callable function\nreturning True if a key should be kept and False if it should be deleted'

        # self.labels should be of the format {'key': {label: label_value...}}
        self.labels = self.read_labels()

        if filter_fn is not None:
            self.labels = {key: self.labels[key] for key in list(self.labels.keys()) if filter_fn(key)} # Filter the labels based on the filter_fn

        # Features will be added to the labels file
        self.generate_features()

        self._initialised = True

        self.cleanup()

        if dataset_save_location is not None:
            # Save dataset if file path provided
            self.save(dataset_save_location)

    def cleanup(self):
        if hasattr(self, 'wav2vec2_model'):
            del self.wav2vec2_model
        if hasattr(self, 'wav2vec2_feature_extractor'):
            del self.wav2vec2_feature_extractor
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'bert_model'):
            del self.bert_model

        # Finally remove any labels where the soft labels and average label don't agree
        for key in list(self.labels.keys()):
            act, val = round(self.labels[key]['act'], 2), round(self.labels[key]['val'], 2)
            soft_act, soft_val = self.labels[key]['soft_act_labels'], self.labels[key]['soft_val_labels']
            mean_act = round(sum(soft_act)/len(soft_act), 2)
            mean_val = round(sum(soft_val)/len(soft_val), 2)
            if mean_act != act or mean_val != val:
                print(f'Removing {key} as average soft label ({mean_act=} {mean_val=}) does not match actual label ({act=} {val=})')
                del self.labels[key]

    def load(self, load_path):
        changed_config_requires_save = False
        with open(load_path, 'rb') as f:
            loaded_attributes = pickle.load(f)
            for key in self.config:
                if key == 'num_labels':
                    # num_labels is how act/val are binned and can vary without concern 
                    continue # This argument does not need to stay the same
                if 'directory' in key and self.config[key] != loaded_attributes['config'][key]:
                    print('WARNING dataset path mismatch. If new data path does not change stored data ignore this warning.')
                    print('If using an updated dataset, then delete the cached dataset file and regenerate')
                    changed_config_requires_save = True
                    continue
                if key not in loaded_attributes['config']:
                    print('POSSIBLE ERROR: Found a new config that was missing from old config')
                    print('if this new config setting doesn\'t affect cached dataset ignore this error')
                    changed_config_requires_save = True
                    continue
                assert self.config[key] == loaded_attributes['config'][key], f'Value mismatch for data_config.yaml value {key}. Current run wants {self.config[key]}, while the dataset was cached with value {loaded_attributes["config"][key]}'
            for key in loaded_attributes:
                if key == 'config':
                    continue # Don't overwrite the current config file 
                setattr(self, key, loaded_attributes[key])
        if changed_config_requires_save:
            print('To prevent constant warnings printing on each run for non-errors, the config will now be rewritten.')
            print('if you are not sure that the warnings can be ignored delete the cached dataset file and re-run')
            self.save(load_path)

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def read_labels(self):
        raise NotImplementedError('Template dataset constructor called. This class should be inherited and have read_labels return a dictionary of loaded labels and a dictionary of label meta information')

    def get_dataset_splits(self, data_split_type):
        assert type(data_split_type) == str or type(data_split_type) == dict, 'data_split_type should be a string\neither describing the type of split that should be used\nor a filepath to a json file defining splits'
        if type(data_split_type) == dict:
            return data_split_type
        if data_split_type[-5:].lower() == '.json' and os.path.exists(data_split_type):
            return load_json(data_split_type)
        return data_split_type # Whatever inherits this template should override this method and handle the string

    def build(self, data_split_type=None, kde_size=5, test_only=False, remove_evaluators_with_few_evaluations=False, only_keep_evaluators=False, sampling=None, **kwargs):
        if remove_evaluators_with_few_evaluations:
            # Need to remove these bad evaluators before calculation, in this case will also adjust the mean act/val ground truth
            for evaluator in self.too_few_evaluations:
                for utt_id in self.individual_evaluators[evaluator]:
                    self.labels[utt_id]['soft_act_labels'].remove(self.individual_evaluators[evaluator][utt_id]['act'])
                    self.labels[utt_id]['soft_val_labels'].remove(self.individual_evaluators[evaluator][utt_id]['val'])
                    self.labels[utt_id]['evaluators'].remove(evaluator)
                    del self.labels[utt_id]['individual_evaluators_act'][evaluator]
                    del self.labels[utt_id]['individual_evaluators_val'][evaluator]
                    self.labels[utt_id]['act'] = np.mean(self.labels[utt_id]['soft_act_labels'])
                    self.labels[utt_id]['val'] = np.mean(self.labels[utt_id]['soft_val_labels'])
                del self.individual_evaluators[evaluator]

        keys_to_scale = self.prepare_labels()
        split_dict = self.get_dataset_splits(data_split_type, **kwargs)
        if split_dict == 'all': # If using all keys just return the one dataset 
            return DatasetInstance(self, list(self.labels.keys()), keys_to_scale, kde_size=kde_size)

        dataset_split_dict = {}
        for split in split_dict:
            keys = split_dict[split]
            if split == 'train_keys':
                split = 'train'
            if split == 'val_key':
                split = 'val'
            if test_only and split in ['train', 'val']:
                continue
            dataset_split_dict[split] = DatasetInstance(self, keys, keys_to_scale, kde_size=kde_size)
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

        num_workers = 2 # TODO: This value should vary based on the settings -- wav2vec2/bert better with lower number over raw values (more processing = less workers)
        with mp_thread.Pool(num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(self.save_speech_features, list(self.wav_keys_to_use.keys())), total=len(self.wav_keys_to_use.keys()), desc='Saving and extracting speech features'):
                pass
        
        print(f'Removed {len(self.removed_wavs)} wav files for being too short/long')
        print(f'Wav files were {self.removed_wavs}')
        del self.removed_wavs

    def save_speech_features(self, wav_key):
        # First check if the wav key has a label
        if not self.create_new_labels and wav_key not in self.labels:
            del self.wav_keys_to_use[wav_key]
            return

        wav_path = self.wav_keys_to_use[wav_key]
        self.labels[wav_key]['wav_path'] = wav_path
        y, sr = read_wav(wav_path)
        duration = librosa.get_duration(y=y, sr=sr)
        too_short = self.config['min_len'] != -1 and duration < self.config['min_len']
        too_long = self.config['max_len'] != -1 and duration > self.config['max_len']
        if too_short or too_long:
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
        elif type(self.config['audio_feature_type']) == str:
            unpooled_audio = get_w2v2(y, sr, w2v2_extractor=self.wav2vec2_feature_extractor, w2v2_model=self.wav2vec2_model)
            self.labels[wav_key]['audio'] = pool_w2v2(unpooled_audio)
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
        if 'transcript_text' not in self.labels[wav_key]:
            raise NotImplementedError('TODO: Implement ASR')
        if self.config['text_feature_type'] == 'raw':
            self.labels[wav_key]['transcript'] = self.labels[wav_key]['transcript_text']
            del self.labels[wav_key]['transcript_text']
        elif type(self.config['text_feature_type']) == str:
            self.labels[wav_key]['transcript'] = get_bert_embedding(self.labels[wav_key]['transcript_text'], bert_tokenizer=self.tokenizer, bert_model=self.bert_model)
            del self.labels[wav_key]['transcript_text']
        else:
            raise IOError(f"Uknown text feature type {self.config['audio_feature_type']} in data_config.yaml")

    def get_wavs(self):
        raise NotImplementedError('Template dataset constructor called. This class should be inherited and have get_wavs return a dictionary of key: wav_file_path. Key should match up the ID in the labels file.')