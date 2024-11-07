import os
import torch
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from datasets import load_dataset, concatenate_datasets
from .podcast import read_podcast
from .improv import read_improv
from .iemocap import read_iemocap
from .muse import read_muse
from .config import Config
from .kde_probability import kde_probability_bs
from .utils import scale_dataset

conf = Config()

# Define directories and paths for use in loading
dataset_paths = {
    'podcast': conf['podcast_directory'],
    'improv': conf['improv_directory'],
    'muse': conf['muse_directory'],
    'iemocap': conf['iemocap_directory'],
}
label_paths = {
    'podcast': os.path.join(conf['podcast_directory'], 'Labels', 'labels_consensus.csv'),
    'improv': os.path.join(conf['improv_directory'], 'Evaluation.txt'),
    'muse': os.path.join(conf['muse_directory'], 'SurveyInformation', 'Emotion data from crowdsourcing (C is when annotators had access to all previous sentences).csv'),
    'iemocap': os.path.join(conf['iemocap_directory'], 'IEMOCAP_EmoEvaluation.txt'),
}
# Define the min and maximum values in the data prior to scaling
dataset_scale_parameters = {
    'podcast': (1, 7),
    'improv': (1, 5),
    'iemocap': (1, 5),
    'muse': (1, 9),
}

# MuSE not named consistently so check if path correct
if not os.path.exists(label_paths['muse']):
    label_paths['muse'] = os.path.join(conf['muse_directory'], 'Survey Information (Questions, Data etc)', 'Emotion data from crowdsourcing (C is when annotators had access to all previous sentences).csv')

file_reader = {
    'podcast': read_podcast,
    'improv': read_improv,
    'muse': read_muse,
    'iemocap': read_iemocap,
}

# Method for calculating length from the audio column in huggingface dataset
def add_length(sample):
    sample['AudioLength'] = sample['Audio']['array'].shape[0]/sample['Audio']['sampling_rate']
    return sample

def format_datasets(type_to_columns, column_masks, *datasets):
    # Only the torch type needs to be formatted, the None types are still left as none
    # only one formatting can be applied so it has to be left implicit for the None types
    column_type = 'torch'
    columns_to_set, kwargs = type_to_columns[column_type]
    columns_to_set = columns_to_set[column_masks[column_type]]
    for dataset in datasets:
        # Dataset may not have all available columns -- remove them at this point
        ds_cols_to_set = [col for col in columns_to_set if col in dataset.features.keys()]
        if len(ds_cols_to_set):
            dataset.set_format(column_type, columns=ds_cols_to_set, output_all_columns=True, **kwargs)

def make_audio_datasets(datasets_to_load=['improv', 'iemocap', 'muse', 'podcast'], kde_size=4):
    """
    Creates audio datasets for training, development, and testing from labeled audio files.

    Code requires setting multiprocessing method to spawn prior to call to create features correctly
    from multiprocess import set_start_method
    set_start_method('spawn')
    
    Args:
        audio_dir (str): The directory containing the audio files.
        labels_path (str): The path to the CSV file containing the labels and file information.
        cached_layer (int, optional): If provided then use the cached w2v2 outputs from this layer. Defaults to None, which does not use cached features.

    Returns:
        train_dataset: A huggingface Dataset object containing the training data.
        dev_dataset: A huggingface Dataset object containing the development data.
        test_dataset: A huggingface Dataset object containing the testing data.
    """
    # First load config and calculate which columns will be used based on the config file 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    type_to_columns = { # None returns just plain python objects -- use for dictionaries and strings
        None: (np.array(['FileName', 'Split_Set', 'annotators']), {}),
        'torch': (np.array(['Audio', 'Text', 'act', 'val', 'soft_act_labels', 'soft_val_labels', 'self-report-act', 'self-report-val']), {'dtype': torch.float32, 'device': device})
    }
    column_masks = {
        None: [True, True, conf['return_annotator_info']],
        'torch': [True, True, conf['return_activation'], conf['return_valence'], conf['return_activation'] and conf['return_soft_labels'], conf['return_valence'] and conf['return_soft_labels'], conf['return_activation'] and conf['return_self_report'], conf['return_valence'] and conf['return_self_report']]
    }
    # Apply masks along configs to keep only columns where mask is True
    columns = [x for coltype in type_to_columns for x in type_to_columns[coltype][0][column_masks[coltype]]]
    print('Using columns:', columns)
    
    train_datasets, dev_datasets, test_datasets = {}, {}, {}

    # Load datasets from cache
    if conf['cache_datasets']:
        loaded_keys = []
        for key in datasets_to_load:
            load_paths = {
                'train': os.path.join(conf['cache_dataset_path'], f'{key}_train.parquet'),
                'dev': os.path.join(conf['cache_dataset_path'], f'{key}_dev.parquet'),
                'test': os.path.join(conf['cache_dataset_path'], f'{key}_test.parquet'),
            }
            train_exists = os.path.exists(load_paths['train'])
            dev_exists = os.path.exists(load_paths['dev'])
            test_exists = os.path.exists(load_paths['test'])
            if all([train_exists, dev_exists, test_exists]):
                # hfdataset = load_dataset('parquet', data_files=load_paths)
                train_datasets[key] = load_dataset('parquet', data_files={'train': load_paths['train']})['train']
                dev_datasets[key] = load_dataset('parquet', data_files={'dev': load_paths['dev']})['dev']
                test_datasets[key] = load_dataset('parquet', data_files={'test': load_paths['test']})['test']
                loaded_keys.append(key)
            elif any([train_exists, dev_exists, test_exists]):
                raise IOError(f'Found partial files of dataset. Either remove all files to cause recalculation, or correct paths of missing files. Checked paths: {load_paths}')

    # Remove any datasets that were loaded from cache
    datasets_to_generate = [key for key in datasets_to_load if key not in loaded_keys]
    if len(datasets_to_generate): # Generate datasets from cache
        for key in datasets_to_generate:
            # Load labels for each dataset
            train_datasets[key], dev_datasets[key], test_datasets[key] = file_reader[key](dataset_paths[key], label_paths[key], columns=columns)
            min_v, max_v = dataset_scale_parameters[key]
            train_datasets[key] = train_datasets[key].map(lambda x: scale_dataset(x, min_v, max_v), num_proc=8)
            dev_datasets[key] = dev_datasets[key].map(lambda x: scale_dataset(x, min_v, max_v), num_proc=8)
            test_datasets[key] = test_datasets[key].map(lambda x: scale_dataset(x, min_v, max_v), num_proc=8)

        # Check that features need to be generated for audio and text 
        audio_features, text_features = conf['audio_feature_type'], conf['text_feature_type']
        feature_generation = audio_features != 'raw' or text_features != 'raw'
        if feature_generation: # Create an object that can be called for creating features
            generator = FeatureGenerator(audio_features, text_features)

        # Now create features 
        for key in datasets_to_generate:
            # Calculate audio and text features 
            if feature_generation:
                processes = 4 #if key != 'podcast' else 1 # Podcast sometimes runs out of memory due to size so use less processes
                train_datasets[key] = train_datasets[key].map(generator, num_proc=processes)
                dev_datasets[key] = dev_datasets[key].map(generator, num_proc=processes)
                test_datasets[key] = test_datasets[key].map(generator, num_proc=processes)

            # Set the correct format on the dataset -- has to be done prior to calculation of KDE labels
            format_datasets(type_to_columns, column_masks, train_datasets[key], dev_datasets[key], test_datasets[key])

            # Calculate KDE 2D labels
            if conf['calculate_kde']:
                train_datasets[key] = train_datasets[key].map(lambda x: create_kde_labels_map(x, kde_size=kde_size, num_calculations=conf['num_kde_calculations']), batched=True, batch_size=256)
                dev_datasets[key] = dev_datasets[key].map(lambda x: create_kde_labels_map(x, kde_size=kde_size, num_calculations=conf['num_kde_calculations']), batched=True, batch_size=256)
                test_datasets[key] = test_datasets[key].map(lambda x: create_kde_labels_map(x, kde_size=kde_size, num_calculations=conf['num_kde_calculations']), batched=True, batch_size=256)
            
            for types, typeg in [('train', train_datasets), ('val', dev_datasets), ('test', test_datasets)]:
                print(key, types, 'Activation min and max:', min(typeg[key]['act']), max(typeg[key]['act']))
                print(key, types, 'Valence min and max:', min(typeg[key]['val']), max(typeg[key]['val']))

            # Store datasets            
            if conf['cache_datasets']:
                train_datasets[key].to_parquet(os.path.join(conf['cache_dataset_path'], f'{key}_train.parquet'))
                dev_datasets[key].to_parquet(os.path.join(conf['cache_dataset_path'], f'{key}_dev.parquet'))
                test_datasets[key].to_parquet(os.path.join(conf['cache_dataset_path'], f'{key}_test.parquet'))

                # For some reason after storing datasets to disk the below audio filtering will hang indefinitely
                # not sure if the underlying huggingface code is trying to write later changes to disk as well 
                # so we just reload these datasets immediately to resolve this problem 
                train_datasets[key] = load_dataset('parquet', data_files={'train': os.path.join(conf['cache_dataset_path'], f'{key}_train.parquet')})['train']
                dev_datasets[key] = load_dataset('parquet', data_files={'dev': os.path.join(conf['cache_dataset_path'], f'{key}_dev.parquet')})['dev']
                test_datasets[key] = load_dataset('parquet', data_files={'test': os.path.join(conf['cache_dataset_path'], f'{key}_test.parquet')})['test']

    print('Datasets loaded, filtering on audio length...')
    # Now that datasets are loaded (and possibly cached to disk) apply filtering on audio length
    # Load csv containing audio lengths for all utterances in each dataset
    len_csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'audiolengths.csv')
    if os.path.exists(len_csv_path):
        lengths = pd.read_csv(len_csv_path)
    else: # If audio length csv doesn't exist then create it
        print('Concatenating datasets to calculate audio lengths and generate audiolengths.csv for filtering')
        lengths = concatenate_datasets(list(train_datasets.values()) + list(dev_datasets.values()) + list(test_datasets.values()))
        lengths = lengths.map(add_length, num_proc=16).to_pandas()
        lengths = lengths[['Dataset', 'FileName', 'AudioLength']]
        lengths.to_csv(len_csv_path)

    # Separate lengths by dataset utterance came from 
    lengths_by_dataset = {
        'podcast': lengths[lengths['Dataset'] == 'MSP-Podcast'].set_index('FileName'),
        'improv': lengths[lengths['Dataset'] == 'MSP-Improv'].set_index('FileName'),
        'muse': lengths[lengths['Dataset'] == 'MuSE'].set_index('FileName'),
        'iemocap': lengths[lengths['Dataset'] == 'IEMOCAP'].set_index('FileName'),
    }

    # If no limit provided then we want to not filter on this so set max/min appropriately
    min_audio_len = None if conf['min_len'] == -1 else conf['min_len']
    max_audio_len = None if conf['max_len'] == -1 else conf['max_len']

    if max_audio_len is None:
        max_audio_len = lengths['AudioLength'].max()+1 # Don't want to filter any so set it higher than the max 
    if min_audio_len is None:
        min_audio_len = lengths['AudioLength'].min()-1 # Don't want to filter any so set it lower than the min

    # Now filter dataset and create new
    def filter_len(x):
        lengths = dataset_lengths.loc[x['FileName']]['AudioLength']
        not_too_short = min_audio_len <= lengths
        not_too_long = lengths <= max_audio_len
        audio_exists = lengths > 0 
        return np.logical_and(np.logical_and(not_too_long, not_too_short), audio_exists)

    for key in datasets_to_load:
        dataset_lengths = lengths_by_dataset[key]
        # Remove samples not in the min/max audio length
        train_datasets[key] = train_datasets[key].filter(filter_len, batched=True)
        dev_datasets[key] = dev_datasets[key].filter(filter_len, batched=True)
        test_datasets[key] = test_datasets[key].filter(filter_len, batched=True)

        # Now make sure dataset is in the correct format 
        format_datasets(type_to_columns, column_masks, train_datasets[key], dev_datasets[key], test_datasets[key])

    return train_datasets, dev_datasets, test_datasets

def create_kde_labels_map(batched_examples, kde_size, num_calculations=1):
    for gen in range(num_calculations):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        soft_act = torch.nn.utils.rnn.pad_sequence(batched_examples['soft_act_labels'], batch_first=True, padding_value=torch.nan).to(device, non_blocking=True)
        soft_val = torch.nn.utils.rnn.pad_sequence(batched_examples['soft_val_labels'], batch_first=True, padding_value=torch.nan).to(device, non_blocking=True)
        batch_size = soft_act.shape[0]
        kde_2d_prob = kde_probability_bs(soft_act, soft_val, use_soft_histogram=False, prob_grid_size=kde_size, temperature=512, density_grid_size=512, precision=torch.float64)
        negs = kde_2d_prob < 0
        if negs.any():
            raise ValueError(f'Negative values in KDE probability. Largest negative:-{kde_2d_prob[negs].abs().max()}')
        kde_2d_prob = kde_2d_prob.view(batch_size,-1)# - kde_2d_prob.view(curr_bs,-1).min(dim=-1).values.unsqueeze(dim=-1)
        kde_2d_prob = kde_2d_prob / kde_2d_prob.sum(dim=-1).unsqueeze(dim=-1)
        kde_2d_prob = kde_2d_prob.view(batch_size,kde_size,kde_size).float()
        batched_examples[f'kde_2d_probability_generation_{gen}'] = kde_2d_prob.cpu()
    if num_calculations == 1:
        batched_examples['kde_2d_probability'] = batched_examples['kde_2d_probability_generation_0']
        del batched_examples['kde_2d_probability_generation_0']
    return batched_examples

class FeatureGenerator:
    def __init__(self, audio_features, text_features):
        self.audio_features = audio_features
        self.text_features = text_features
        self.initialised = False

    def __call__(self, sample):
        if not self.initialised:
            from transformers.utils import logging # Silence warnings as model loads result in repeated warnings per process
            logging.set_verbosity_error() 
            if self.audio_features != 'raw' and self.audio_features != 'mfb' and type(self.audio_features) == str:
                self.wav2vec2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.audio_features)
                self.wav2vec2_model = Wav2Vec2Model.from_pretrained(self.audio_features)
                if torch.cuda.is_available():
                    self.wav2vec2_model = self.wav2vec2_model.to('cuda')
                self.wav2vec2_model.eval()

            if self.text_features != 'raw' and type(self.text_features) == str:
                self.tokenizer = AutoTokenizer.from_pretrained(self.text_features)
                self.bert_model = AutoModel.from_pretrained(self.text_features)
                if torch.cuda.is_available():
                    self.bert_model = self.bert_model.to('cuda')
            self.initialised = True

        with torch.no_grad():
            if sample['Audio']['sampling_rate'] != 16000:
                raise ValueError('Sampling rate should be 16000')
            if len(sample['Audio']['array']):
                audio_features = self.wav2vec2_feature_extractor(sample['Audio']['array'], sampling_rate=16000, return_tensors='pt')
                if torch.cuda.is_available():
                    audio_features = audio_features.to('cuda')
                assert len(audio_features) == 1
                audio_features = self.wav2vec2_model(**audio_features)['last_hidden_state']

                unpooled_audio = audio_features.cpu().numpy()
                audio_features_pooled = torch.mean(torch.as_tensor(unpooled_audio), dim=1)
                audio_features_pooled = audio_features_pooled.squeeze(dim=0).cpu().numpy()
            else:
                print('ERROR: Empty wav file -- returning empty tensor for w2v2 features')
                audio_features_pooled = torch.tensor([]).squeeze()
            bert_tokens = self.tokenizer.encode_plus(sample['Text'], add_special_tokens=True, return_tensors='pt')
            # if torch.cuda.is_available():
                # bert_tokens = bert_tokens.to('cuda')
            bert_tokens = bert_tokens.to(self.bert_model.device)
            out = self.bert_model(**bert_tokens).last_hidden_state
            cls_tok = out.squeeze()[0].cpu()

        sample['Audio'] = audio_features_pooled
        sample['Text'] = cls_tok
        return sample

class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.dataset_to_id = {'MSP-Podcast': 0, 'MSP-Improv': 1, 'MuSE': 2, 'IEMOCAP': 3}

    def __call__(self, batch):
        if not hasattr(self, 'using_cache'):
            self.using_cache = 'W2V2CachedLayer' in batch[0]
        labels_act = [sample['EmoAct'] for sample in batch]
        labels_val = [sample['EmoVal'] for sample in batch]
        transcripts = [sample['Text'] for sample in batch]
        dataset_ids = [self.dataset_to_id[sample['Dataset']] for sample in batch]

        if not self.using_cache:
            audios = [torch.from_numpy(sample['Audio']['array']) for sample in batch]
            # Pad to longest seq length in the batch
            max_len = max([len(a) for a in audios])
            audios = [torch.nn.functional.pad(a, (0, max_len - len(a))) for a in audios]
            inputs = self.processor(audios, sampling_rate=16000, padding=True, return_tensors='pt').input_values[0]
        else:
            inputs = torch.nn.utils.rnn.pad_sequence([sample['W2V2CachedLayer'].squeeze() for sample in batch], batch_first=True)

        labels_act = torch.tensor(labels_act)
        labels_val = torch.tensor(labels_val)
        # labels = torch.stack([labels_act, labels_val], dim=1)
        return inputs, transcripts, dataset_ids, labels_act, labels_val

def get_dataloaders(multidomain_trainining=True, datasets_to_load=['podcast', 'improv', 'iemocap', 'muse'], kde_size=4):
    print('Warning -- only use get dataloaders when loading raw audio as it uses a collator assuming padding raw audio')
    train_datasets, dev_datasets, test_datasets = make_audio_datasets(datasets_to_load, kde_size)
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    if multidomain_trainining:
        # Train datasets and dev datasets should be merged into one dataset 
        train_dataset = concatenate_datasets(train_datasets.values())
        dev_dataset = concatenate_datasets(dev_datasets.values())
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=8,
            collate_fn=Collator(processor),
        )
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=Collator(processor),
        )
    else:
        train_dataloader = {
            key: DataLoader(
                train_datasets[key],
                batch_size=32,
                shuffle=True,
                num_workers=8,
                collate_fn=Collator(processor),
            ) for key in train_datasets
        }
        dev_dataloader = {
            key: DataLoader(
                dev_datasets[key],
                batch_size=32,
                shuffle=True,
                num_workers=8,
                collate_fn=Collator(processor),
            ) for key in dev_datasets
        }
    test_dataloaders = {
        key: DataLoader(
            test_datasets[key],
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=Collator(processor),
        ) for key in test_datasets
    }
    return train_dataloader, dev_dataloader, test_dataloaders