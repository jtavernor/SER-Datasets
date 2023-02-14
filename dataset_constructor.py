from .config import Config
from .utils import load_json

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

    def __init__(self, dataset_id, data_split_type=None, filter_fn=None, dataset_save_location=None):
        # If this is a singleton class that was already initialised then don't load data again
        if self._initialised:
            return

        if dataset_save_location is not None:
            # Load dataset if file exists 

        self.data_split_type = data_split_type

        assert filter_fn is None or callable(filter_fn), 'If filter_fn is defined it should be a callable function\nreturning True if a key should be kept and False if it should be deleted'

        self.config = Config()

        # self.labels should be of the format {'key': {label: label_value...}}
        self.labels, self.labels_info = self.read_labels()

        self.prepare_labels()

        if self.filter_fn is not None:
            pass # Filter the labels based on the filter_fn

        # Features will be added to the labels file
        self.generate_features()

    def read_labels(self):
        raise NotImplementedError('Template dataset constructor called. This class should be inherited and have read_labels return a dictionary of loaded labels and a dictionary of label meta information')

    def prepare_labels(self):
        # function min-max scales continuous labels and bins categorical labels
        pass

    def get_dataset_splits(self):
        assert type(self.data_split_type) == str, 'data_split_type should be a string\neither describing the type of split that should be used\nor a filepath to a json file defining splits'
        if self.data_split_type[-5:].lower() == '.json' and os.path.exists(self.data_split_type):
            return load_json(self.data_split_type)
        return self.data_split_type # Whatever inherits this template should override this method and handle the string

    def build(self):
        split_dict = self.get_dataset_splits()
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

        for _ in tqdm(pool.imap_unordered(self.read_audio_features, self.wav_keys_to_use.keys()), total=len(self.wav_keys_to_use.keys()), desc='Saving audio features'):
            pass
        
        pool.close()
        pool.join()

    def read_audio_features(self, wav_key):
        # First check if the wav key has a label
        if not self.create_new_labels and wav_key not in self.labels:
            del self.wav_keys_to_use[wav_key]
            return

        wav_path = self.wav_keys_to_use[wav_key]
        y, sr = read_wav(wav_path)
        duration = get_duration(y, sr)
        if duration < self.config['min_len'] or duration > self.config['max_len']:
            del self.wav_keys_to_use[wav_key]
            if wav_key in self.labels:
                del self.labels[wav_key]

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
