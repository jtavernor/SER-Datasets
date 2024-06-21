import os
import re
from glob import glob
from .iemocap import IEMOCAPDatasetConstructor
from .config import Config
from .utils import load_pk

class SegmentedIEMOCAPDatasetConstructor(IEMOCAPDatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None):
        self.segmented_iemo_directory = Config()['segmented_iemocap_directory']
        super().__init__(filter_fn, dataset_save_location)
        self.dataset_id = 99

    def read_labels(self):
        labels = super().read_labels()
        cached_segments = load_pk(os.path.join(self.segmented_iemo_directory, 'cache.pk'))
        new_labels = {}
        for original_label in cached_segments:
            for segment_wav_name in cached_segments[original_label]:
                transcript, word_times, confidence = cached_segments[original_label][segment_wav_name]
                new_labels[segment_wav_name.replace('.wav', '')] = {
                    **labels[original_label],
                    'transcript': ' '.join([w[0] for w in transcript])
                }

        return new_labels

    def get_dataset_splits(self, data_split_type):
        # Use grandparent super to avoid using iemocap dataset splits when the item is a string
        split_type = super(IEMOCAPDatasetConstructor, self).get_dataset_splits(data_split_type)
        all_keys = list(self.labels.keys())
        if type(split_type) == dict:
            return split_type
        elif type(split_type) == str:
            # Now we want to define custom data splits 
            if split_type == 'full':
                return {'full': all_keys}
            elif split_type == 'speaker-split':
                speaker_split = {}
                for session in range(1,6):
                    for speaker in ['M', 'F']:
                        speaker_split[f'0{session}{speaker}'] = [key for key in all_keys if re.match(rf'^Ses0{session}[MF].*{speaker}.*$', key)]
                        if len(speaker_split[f'0{session}{speaker}']) < 1:
                            del speaker_split[f'0{session}{speaker}']
                return speaker_split
        else:
            raise ValueError(f'Unknown split type {data_split_type}')

    def get_wavs(self):
        all_wavs = glob(os.path.join(self.segmented_iemo_directory, '*.wav'))
        label_id_to_wav = {
            wav_path.split('/')[-1].replace('.wav',''): wav_path for wav_path in all_wavs
        }
        return label_id_to_wav

class SegmentedNoisyIEMOCAPDatasetConstructor(IEMOCAPDatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None):
        self.segmented_noisy_iemocap_directory = Config()['segmented_noisy_iemocap_directory']
        super().__init__(filter_fn, dataset_save_location)
        self.dataset_id = 100

    def read_labels(self):
        labels = super().read_labels()
        cached_segments = load_pk(os.path.join(self.segmented_noisy_iemocap_directory, 'cache.pk'))
        new_labels = {}
        for original_label in cached_segments:
            iemocap_label = '_'.join(original_label.split('_')[1:])
            for segment_wav_name in cached_segments[original_label]:
                transcript, word_times, confidence = cached_segments[original_label][segment_wav_name]
                new_labels[segment_wav_name.replace('.wav', '')] = {
                    **labels[iemocap_label],
                    'transcript': ' '.join([w[0] for w in transcript])
                }

        return new_labels

    def get_dataset_splits(self, data_split_type):
        # Use grandparent super to avoid using iemocap dataset splits when the item is a string
        split_type = super(IEMOCAPDatasetConstructor, self).get_dataset_splits(data_split_type)
        all_keys = list(self.labels.keys())
        if type(split_type) == dict:
            return split_type
        elif type(split_type) == str:
            # Now we want to define custom data splits 
            if split_type == 'full':
                return {'full': all_keys}
            elif split_type == 'speaker-split':
                speaker_split = {}
                for session in range(1,6):
                    for speaker in ['M', 'F']:
                        speaker_split[f'0{session}{speaker}'] = [key for key in all_keys if re.match(rf'^Ses0{session}[MF].*{speaker}.*$', key)]
                        if len(speaker_split[f'0{session}{speaker}']) < 1:
                            del speaker_split[f'0{session}{speaker}']
                return speaker_split
        else:
            raise ValueError(f'Unknown split type {data_split_type}')

    def get_wavs(self):
        all_wavs = glob(os.path.join(self.segmented_noisy_iemocap_directory, '*.wav'))
        label_id_to_wav = {
            wav_path.split('/')[-1].replace('.wav',''): wav_path for wav_path in all_wavs
        }
        return label_id_to_wav