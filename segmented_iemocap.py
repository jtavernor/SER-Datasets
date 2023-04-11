import os
import glob
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

    def get_wavs(self):
        all_wavs = glob(os.path.join(self.segmented_iemo_directory, '**/sentences/wav/**/*.wav'))
        label_id_to_wav = {
            wav_path.split('/')[-1].replace('.wav',''): wav_path for wav_path in all_wavs
        }
        return label_id_to_wav

class SegmentedNoisyIEMOCAPDatasetConstructor(SegmentedIEMOCAPDatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None):
        Config()['segmented_iemocap_directory'] = Config()['segmented_noisy_iemocap_directory']
        super().__init__(filter_fn, dataset_save_location)
        self.dataset_id = 100