from .dataset_constructor import DatasetConstructor
from .config import Config
from glob import glob
import os
import re
import pickle

class PodcastDatasetConstructor(DatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None):
        self.podcast_directory = Config()['podcast_directory']
        super().__init__(2, filter_fn, dataset_save_location)

    def read_labels(self):
        self.podcast_data_split = {'train': [], 'val': [], 'test_set_1': [], 'test_set_2': []}
        lab_file = os.path.join(self.podcast_directory, "Labels", "labels_concensus.csv")
        evaluation_line_matcher = re.compile(r'(?P<utt_id>MSP-PODCAST_[0-9_]*).wav,(?P<cat_lbl>\w),(?P<act_lbl>\d+\.\d+),(?P<val_lbl>\d+\.\d+),(?P<dom_lbl>\d+\.\d+),(?P<spkr_id>\d+|Unknown),(?P<gender>Male|Female|Unknown),(?P<split>Train|Validation|Test1|Test2)')
        df_lst = []
        labels = {}
        with open(lab_file, 'r') as r:
            for line in r.readlines():
                line=line.rstrip()
                if line.startswith('MSP-PODCAST_'):
                    utt_results = evaluation_line_matcher.match(line)
                    if utt_results is None:
                        raise IOError(f'Failed to read values from line: {line}')
                    utt_id = utt_results.group('utt_id')
                    if utt_id not in labels:
                        labels[utt_id] = {}
                    else:
                        raise IOError(f'Encountered duplicate label {utt_id}')
                    labels[utt_id]['act'] = float(utt_results.group('act_lbl')) # Improv stores activation values high to low (1 to 5), not low to high so we need to flip this so that 0 is the lowest and 4 is the highest.
                    labels[utt_id]['val'] = float(utt_results.group('val_lbl'))
                    labels[utt_id]['gender'] = utt_results.group('gender')
                    labels[utt_id]['speaker_id'] = utt_results.group('spkr_id')
                    utt_split = utt_results.group('split')
                    if utt_split == 'Train':
                        self.podcast_data_split['train'].append(utt_id)
                    elif utt_split == 'Validation':
                        self.podcast_data_split['val'].append(utt_id)
                    elif utt_split == 'Test1':
                        self.podcast_data_split['test_set_1'].append(utt_id)
                    elif utt_split == 'Test2':
                        self.podcast_data_split['test_set_2'].append(utt_id)
                    else:
                        raise IOError(f'Uknown split {utt_split} for utterance {utt_id}')
                    # labels[utt_id]['gender'] = utterance_matcher.match(full_utt_id).group('gender')
                    # cat_lbl = utt_results.group('cat_lbl') # Categorical emotion
                    # dom_lbl = utt_results.group('dom_lbl') # Dominance 
                    # labels[full_utt_id]['dom'] = None if dom_lbl == 'NaN' else float(dom_lbl)

        # Now load transcripts for each label 
        transcripts = {}
        for file_path in glob(os.path.join(self.podcast_directory, 'azure_transcripts/*.pk')):
            with open(file_path, 'rb') as f:
                partial_transcripts = pickle.load(f)
                for key in partial_transcripts:
                    transcripts[key] = partial_transcripts[key]
        
        for key in list(labels.keys()):
            if key in transcripts:
                labels[key]['transcript'] = transcripts[key]
            else:
                print('No transcript found for', key)

        return labels

    def prepare_labels(self):
        super().prepare_labels(items_to_scale=['act', 'val'])

    def get_wavs(self):
        all_wavs = glob(os.path.join(self.podcast_directory, 'audio_16kHz', '*.wav'))
        label_id_to_wav = {
            wav_path.split('/')[-1].replace('.wav',''): wav_path for wav_path in all_wavs
        }
        return label_id_to_wav

    def get_dataset_splits(self):
        # MSP-Podcast only supports the default split 
        return self.podcast_data_split