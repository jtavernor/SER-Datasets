from .dataset_constructor import DatasetConstructor
from .config import Config
from glob import glob
import os
import re
import pickle
from tqdm import tqdm
import numpy as np
import random

class PodcastDatasetConstructor(DatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None, num_evaluations=None):
        self.dataset_name = 'MSP-Podcast'
        self.num_evaluations = num_evaluations
        self.podcast_directory = Config()['podcast_directory']
        super().__init__(0, filter_fn, dataset_save_location)

    def read_labels(self):
        self.podcast_data_split = {'train': [], 'val': [], 'test_set_1': [], 'test_set_2': []}
        lab_file = os.path.join(self.podcast_directory, "Labels", "labels_concensus.csv")
        self.podcast_v = '1.8'
        if not os.path.exists(lab_file):
            # Old podcast has a typo in label file name, if it doesn't exist then we are loading new podcast
            lab_file = os.path.join(self.podcast_directory, 'Labels', 'labels_consensus.csv')
            self.podcast_v = '1.11'
        detailed_lab_file = os.path.join(self.podcast_directory, "Labels", "labels_detailed.csv")
        evaluation_line_matcher = re.compile(r'(?P<utt_id>MSP-PODCAST_[0-9_]*).wav,(?P<cat_lbl>\w),(?P<act_lbl>\d+\.\d+),(?P<val_lbl>\d+\.\d+),(?P<dom_lbl>\d+\.\d+),(?P<spkr_id>\d+|Unknown),(?P<gender>Male|Female|Unknown),(?P<split>Train|Validation|Development|Test1|Test2)')
        soft_matcher = re.compile(r'(?P<utt_id>MSP-PODCAST_[0-9_]*).wav,"?(?P<annotator>WORKER\d+);\s(?P<cat_emotion>[A-Za-z() \-|/;.?"!:\[\]&,\s\d_]+);\s(?P<soft_emotions>([A-Za-z() \-|/;.?"!:\[\]&\s\d],?)+|);\sA:(?P<act>[0-9.]+);\sV:(?P<val>[0-9.]+);\sD:(?P<dom>[0-9.]+);"?')
        df_lst = []
        labels = {}
        self.individual_annotators = {}
        with open(lab_file, 'r') as r:
            for line in r.readlines():
                line=line.rstrip()
                if line.startswith('MSP-PODCAST_'):
                    utt_results = evaluation_line_matcher.match(line)
                    if utt_results is None:
                        raise IOError(f'Failed to read values from line: {line}')
                    utt_id = utt_results.group('utt_id')
                    if utt_id not in labels:
                        labels[utt_id] = {'soft_act_labels': [], 'soft_val_labels': [], 'annotators': [], 'individual_annotators_act': {}, 'individual_annotators_val': {}}
                    else:
                        raise IOError(f'Encountered duplicate label {utt_id}')
                    labels[utt_id]['act'] = float(utt_results.group('act_lbl')) # Improv stores activation values high to low (1 to 5), not low to high so we need to flip this so that 0 is the lowest and 4 is the highest.
                    labels[utt_id]['val'] = float(utt_results.group('val_lbl'))
                    labels[utt_id]['gender'] = utt_results.group('gender')
                    labels[utt_id]['speaker_id'] = utt_results.group('spkr_id')
                    utt_split = utt_results.group('split')
                    if utt_split == 'Train':
                        self.podcast_data_split['train'].append(utt_id)
                    elif utt_split == 'Validation' or utt_split == 'Development':
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

        # Now load the soft labels for act/val/dom
        utterances_with_duplicates = []
        with open(detailed_lab_file, 'r') as r:
            for line in r.readlines():
                line=line.rstrip()
                if line == 'FileName,EmoDetail':
                    continue # Skip first line
                # print(line)
                m = soft_matcher.match(line)
                if m is None:
                    raise IOError(f'Unable to match soft labels in line: {line}')
                labels[m.group('utt_id')]['soft_act_labels'].append(int(float(m.group('act'))))
                labels[m.group('utt_id')]['soft_val_labels'].append(int(float(m.group('val'))))
                annotator = m.group('annotator')
                if annotator in labels[m.group('utt_id')]['annotators']:
                    print('duplicate', annotator, 'in', m.group('utt_id'), 'averaging')
                    utterances_with_duplicates.append((m.group('utt_id'), annotator))
                    if type(labels[m.group('utt_id')]['individual_annotators_act'][annotator]) != list:
                        labels[m.group('utt_id')]['individual_annotators_act'][annotator] = [labels[m.group('utt_id')]['individual_annotators_act'][annotator]]
                        labels[m.group('utt_id')]['individual_annotators_val'][annotator] = [labels[m.group('utt_id')]['individual_annotators_val'][annotator]]

                    labels[m.group('utt_id')]['individual_annotators_act'][annotator].append(int(float(m.group('act'))))
                    labels[m.group('utt_id')]['individual_annotators_val'][annotator].append(int(float(m.group('val'))))
                    continue
                labels[m.group('utt_id')]['annotators'].append(annotator)
                labels[m.group('utt_id')]['individual_annotators_act'][annotator] = int(float(m.group('act')))
                labels[m.group('utt_id')]['individual_annotators_val'][annotator] = int(float(m.group('val')))
                # labels[m.group('utt_id')]['soft_dom_label'].append(float(m.group('dom')))
                if annotator not in self.individual_annotators:
                    self.individual_annotators[annotator] = {}
                self.individual_annotators[annotator][m.group('utt_id')] = {'act': int(float(m.group('act'))), 'val': int(float(m.group('val')))}

        for utt_id, annotator in set(utterances_with_duplicates):
            print(utt_id, annotator)
            sub_act = labels[utt_id]['individual_annotators_act'][annotator]
            sub_val = labels[utt_id]['individual_annotators_val'][annotator]
            print(sub_act, sub_val)
            annotator_act = np.mean(sub_act).item()
            annotator_val = np.mean(sub_val).item()
            curr_len = len(labels[utt_id]['soft_act_labels'])
            print(labels[utt_id]['soft_act_labels'], labels[utt_id]['soft_val_labels'])
            for act in sub_act:
                labels[utt_id]['soft_act_labels'].remove(act)
                curr_len -= 1
                assert len(labels[utt_id]['soft_act_labels']) == curr_len
            curr_len = len(labels[utt_id]['soft_val_labels'])
            for val in sub_val:
                labels[utt_id]['soft_val_labels'].remove(val)
                curr_len -= 1
                assert len(labels[utt_id]['soft_val_labels']) == curr_len
            labels[utt_id]['soft_act_labels'].append(annotator_act)
            labels[utt_id]['soft_val_labels'].append(annotator_val)
            print(labels[utt_id]['soft_act_labels'], labels[utt_id]['soft_val_labels'])
            labels[utt_id]['individual_annotators_act'][annotator] = annotator_act
            labels[utt_id]['individual_annotators_val'][annotator] = annotator_val
            labels[utt_id]['act'] = np.mean(labels[utt_id]['soft_act_labels']).item()
            labels[utt_id]['val'] = np.mean(labels[utt_id]['soft_val_labels']).item()
            self.individual_annotators[annotator][utt_id] = {'act': annotator_act, 'val': annotator_val}

        # Now load transcripts for each label 
        transcripts = {}
        if self.podcast_v == '1.8':
            for file_path in glob(os.path.join(self.podcast_directory, 'azure_transcripts/*.pk')):
                with open(file_path, 'rb') as f:
                    partial_transcripts = pickle.load(f)
                    for key in partial_transcripts:
                        transcripts[key] = partial_transcripts[key]
            for key in list(labels.keys()):
                if key in transcripts:
                    labels[key]['transcript_text'] = ' '.join([word[0] for word in transcripts[key]['features']])
                else:
                    print('No transcript found for', key)
        else:
            assert self.podcast_v == '1.11'
            transcript_dir = os.path.join(self.podcast_directory, 'Transcripts')
            for key in list(labels.keys()):
                t_path = os.path.join(transcript_dir, f'{key}.txt')
                if os.path.exists(t_path):
                    with open(t_path, 'r') as f:
                        labels[key]['transcript_text'] = f.read().strip()
                else:
                    print('No transcript found for', key)

        return labels

    def prepare_labels(self):
        return ['act', 'val']

    def get_wavs(self):
        if self.podcast_v == '1.8':
            all_wavs = glob(os.path.join(self.podcast_directory, 'audio_16kHz', '*.wav'))
        else:
            assert self.podcast_v == '1.11'
            all_wavs = glob(os.path.join(self.podcast_directory, 'Audios', '*.wav'))
        label_id_to_wav = {
            wav_path.split('/')[-1].replace('.wav',''): wav_path for wav_path in all_wavs
        }
        return label_id_to_wav

    def get_dataset_splits(self, data_split_type=None):
        # MSP-Podcast only supports the default split
        # ensure no labels are contained in the data split that were later removed for length 
        if data_split_type == 'all':
            return 'all' # Allow getting full dataset, otherwise just return the default splits
        return_split = {
            'train': [key for key in self.podcast_data_split['train'] if key in self.labels.keys()],
            'val': [key for key in self.podcast_data_split['val'] if key in self.labels.keys()],
            'test_set_1': [key for key in self.podcast_data_split['test_set_1'] if key in self.labels.keys()],
            'test_set_2': [key for key in self.podcast_data_split['test_set_2'] if key in self.labels.keys()],
        }
        return return_split