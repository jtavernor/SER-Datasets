from .dataset_constructor import DatasetConstructor
from .config import Config
from glob import glob
import os
import re

class ImprovDatasetConstructor(DatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None):
        self.improv_directory = Config()['improv_directory']
        super().__init__(1, filter_fn, dataset_save_location)

    def read_labels(self):
        lab_file = os.path.join(self.improv_directory, "Evaluation.txt")
        evaluation_line_matcher = re.compile(r'UTD-IMPROV-(?P<utt_id>[A-Z0-9\-]+)\.avi;\s+(?P<cat_lbl>\w);\s+A:(?P<act_lbl>\d+\.\d+)\s*;\s+V:(?P<val_lbl>\d+\.\d+)\s*;\s+D:(?P<dom_lbl>\d+\.\d+|NaN)\s*;.*')
        utterance_matcher = re.compile(r'MSP-IMPROV-S(?P<sentence>\d\d)(?P<intended_emotion>[AHSN])-(?P<speaker>(?P<gender>[MF])\d\d)-(?P<scenario>[PRST])-(?P<listener>[FM])(?P<dyadic_speaker>[FM])(?P<turn_number>\d\d)')
        df_lst = []
        labels = {}
        with open(lab_file, 'r') as r:
            for line in r.readlines():
                line=line.rstrip()
                if line.startswith('UTD-IMPROV-'):
                    utt_results = evaluation_line_matcher.match(line)
                    if utt_results is None:
                        raise IOError(f'Failed to read values from line: {line}')
                    utt_id = utt_results.group('utt_id')
                    full_utt_id = f'MSP-IMPROV-{utt_id}'
                    if full_utt_id not in labels:
                        labels[full_utt_id] = {}
                    else:
                        raise IOError(f'Encountered duplicate label {full_utt_id}')
                    labels[full_utt_id]['act'] = 5.0 - float(utt_results.group('act_lbl')) # Improv stores activation values high to low (1 to 5), not low to high so we need to flip this so that 0 is the lowest and 4 is the highest.
                    labels[full_utt_id]['val'] = float(utt_results.group('val_lbl'))
                    utt_details = utterance_matcher.match(full_utt_id)
                    labels[full_utt_id]['gender'] = utt_details.group('gender')
                    labels[full_utt_id]['speaker_id'] = utt_details.group('speaker')
                    # cat_lbl = utt_results.group('cat_lbl') # Categorical emotion
                    # dom_lbl = utt_results.group('dom_lbl') # Dominance 
                    # labels[full_utt_id]['dom'] = None if dom_lbl == 'NaN' else float(dom_lbl)

        # Now load transcripts for each label 
        for key in list(labels.keys()):
            transcript_file = os.path.join(self.improv_directory, 'Text', f'{key}.txt')
            if not os.path.exists(transcript_file):
                print(f'Could not find transcript for {key}. Removing label.')
                del labels[key]
                continue
            with open(transcript_file, 'r') as f:
                labels[key]['transcript'] = f.read()

        return labels

    def prepare_labels(self):
        return ['act', 'val']

    def get_wavs(self):
        all_wavs = glob(os.path.join(self.improv_directory, 'Audio/**/**/**/*.wav'))
        label_id_to_wav = {
            wav_path.split('/')[-1].replace('.wav',''): wav_path for wav_path in all_wavs
        }
        return label_id_to_wav

    def get_dataset_splits(self, data_split_type):
        split_type = super().get_dataset_splits(data_split_type)
        all_keys = list(self.labels.keys())
        if type(split_type) == dict:
            return split_type
        elif type(split_type) == str:
            # Now we want to define custom data splits 
            if split_type == 'speaker-independent':
                # Training is session 1-4
                # Validation is session 5
                # Testing is session 6
                # Improv sessions have speakers M01 F01 for session1, M02 F02 for session2 as so on
                train_keys = [key for key in all_keys if re.search(r'-[FM]0[1234]-', key)]
                val_keys = [key for key in all_keys if re.search(r'-[FM]05-', key)]
                test_keys = [key for key in all_keys if re.search(r'-[FM]06-', key)]
                speaker_ind = {
                    'train': train_keys,
                    'val': val_keys,
                    'test': test_keys,
                }
                return speaker_ind
            elif split_type == 'no-lexical-repeat':
                raise NotImplementedError('No lexical repeat not yet implemented')
            elif split_type == 'speaker-split':
                speaker_split = {}
                for session in range(1,7):
                    for gender in ['M', 'F']:
                        speaker_regex = rf'-{gender}0{session}-'
                        speaker_split[f'Speaker{gender}0{session}'] = [key for key in all_keys if re.search(speaker_regex, key)]
                return speaker_split
        else:
            raise ValueError(f'Unknown split type {data_split_type}')