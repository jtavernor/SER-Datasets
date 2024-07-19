from .dataset_constructor import DatasetConstructor
from .config import Config
from glob import glob
import os
import re
import numpy as np

class IEMOCAPDatasetConstructor(DatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None):
        self.dataset_name = 'IEMOCAP'
        self.iemocap_directory = Config()['iemocap_directory']
        super().__init__(2, filter_fn, dataset_save_location)

    def read_labels(self):
        label_file = os.path.join(self.iemocap_directory, 'IEMOCAP_EmoEvaluation.txt')
        get_evaluator_scores = re.compile(r'.*val\s+([1-5]\.?\d?);\s+act\s+([1-5]\.?\d?);\s+dom\s+([1-5]\.?\d?);.*')
        label_info = {}
        labels = {}
        with open(label_file, 'r') as file:
            section = []
            for line in file:
                # Skip the first line
                if 'TURN_NAME' in line: continue
                line = line.rstrip()
                if line == '' and len(section):
                    # Empty line - end of this section, store in the label info
                    session_id = section[0].split('\t')[1]
                    label_info[session_id] = section
                    section = []
                elif line != '':
                    section.append(line)
        
        # We now just need to process each of the sections and also load the transcript for each id
        for label_id in label_info:
            if label_id in labels:
                raise IOError(f'Multiple labels for the same speech {label_id}')
            labels[label_id] = {'soft_act_labels': [], 'soft_val_labels': []}
            for line in label_info[label_id]:
                # First line contains averaged labels
                if line.startswith('[') and line.endswith(']'):
                    cat_lbl = line.split("\t")[2]
                    val_lbl = float(line.split("\t")[3][1:-1].split(", ")[0])
                    act_lbl = float(line.split("\t")[3][1:-1].split(", ")[1])
                    dom_lbl = float(line.split("\t")[3][1:-1].split(", ")[2])
                    labels[label_id]['act'] = act_lbl
                    labels[label_id]['val'] = val_lbl
                    # labels[label_id]['dom'] = dom_lbl
                    labels[label_id]['gender'] = re.match(r'.*(?P<gender>[FM])\d+$', label_id).group('gender')
                    session = re.match(r'^Ses(?P<session>\d\d).*$', label_id).group('session')
                    labels[label_id]['speaker_id'] = f'{session}{labels[label_id]["gender"]}'
                elif line.startswith('A-E'):
                    # Attribute perception of other evaluator
                    regex_match = get_evaluator_scores.match(line)
                    if regex_match:
                        labels[label_id]['soft_val_labels'].append(int(regex_match.group(1)))
                        labels[label_id]['soft_act_labels'].append(int(regex_match.group(2)))
                    else:
                        # Bad label that is not in the range 1-5 or is just blank 
                        print(f'Bad label {label_id}: {line}')
                        del labels[label_id]
                        break
                elif line.startswith('A-F') or line.startswith('A-M'):
                    # Attribute perception of self evaluator
                    if 'self-report' in labels[label_id]:
                        raise IOError(f'Found multiple self-report scores for {label_id}.')
                    regex_match = get_evaluator_scores.match(line)
                    if regex_match:
                        labels[label_id]['self-report-val'] = float(regex_match.group(1))
                        labels[label_id]['self-report-act'] = float(regex_match.group(2))
                        # labels[label_id]['self-report-dom'] = float(regex_match.group(3))
                    else:
                        # Bad label that is not in the range 1-5 or is just blank 
                        print(f'Bad label {label_id}: {line}')
                        del labels[label_id]
                elif line.startswith('C-'):
                    # All categorical based evaluations
                    pass
                else:
                    raise IOError('Unknown line in IEMOCAP label file - cannot process:', line)

        # Now read the transcripts
        transcript_files = glob(os.path.join(self.iemocap_directory, '**/dialog/transcriptions/*.txt'))
        for transcript_file in transcript_files:
            with open(transcript_file, 'r') as file:
                for line in file:
                    line = line.rstrip()
                    split_line = line.split()
                    utt_id, transcript = split_line[0], ' '.join(split_line[2:])
                    if utt_id in labels:
                        labels[utt_id]['transcript_text'] = transcript
                    else:
                        print(f'No label found for transcript {utt_id} in file {transcript_file}')

        return labels

    def prepare_labels(self):
        return ['act', 'val']

    def get_wavs(self):
        all_wavs = glob(os.path.join(self.iemocap_directory, '**/sentences/wav/**/*.wav'))
        label_id_to_wav = {
            wav_path.split('/')[-1].replace('.wav',''): wav_path for wav_path in all_wavs
        }
        return label_id_to_wav

    def get_dataset_splits(self, data_split_type, perception_of_self_only=False):
        if perception_of_self_only:
            self.dataset_id = -self.dataset_id * 1000 # 1000 multiplication to remove any chance of overlap with other dataset ids
        else:
            self.dataset_id = self.dataset_id
        split_type = super().get_dataset_splits(data_split_type)
        all_keys = list(self.labels.keys())
        if perception_of_self_only:
            percep_self_keys = []
            for key in all_keys:
                if 'self-report-act' in self.labels[key]:
                    assert 'self-report-val' in self.labels[key]
                    percep_self_keys.append(key)
            all_keys = percep_self_keys
        if type(split_type) == dict:
            return split_type
        elif type(split_type) == str:
            # Now we want to define custom data splits 
            if split_type == 'speaker-independent':
                # Training is session 1-3
                # Validation is session 4
                # Testing is session 5
                train_keys = [key for key in all_keys if re.match(r'^Ses0[123].*$', key)]
                val_keys = [key for key in all_keys if re.match(r'^Ses04.*$', key)]
                test_keys = [key for key in all_keys if re.match(r'^Ses05.*$', key)]
                test05m_keys = [key for key in all_keys if re.match(r'^Ses05.*M\d+$', key)]
                test05f_keys = [key for key in all_keys if re.match(r'^Ses05.*F\d+$', key)]
                speaker_ind = {
                    'train': train_keys,
                    'val': val_keys,
                    'test_05m': test05m_keys,
                    'test_05f': test05f_keys,
                    'test_05_full': test_keys,
                }
                # For self-report some of these may be empty so remove in that case 
                for key in list(speaker_ind.keys()):
                    if len(speaker_ind[key]) == 0:
                        del speaker_ind[key]
                return speaker_ind
            elif split_type == 'speaker-independent-2':
                # for perception of self there are a lot less keys 
                # 598 for session 1
                # 305 for session 2 (only F speech)
                # 900 for session 3
                # 0 for session 4
                # 182 for session 5 (only M speech) 
                # So, to keep it as close as possible to the original we will have non-speaker independent validation, and use only 05m for testing
                train_keys = [key for key in all_keys if re.match(r'^Ses0[12].*$', key)]
                ses03f_keys = [key for key in all_keys if re.match(r'^Ses03.*F\d+$', key)]
                ses03m_keys = [key for key in all_keys if re.match(r'^Ses03.*M\d+$', key)]
                test05m_keys = [key for key in all_keys if re.match(r'^Ses05.*M\d+$', key)]
                np.random.shuffle(ses03f_keys)
                np.random.shuffle(ses03m_keys)
                val_keys = ses03f_keys[:len(ses03f_keys)//2] + ses03m_keys[:len(ses03m_keys)//2]
                train_keys = train_keys + ses03f_keys[len(ses03f_keys)//2:] + ses03m_keys[len(ses03m_keys)//2:]
                speaker_ind = {
                    'train': train_keys,
                    'val': val_keys,
                    'test_05m': test05m_keys
                }
                return speaker_ind
            elif split_type == 'no-lexical-repeat':
                raise NotImplementedError('No lexical repeat not yet implemented')
            elif split_type == 'speaker-split':
                speaker_split = {}
                for session in range(1,6):
                    for gender in ['M', 'F']:
                        speaker_regex = rf'^Ses0{session}.*{gender}\d+$'
                        speaker_split[f'Session0{session}{gender}'] = [key for key in all_keys if re.match(speaker_regex, key)]
                return speaker_split
            elif split_type == 'sessions':
                return {f'Session0{session}': [key for key in all_keys if re.match(f'^Ses0{session}.*$', key)] for session in range(1,6)}
            elif split_type == 'all':
                return 'all'
        else:
            raise ValueError(f'Unknown split type {data_split_type}')