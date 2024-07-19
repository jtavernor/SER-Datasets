from .dataset_constructor import DatasetConstructor
from .config import Config
from glob import glob
import os
import re
from tqdm import tqdm
import numpy as np 

class ImprovDatasetConstructor(DatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None):
        self.dataset_name = 'MSP-Improv'
        self.improv_directory = Config()['improv_directory']
        super().__init__(1, filter_fn, dataset_save_location)

    def read_labels(self):
        lab_file = os.path.join(self.improv_directory, "Evaluation.txt")
        evaluation_line_matcher = re.compile(r'UTD-IMPROV-(?P<utt_id>[A-Z0-9\-]+)\.avi;\s+(?P<cat_lbl>\w);\s+A:(?P<act_lbl>\d+\.\d+)\s*;\s+V:(?P<val_lbl>\d+\.\d+)\s*;\s+D:(?P<dom_lbl>\d+\.\d+|NaN)\s*;.*')
        utterance_matcher = re.compile(r'MSP-IMPROV-S(?P<sentence>\d\d)(?P<intended_emotion>[AHSN])-(?P<speaker>(?P<gender>[MF])\d\d)-(?P<scenario>[PRST])-(?P<listener>[FM])(?P<dyadic_speaker>[FM])(?P<turn_number>\d\d)')
        soft_matcher = re.compile(r'(?P<evaluator>[A-Za-z\-0-9_]+);\s(?P<cat_emotion>[A-Za-z]+);\s(?P<soft_emotions>([A-Za-z() \-|/;.?"!:\[\]],?)+|);\sA:(?P<act>[0-9.]+);\sV:(?P<val>[0-9.]+);\sD:(?P<dom>[0-9.]+|NaN);\sN:(?P<naturalness>[0-9.]+|NaN);')
        labels = {}
        self.individual_evaluators = {}

        utterances_with_duplicates = []
        with open(lab_file, 'r') as r:
            current_utt = None
            for line in r.readlines():
                line=line.rstrip()
                if line.startswith('UTD-IMPROV-'):
                    utt_results = evaluation_line_matcher.match(line)
                    if utt_results is None:
                        raise IOError(f'Failed to read values from line: {line}')
                    utt_id = utt_results.group('utt_id')
                    full_utt_id = f'MSP-IMPROV-{utt_id}'
                    current_utt = full_utt_id
                    if full_utt_id not in labels:
                        labels[full_utt_id] = {'soft_act_labels': [], 'soft_val_labels': [], 'evaluators': [], 'individual_evaluators_act': {}, 'individual_evaluators_val': {}, 'naturalness': []}
                    else:
                        raise IOError(f'Encountered duplicate label {full_utt_id}')
                    labels[full_utt_id]['act'] = 6.0 - float(utt_results.group('act_lbl')) # Improv stores activation values high to low (1 to 5), not low to high so we need to flip this so that 0 is the lowest and 4 is the highest.
                    labels[full_utt_id]['val'] = float(utt_results.group('val_lbl'))
                    utt_details = utterance_matcher.match(full_utt_id)
                    labels[full_utt_id]['gender'] = utt_details.group('gender')
                    labels[full_utt_id]['speaker_id'] = utt_details.group('speaker')
                    labels[full_utt_id]['categorical'] = utt_results.group('cat_lbl') # Categorical emotion
                    # dom_lbl = utt_results.group('dom_lbl') # Dominance 
                    # labels[full_utt_id]['dom'] = None if dom_lbl == 'NaN' else float(dom_lbl)
                elif line == '':
                    current_utt = None
                else:
                    # Store the soft labels 
                    matches = soft_matcher.match(line)
                    if matches is None:
                        print(utt_id, line)
                    evaluator = matches.group('evaluator')
                    if evaluator not in self.individual_evaluators:
                        self.individual_evaluators[evaluator] = {}
                    evaluator_act = int(6.0 - float(matches.group('act')))
                    evaluator_val = int(float(matches.group('val')))
                    labels[current_utt]['soft_act_labels'].append(evaluator_act)
                    labels[current_utt]['soft_val_labels'].append(evaluator_val)
                    if evaluator in labels[current_utt]['evaluators']:
                        print('duplicate', evaluator, 'in', current_utt, 'averaging')
                        utterances_with_duplicates.append((current_utt, evaluator))
                        labels[current_utt]['individual_evaluators_act'][evaluator] = [labels[current_utt]['individual_evaluators_act'][evaluator]] + [evaluator_act]
                        labels[current_utt]['individual_evaluators_val'][evaluator] = [labels[current_utt]['individual_evaluators_val'][evaluator]] + [evaluator_val]
                        continue
                    labels[current_utt]['evaluators'].append(evaluator)
                    labels[current_utt]['individual_evaluators_act'][evaluator] = evaluator_act
                    labels[current_utt]['individual_evaluators_val'][evaluator] = evaluator_val
                    labels[current_utt]['naturalness'].append(matches.group('naturalness'))
                    self.individual_evaluators[evaluator][current_utt] = {'act': evaluator_act, 'val': evaluator_val}

        for utt_id, evaluator in set(utterances_with_duplicates):
            print(utt_id, evaluator)
            sub_act = labels[utt_id]['individual_evaluators_act'][evaluator]
            sub_val = labels[utt_id]['individual_evaluators_val'][evaluator]
            print(sub_act, sub_val)
            evaluator_act = np.mean(sub_act).item()
            evaluator_val = np.mean(sub_val).item()
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
            labels[utt_id]['soft_act_labels'].append(evaluator_act)
            labels[utt_id]['soft_val_labels'].append(evaluator_val)
            print(labels[utt_id]['soft_act_labels'], labels[utt_id]['soft_val_labels'])
            labels[utt_id]['individual_evaluators_act'][evaluator] = evaluator_act
            labels[utt_id]['individual_evaluators_val'][evaluator] = evaluator_val
            labels[utt_id]['act'] = np.mean(labels[utt_id]['soft_act_labels']).item()
            labels[utt_id]['val'] = np.mean(labels[utt_id]['soft_val_labels']).item()
            self.individual_evaluators[evaluator][utt_id] = {'act': evaluator_act, 'val': evaluator_val}

        too_few_evaluations = []
        for evaluator in self.individual_evaluators:
            num_evals = len(self.individual_evaluators[evaluator].keys())
            if num_evals < 0:
                too_few_evaluations.append(evaluator)

        self.too_few_evaluations = set(too_few_evaluations) # Use this to amend target labels during training when using new method
        removed_evaluators = len(self.too_few_evaluations)
        for evaluator in tqdm(self.too_few_evaluations, desc='Adjusting labels to remove evaluators with less than 50 evaluations'):
            eval_ratings = self.individual_evaluators[evaluator]
            for utt_id in eval_ratings:
                eval_act = eval_ratings[utt_id]['act']
                eval_val = eval_ratings[utt_id]['val']
                labels[utt_id]['evaluators'].remove(evaluator)
                assert evaluator not in labels[utt_id]['evaluators']
                labels[utt_id]['soft_act_labels'].remove(eval_act)
                labels[utt_id]['soft_val_labels'].remove(eval_val)
                del labels[utt_id]['individual_evaluators_act'][evaluator]
                del labels[utt_id]['individual_evaluators_val'][evaluator]
                # If there are no more evaluators for this utterance remove 
                # Now correct the labels mean values 
                labels[utt_id]['act'] = np.mean(labels[utt_id]['soft_act_labels']).item()
                labels[utt_id]['val'] = np.mean(labels[utt_id]['soft_val_labels']).item()
            del self.individual_evaluators[evaluator]

        # Remove labels that no longer have enough evaluators
        removed_utterances = 0
        removed_utterances2 = 0
        for utt_id in list(labels.keys()):
            if len(labels[utt_id]['evaluators']) < 2:
                removed_utterances += 1
                if len(labels[utt_id]['evaluators']):
                    removed_utterances2 += 1
                del labels[utt_id]
                continue

        print(f'Removed {removed_evaluators} evaluators who annotated less than 50 samples. Removed {removed_utterances} ({removed_utterances2}) utterances that no longer had any annotators (or had 1 annotator).')

        # Now load transcripts for each label 
        for key in list(labels.keys()):
            transcript_file = os.path.join(self.improv_directory, 'Text', f'{key}.txt')
            if not os.path.exists(transcript_file):
                print(f'Could not find transcript for {key}. Removing label.')
                del labels[key]
                continue
            with open(transcript_file, 'r') as f:
                labels[key]['transcript_text'] = f.read()

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
                # ['F1', 'F3', 'F5', 'F6', 'M2', 'M3', 'M4', 'M6'], ['F2', 'F4', 'M1', 'M5']
                # train_keys = [key for key in all_keys if re.search(r'-[FM]0[1234]-', key)]
                # val_keys = [key for key in all_keys if re.search(r'-[FM]05-', key)]
                # test_keys = [key for key in all_keys if re.search(r'-[FM]06-', key)]
                train_keys = [key for key in all_keys if re.search(r'-(F0[1356]|M0[2346])-', key)]
                val_keys = [key for key in all_keys if re.search(r'-(F02|M01)-', key)]
                test_keys = [key for key in all_keys if re.search(r'-(F04|M05)-', key)]
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
            elif split_type == 'all':
                return 'all'
        else:
            raise ValueError(f'Unknown split type {data_split_type}')