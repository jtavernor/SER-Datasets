import re
import os
import pandas as pd
import numpy as np
from glob import glob
from datasets import Dataset, Audio
from .utils import scale_dataset, read_transcript

def read_improv(dataset_dir, labels_path, columns):
    all_wavs = glob(os.path.join(dataset_dir, 'Audio/**/**/**/*.wav'))
    label_id_to_wav = {
        wav_path.split('/')[-1].replace('.wav',''): wav_path for wav_path in all_wavs
    }

    evaluation_line_matcher = re.compile(r'UTD-IMPROV-(?P<utt_id>[A-Z0-9\-]+)\.avi;\s+(?P<cat_lbl>\w);\s+A:(?P<act_lbl>\d+\.\d+)\s*;\s+V:(?P<val_lbl>\d+\.\d+)\s*;\s+D:(?P<dom_lbl>\d+\.\d+|NaN)\s*;.*')
    utterance_matcher = re.compile(r'MSP-IMPROV-S(?P<sentence>\d\d)(?P<intended_emotion>[AHSN])-(?P<speaker>(?P<gender>[MF])\d\d)-(?P<scenario>[PRST])-(?P<listener>[FM])(?P<dyadic_speaker>[FM])(?P<turn_number>\d\d)')
    soft_matcher = re.compile(r'(?P<annotator>[A-Za-z\-0-9_]+);\s(?P<cat_emotion>[A-Za-z]+);\s(?P<soft_emotions>([A-Za-z() \-|/;.?"!:\[\]],?)+|);\sA:(?P<act>[0-9.]+);\sV:(?P<val>[0-9.]+);\sD:(?P<dom>[0-9.]+|NaN);\sN:(?P<naturalness>[0-9.]+|NaN);')
    labels = {}
    individual_annotators = {}

    utterances_with_duplicates = []
    with open(labels_path, 'r') as r:
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
                    labels[full_utt_id] = {'soft_act_labels': [], 'soft_val_labels': [], 'annotators': [], 'individual_annotators_act': {}, 'individual_annotators_val': {}, 'naturalness': []}
                else:
                    raise IOError(f'Encountered duplicate label {full_utt_id}')
                labels[full_utt_id]['act'] = 6.0 - float(utt_results.group('act_lbl')) # Improv stores activation values high to low (1 to 5), not low to high so we need to flip this so that 0 is the lowest and 4 is the highest.
                labels[full_utt_id]['val'] = float(utt_results.group('val_lbl'))
                labels[full_utt_id]['dom'] = float(utt_results.group('dom_lbl'))
                utt_details = utterance_matcher.match(full_utt_id)
                labels[full_utt_id]['gender'] = utt_details.group('gender')
                labels[full_utt_id]['speaker_id'] = utt_details.group('speaker')
                labels[full_utt_id]['categorical'] = utt_results.group('cat_lbl') # Categorical emotion
                labels[full_utt_id]['Audio'] = label_id_to_wav[full_utt_id]
                if not os.path.exists(labels[full_utt_id]['Audio']):
                    print('Warning no audio found for:', full_utt_id, labels[full_utt_id]['Audio'])
                    del labels[full_utt_id]
                    continue
                if re.search(r'-(F0[1356]|M0[2346])-', full_utt_id):
                    split = 'Train'
                elif re.search(r'-(F02|M01)-', full_utt_id):
                    split = 'Validation'
                elif re.search(r'-(F04|M05)-', full_utt_id):
                    split = 'Test'
                else:
                    raise ValueError(f'Unknown split for {full_utt_id}')
                labels[full_utt_id]['Split_Set'] = split
                labels[full_utt_id]['FileName'] = f'{full_utt_id}.wav'
                labels[full_utt_id]['Dataset'] = 'MSP-Improv'
            elif line == '':
                current_utt = None
            else:
                # Store the soft labels 
                matches = soft_matcher.match(line)
                if matches is None:
                    print(utt_id, line)
                annotator = matches.group('annotator')
                if annotator not in individual_annotators:
                    individual_annotators[annotator] = {}
                annotator_act = int(6.0 - float(matches.group('act')))
                annotator_val = int(float(matches.group('val')))
                labels[current_utt]['soft_act_labels'].append(annotator_act)
                labels[current_utt]['soft_val_labels'].append(annotator_val)
                if annotator in labels[current_utt]['annotators']:
                    print('duplicate', annotator, 'in', current_utt, 'averaging')
                    utterances_with_duplicates.append((current_utt, annotator))
                    labels[current_utt]['individual_annotators_act'][annotator] = [labels[current_utt]['individual_annotators_act'][annotator]] + [annotator_act]
                    labels[current_utt]['individual_annotators_val'][annotator] = [labels[current_utt]['individual_annotators_val'][annotator]] + [annotator_val]
                    continue
                labels[current_utt]['annotators'].append(annotator)
                labels[current_utt]['individual_annotators_act'][annotator] = annotator_act
                labels[current_utt]['individual_annotators_val'][annotator] = annotator_val
                labels[current_utt]['naturalness'].append(matches.group('naturalness'))
                individual_annotators[annotator][current_utt] = {'act': annotator_act, 'val': annotator_val}

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
        individual_annotators[annotator][utt_id] = {'act': annotator_act, 'val': annotator_val}

    too_few_evaluations = []
    for annotator in individual_annotators:
        num_evals = len(individual_annotators[annotator].keys())
        if num_evals < 0:
            too_few_evaluations.append(annotator)

    # Now load transcripts for each label 
    for key in list(labels.keys()):
        transcript_file = os.path.join(dataset_dir, 'Text', f'{key}.txt')
        if not os.path.exists(transcript_file):
            print(f'Could not find transcript for {key}. Removing label.')
            del labels[key]
            continue
        with open(transcript_file, 'r') as f:
            labels[key]['Text'] = f.read()


    labels_df = pd.DataFrame(labels.values())
    labels_df = labels_df[columns]

    train_df = labels_df[labels_df['Split_Set'] == 'Train']
    dev_df = labels_df[labels_df['Split_Set'] == 'Validation']
    test_df = labels_df[labels_df['Split_Set'] == 'Test']

    train_dataset = Dataset.from_pandas(train_df).cast_column('Audio', Audio(sampling_rate=16000, mono=True))
    dev_dataset = Dataset.from_pandas(dev_df).cast_column('Audio', Audio(sampling_rate=16000, mono=True))
    test_dataset = Dataset.from_pandas(test_df).cast_column('Audio', Audio(sampling_rate=16000, mono=True))
    train_dataset = train_dataset.map(lambda x: scale_dataset(x, minv=1, maxv=5), num_proc=8)
    dev_dataset = dev_dataset.map(lambda x: scale_dataset(x, minv=1, maxv=5), num_proc=8)
    test_dataset = test_dataset.map(lambda x: scale_dataset(x, minv=1, maxv=5), num_proc=8)
    return train_dataset, dev_dataset, test_dataset
