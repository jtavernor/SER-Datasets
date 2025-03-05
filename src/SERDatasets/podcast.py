import os
import re
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob
from datasets import Dataset, Audio, concatenate_datasets

def read_podcast(dataset_dir, labels_path, columns, podcast_v='1.11'):
    detailed_lab_file = labels_path.replace('consensus', 'detailed')
    evaluation_line_matcher = re.compile(r'(?P<utt_id>MSP-PODCAST_[0-9_]*).wav,(?P<cat_lbl>\w),(?P<act_lbl>\d+\.\d+),(?P<val_lbl>\d+\.\d+),(?P<dom_lbl>\d+\.\d+),(?P<spkr_id>\d+|Unknown),(?P<gender>Male|Female|Unknown),(?P<split>Train|Validation|Development|Test1|Test2)')
    soft_matcher = re.compile(r'(?P<utt_id>MSP-PODCAST_[0-9_]*).wav,"?(?P<annotator>WORKER\d+);\s(?P<cat_emotion>[A-Za-z() \-|/;.?"!:\[\]&,\s\d_]+);\s(?P<soft_emotions>([A-Za-z() \-|/;.?"!:\[\]&\s\d],?)+|);\sA:(?P<act>[0-9.]+);\sV:(?P<val>[0-9.]+);\sD:(?P<dom>[0-9.]+);"?')
    labels = {}
    with open(labels_path, 'r') as r:
        for line in r.readlines():
            line=line.rstrip()
            if line.startswith('MSP-PODCAST_'):
                utt_results = evaluation_line_matcher.match(line)
                if utt_results is None:
                    raise IOError(f'Failed to read values from line: {line}')
                utt_id = utt_results.group('utt_id')
                if utt_id not in labels:
                    labels[utt_id] = {'soft_act_labels': [], 'soft_val_labels': [], 'soft_dom_labels': [], 'annotators': []}
                else:
                    raise IOError(f'Encountered duplicate label {utt_id}')
                labels[utt_id]['FileName'] = f'{utt_id}.wav'
                labels[utt_id]['Audio'] = os.path.join(dataset_dir, 'Audios', labels[utt_id]['FileName'])
                labels[utt_id]['act'] = float(utt_results.group('act_lbl')) # Improv stores activation values high to low (1 to 5), not low to high so we need to flip this so that 0 is the lowest and 4 is the highest.
                labels[utt_id]['val'] = float(utt_results.group('val_lbl'))
                labels[utt_id]['gender'] = utt_results.group('gender')
                labels[utt_id]['speaker_id'] = utt_results.group('spkr_id')
                utt_split = utt_results.group('split')
                if utt_split == 'Train':
                    split = 'Train'
                elif utt_split == 'Validation' or utt_split == 'Development':
                    split = 'Development'
                elif utt_split == 'Test1':
                    split = 'Test1'
                elif utt_split == 'Test2':
                    split = 'Test2'
                else:
                    raise IOError(f'Uknown split {utt_split} for utterance {utt_id}')
                labels[utt_id]['Split_Set'] = split
                cat_lbl = utt_results.group('cat_lbl') # Categorical emotion
                dom_lbl = utt_results.group('dom_lbl') # Dominance 
                labels[utt_id]['dom'] = np.NaN if dom_lbl == 'NaN' else float(dom_lbl)
                labels[utt_id]['categorical'] = cat_lbl

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
            labels[m.group('utt_id')]['soft_dom_labels'].append(int(float(m.group('dom'))))
            annotator = m.group('annotator')
            if annotator in labels[m.group('utt_id')]['annotators']:
                print('Note: duplicate', annotator, 'in', m.group('utt_id'), 'NOT averaging, includes duplicates in output')
                # utterances_with_duplicates.append((m.group('utt_id'), annotator))
                # if type(labels[m.group('utt_id')]['individual_annotators_act'][annotator]) != list:
                #     labels[m.group('utt_id')]['individual_annotators_act'][annotator] = [labels[m.group('utt_id')]['individual_annotators_act'][annotator]]
                #     labels[m.group('utt_id')]['individual_annotators_val'][annotator] = [labels[m.group('utt_id')]['individual_annotators_val'][annotator]]

                # labels[m.group('utt_id')]['individual_annotators_act'][annotator].append(int(float(m.group('act'))))
                # labels[m.group('utt_id')]['individual_annotators_val'][annotator].append(int(float(m.group('val'))))
                # continue
            labels[m.group('utt_id')]['annotators'].append(annotator)

    # Now load transcripts for each label 
    transcripts = {}
    if podcast_v == '1.8':
        for file_path in glob(os.path.join(dataset_dir, 'azure_transcripts/*.pk')):
            with open(file_path, 'rb') as f:
                partial_transcripts = pickle.load(f)
                for key in partial_transcripts:
                    transcripts[key] = partial_transcripts[key]
        for key in list(labels.keys()):
            if key in transcripts:
                labels[key]['Text'] = ' '.join([word[0] for word in transcripts[key]['features']])
            else:
                print('No transcript found for', key)
    else:
        assert podcast_v == '1.11'
        transcript_dir = os.path.join(dataset_dir, 'Transcripts')
        for key in list(labels.keys()):
            t_path = os.path.join(transcript_dir, f'{key}.txt')
            if os.path.exists(t_path):
                with open(t_path, 'r') as f:
                    labels[key]['Text'] = f.read().strip()
            else:
                print('No transcript found for', key)

    labels_df = pd.DataFrame(labels.values())
    labels_df['Dataset'] = 'MSP-Podcast'
    missing = [col for col in columns if col not in labels_df.columns]
    columns = [col for col in columns if col in labels_df.columns]
    if len(missing): # TODO: Better way of doing this rather than copying from muse
        # Missing are due to new columns generated during caching (AudioFeatures) not present in pre-processed dataset
        print('Warning MSP-Podcast returning empty columns for:', missing)
    labels_df = labels_df[columns]

    train_df = labels_df[labels_df['Split_Set'] == 'Train']
    dev_df = labels_df[labels_df['Split_Set'] == 'Development']
    test_df = labels_df[labels_df['Split_Set'] == 'Test1']

    return train_df, dev_df, test_df
