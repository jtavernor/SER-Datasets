import os
import re
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob
from datasets import Dataset, Audio, concatenate_datasets
from .utils import scale_dataset, read_transcript

def read_podcast(dataset_dir, labels_path, columns, podcast_v='1.11'):
    detailed_lab_file = labels_path.replace('consensus', 'detailed')
    evaluation_line_matcher = re.compile(r'(?P<utt_id>MSP-PODCAST_[0-9_]*).wav,(?P<cat_lbl>\w),(?P<act_lbl>\d+\.\d+),(?P<val_lbl>\d+\.\d+),(?P<dom_lbl>\d+\.\d+),(?P<spkr_id>\d+|Unknown),(?P<gender>Male|Female|Unknown),(?P<split>Train|Validation|Development|Test1|Test2)')
    soft_matcher = re.compile(r'(?P<utt_id>MSP-PODCAST_[0-9_]*).wav,"?(?P<annotator>WORKER\d+);\s(?P<cat_emotion>[A-Za-z() \-|/;.?"!:\[\]&,\s\d_]+);\s(?P<soft_emotions>([A-Za-z() \-|/;.?"!:\[\]&\s\d],?)+|);\sA:(?P<act>[0-9.]+);\sV:(?P<val>[0-9.]+);\sD:(?P<dom>[0-9.]+);"?')
    labels = {}
    individual_annotators = {}
    with open(labels_path, 'r') as r:
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
            if annotator not in individual_annotators:
                individual_annotators[annotator] = {}
            individual_annotators[annotator][m.group('utt_id')] = {'act': int(float(m.group('act'))), 'val': int(float(m.group('val')))}

    for utt_id, annotator in set(utterances_with_duplicates):
        # print(utt_id, annotator)
        sub_act = labels[utt_id]['individual_annotators_act'][annotator]
        sub_val = labels[utt_id]['individual_annotators_val'][annotator]
        # print(sub_act, sub_val)
        annotator_act = np.mean(sub_act).item()
        annotator_val = np.mean(sub_val).item()
        curr_len = len(labels[utt_id]['soft_act_labels'])
        # print(labels[utt_id]['soft_act_labels'], labels[utt_id]['soft_val_labels'])
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
        # print(labels[utt_id]['soft_act_labels'], labels[utt_id]['soft_val_labels'])
        labels[utt_id]['individual_annotators_act'][annotator] = annotator_act
        labels[utt_id]['individual_annotators_val'][annotator] = annotator_val
        labels[utt_id]['act'] = np.mean(labels[utt_id]['soft_act_labels']).item()
        labels[utt_id]['val'] = np.mean(labels[utt_id]['soft_val_labels']).item()
        individual_annotators[annotator][utt_id] = {'act': annotator_act, 'val': annotator_val}

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

    # Converting to pandas dataframe takes a long time 
    # Instead go directly to huggingface
    # hfdataset = Dataset.from_list(list(labels.values()))
    # crash
    # Convert piece by piece 
    print('Converting to pandas')
    train_samples = [l for l in labels.values() if l['Split_Set'] == 'Train']
    dev_samples = [l for l in labels.values() if l['Split_Set'] == 'Development']
    test_samples = [l for l in labels.values() if l['Split_Set'] == 'Test1']
    # print(len(train_samples), len(dev_samples), len(test_samples))
    print('Converting to pandas')
    temp = pd.DataFrame(test_samples)
    temp['Dataset'] = 'MSP-Podcast'
    temp = temp[columns]
    # print('Converting from pandas')
    test_dataset = Dataset.from_pandas(temp).cast_column('Audio', Audio(sampling_rate=16000, mono=True))
    # print('Converting to pandas')
    temp = pd.DataFrame(dev_samples)
    temp['Dataset'] = 'MSP-Podcast'
    temp = temp[columns]
    # print('Converting from pandas')
    dev_dataset = Dataset.from_pandas(temp).cast_column('Audio', Audio(sampling_rate=16000, mono=True))
    # print('Converting to pandas')
    temp = pd.DataFrame(train_samples)
    temp['Dataset'] = 'MSP-Podcast'
    temp = temp[columns]
    # print('Converting from pandas')
    train_dataset = Dataset.from_pandas(temp).cast_column('Audio', Audio(sampling_rate=16000, mono=True))

    # chunk_size = 25000
    # to_chunk = list(labels.values())
    # datasets = {'train': [], 'dev': [], 'test': []}
    # for chunk_idx in tqdm(range(int(np.ceil(len(to_chunk)/chunk_size))), desc='Converting chunks of labels to pandas (infeasibly slow using full podcast)'):
    #     start = chunk_idx * chunk_size
    #     end = min((chunk_idx+1)*chunk_size, len(labels))
    #     labels_df = pd.DataFrame(to_chunk[start:end])
    #     labels_df['Dataset'] = 'MSP-Podcast'
    #     labels_df = labels_df[columns]

    #     train_df = labels_df[labels_df['Split_Set'] == 'Train']
    #     dev_df = labels_df[labels_df['Split_Set'] == 'Development']
    #     test_df = labels_df[labels_df['Split_Set'] == 'Test1']

    #     train_dataset = Dataset.from_pandas(train_df).cast_column('Audio', Audio(sampling_rate=16000, mono=True))
    #     dev_dataset = Dataset.from_pandas(dev_df).cast_column('Audio', Audio(sampling_rate=16000, mono=True))
    #     test_dataset = Dataset.from_pandas(test_df).cast_column('Audio', Audio(sampling_rate=16000, mono=True))

    #     datasets['train'].append(train_dataset)
    #     datasets['dev'].append(dev_dataset)
    #     datasets['test'].append(test_dataset)

    # train_dataset = concatenate_datasets(datasets['train'])
    # dev_dataset = concatenate_datasets(datasets['dev'])
    # test_dataset = concatenate_datasets(datasets['test'])

    train_dataset = train_dataset.map(scale_dataset, num_proc=1)
    dev_dataset = dev_dataset.map(scale_dataset, num_proc=1)
    test_dataset = test_dataset.map(scale_dataset, num_proc=1)

    # assert len(to_chunk) == len(train_dataset) + len(dev_dataset) + len(test_dataset)

    return train_dataset, dev_dataset, test_dataset
