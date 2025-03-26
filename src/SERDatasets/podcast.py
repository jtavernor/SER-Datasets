import os
import re
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob
from datasets import Dataset, Audio, concatenate_datasets
import whisper

def read_podcast(dataset_dir, labels_path, columns, podcast_v='1.11'):
    if podcast_v in ['1.6', '1.7', '1.8', '1.9', '1.10']:
        # Do not need to track files that have been renamed -- none of these occur in 1.6 and newer
        # label directory is named for 1.6=label, 1.7=Labels, 1.8=Labels, 1.9=labels, 1.10=Labels
        if podcast_v == '1.6':
            label_dir = 'label'
        elif podcast_v == '1.9':
            label_dir = 'labels'
        else:
            label_dir = 'Labels'

        # 1.10 corrected the spelling of consensus from concensus 
        if podcast_v == '1.10':
            consensus_file_name = 'consensus'
        else:
            consensus_file_name = 'concensus'

        # Change the label path to load this versions labels
        labels_path = os.path.join(dataset_dir, f'Previous_releases_information/previous_releases_info/v{podcast_v}/{label_dir}/labels_{consensus_file_name}.csv')
        detailed_lab_file = labels_path.replace(consensus_file_name, 'detailed')
        # Finally need to also provide audio paths for labels that have been removed in podcast 1.11
        # This will be done in the audio loading later
    elif podcast_v == '1.11':
        # Do nothing other than define detailed lab file
        detailed_lab_file = labels_path.replace('consensus', 'detailed')
    else:
        # Podcast 1.0-1.5 use a different label file so easier to not implement for now until needed 
        raise ValueError(f'Only implemented loading for podcast versions >= 1.6')

    labels = pd.read_csv(labels_path)
    if podcast_v != '1.11':
        counts = {'found': 0, 'removed': 0}
    def get_audio_path(x):
        in_v1_11_path = os.path.join(dataset_dir, 'Audios', x)
        if os.path.exists(in_v1_11_path):
            return in_v1_11_path
        counts['removed'] += 1
        in_removed_path = os.path.join(dataset_dir,'Previous_releases_information', 'files_removed_from_prev_versions', f'v{podcast_v}_removed_files', x)
        # print('IS IN THE REMOVED SET?', x.replace('.wav','') in removed_set)
        if os.path.exists(in_removed_path):
            counts['found'] += 1
            return in_removed_path
        # raise ValueError(f'Could not find audio path for {x} in podcast v{podcast_v} (checked {in_v1_11_path} and {in_removed_path})')
        print(f'Could not find audio path for {x} in podcast v{podcast_v} (checked {in_v1_11_path} and {in_removed_path})')
        return np.nan # Can now use dropna to remove these values

    labels['Audio'] = labels['FileName'].map(get_audio_path)
    if podcast_v != '1.11':
        print(f'Of audio files not present in 1.11 found {counts["found"]}/{counts["removed"]}')
    labels = labels.rename(columns={'EmoAct': 'act', 'EmoVal': 'val', 'EmoDom': 'dom', 'Gender': 'gender', 'SpkrID': 'speaker_id'})

    # Now load the soft labels for act/val/dom
    detailed = pd.read_csv(detailed_lab_file)
    if podcast_v == '1.6':
        detailed = detailed.rename(columns={'Workers': 'annotators', 'EmoClass_Major': 'cat_emotions', 'EmoClass_Second': 'soft_emotions', 'EmoAct': 'soft_act_labels', 'EmoVal': 'soft_val_labels', 'EmoDom': 'soft_dom_labels'})
    else:
        extracted = detailed['EmoDetail'].str.extract(r'(?P<annotators>WORKER\d+);\s(?P<cat_emotions>[A-Za-z() \-|/;.?"!:\[\]&,\s\d_]+);\s(?P<soft_emotions>(?:[A-Za-z() \-|/;.?"!:\[\]&\s\d],?)+|);\sA:(?P<soft_act_labels>[0-9.]+);\sV:(?P<soft_val_labels>[0-9.]+);\sD:(?P<soft_dom_labels>[0-9.]+);')
        detailed = pd.concat([detailed, extracted], axis=1).drop(columns=['EmoDetail'])
    for key in ['act', 'val', 'dom']:
        detailed[f'soft_{key}_labels'] = pd.to_numeric(detailed[f'soft_{key}_labels'])
    detailed = detailed.groupby('FileName').agg(list).reset_index()
    assert sorted(labels['FileName']) == sorted(detailed['FileName'])
    labels = pd.merge(labels, detailed, on='FileName')
    print('pre drop', len(labels))
    labels = labels.dropna() # Remove files that are missing audio labels
    print('post drop', len(labels))

    # Now load transcripts for each label 
    transcript_dir = os.path.join(dataset_dir, 'Transcripts')
    if podcast_v != '1.11':
        transcript_dir_gen = os.path.join(dataset_dir, 'Previous_releases_information', 'files_removed_from_prev_versions', 'transcripts')

    model_obj = {'model': None, 'printed_warning': False} # Will initialise one instance of whisper rather than reinitialising if needed
    def read_transcript(x):
        t_path = os.path.join(transcript_dir, f'{x.replace(".wav", "")}.txt')
        if os.path.exists(t_path):
            with open(t_path, 'r') as f:
                text = f.read().strip()
        else:
            print('No transcript found for', x)
            if podcast_v != '1.11':
                if not model_obj['printed_warning']:
                    print('Old podcast -- using whisper to transcribe')
                    model_obj['printed_warning'] = True
                t_path_gen = os.path.join(transcript_dir_gen, f'{x.replace(".wav", "")}.txt')
                if os.path.exists(t_path_gen):
                    with open(t_path_gen, 'r') as f:
                        text = f.read().strip()
                else:
                    if model_obj['model'] is None:
                        model_obj['model'] = whisper.load_model('turbo', device='cuda')
                    model = model_obj['model']
                    result = model.transcribe(labels[labels['FileName'] == x]['Audio'].item())
                    text = result['text']
                    print('Whisper transcription:', text)
                with open(t_path_gen, 'w') as f:
                    f.write(text)

        return text

    labels['Text'] = labels['FileName'].map(read_transcript)

    labels['Dataset'] = 'MSP-Podcast'
    missing = [col for col in columns if col not in labels.columns]
    columns = [col for col in columns if col in labels.columns]
    if len(missing): # TODO: Better way of doing this rather than copying from muse
        # Missing are due to new columns generated during caching (AudioFeatures) not present in pre-processed dataset
        print('Warning MSP-Podcast returning empty columns for:', missing)
    labels = labels[columns]

    train_df = labels[labels['Split_Set'] == 'Train']
    if ('Development' == labels['Split_Set']).any():
        dev_df = labels[labels['Split_Set'] == 'Development']
    else:
        # print(labels['Split_Set'].unique())
        assert ('Validation' == labels['Split_Set']).any()
        dev_df = labels[labels['Split_Set'] == 'Validation']
    test_df = labels[labels['Split_Set'] == 'Test1']

    return train_df, dev_df, test_df
