from dataset_constructor import DatasetConstructor
from config import Config
from glob import glob
import os
import re
import pickle

class MuSEDatasetConstructor(DatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None):
        self.muse_directory = Config()['muse_directory']
        super().__init__(0, filter_fn, dataset_save_location)

# The forced alignment code isn't actually necessary as IEMOCAP has aligned word timings already 
    def read_labels(self):
        labels = {}
        lab_file = os.path.join(self.muse_directory, "Survey Information (Questions, Data etc)", "Emotion data from crowdsourcing (C is when annotators had access to all previous sentences).csv")
        get_utt_id = re.compile(r'(?P<utt_id>.*).wav')
        with open(lab_file, 'r') as r:
            for i, line in enumerate(r.readlines()):
                if i == 0: # Skip the first line with headers
                    continue
                line=line.rstrip()
                Sentence_name,Duration,Gender,Type,Ques_Type,Activation_Mean,Activation_SD,Valence_Mean,Valence_SD,Activation_Annotation,Valence_Annotation,C_Activation_Mean,C_Activation_SD,C_Valence_Mean,C_Valence_SD,C_Activation_Annotation,C_Valence_Annotation,S_Activation,S_Valence = line.split(',')
                utt_id = get_utt_id.match(Sentence_name).group('utt_id')
                transcript = self.read_transcript(utt_id)
                if transcript is None:
                    continue
                if utt_id not in labels:
                    labels[utt_id] = {}
                else:
                    raise IOError(f'Encountered duplicate label {utt_id}')
                labels[utt_id]['act'] = float(Activation_Mean)
                labels[utt_id]['val'] = float(Valence_Mean)
                labels[utt_id]['transcript'] = transcript
                labels[utt_id]['gender'] = Gender
                labels[utt_id]['type'] = Type # S == stressed, NS == non_stressed

        return labels

    def read_transcript(self, utt_id):
        transcript_file = os.path.join(self.muse_directory, 'Combined', 'Utterance Transcripts', f'{utt_id}.txt')
        transcript = None
        if not os.path.isfile(transcript_file):
            print(f'Warning - no transcript found for {utt_id}, skipping.')
            return None
        with open(transcript_file, 'r') as f:
            transcript = f.read()
        
        return transcript

    def prepare_labels(self):
        super().prepare_labels(items_to_scale=['act', 'val'])

    def get_wavs(self):
        non_stressed_wavs = glob(os.path.join(self.muse_directory, 'Non Stressed', 'Audio', 'Non Stressed Question Monologue Audio*', '*.wav'))
        stressed_wavs = glob(os.path.join(self.muse_directory, 'Stressed', 'Audio', 'Stressed Question Monologue Audio', '*.wav'))
        label_id_to_wav = {
            wav_path.split('/')[-1].replace('.wav', ''): wav_path for wav_path in non_stressed_wavs
        }
        label_id_to_wav_stressed = {
            wav_path.split('/')[-1].replace('.wav', ''): wav_path for wav_path in stressed_wavs
        }
        assert set(label_id_to_wav.keys()) - set(label_id_to_wav_stressed.keys()) == set(label_id_to_wav.keys())
        assert set(label_id_to_wav_stressed.keys()) - set(label_id_to_wav.keys()) == set(label_id_to_wav_stressed.keys())
        for key in label_id_to_wav_stressed:
            label_id_to_wav[key] = label_id_to_wav_stressed[key]

        return label_id_to_wav

    def get_dataset_splits(self):
        # MuSE only supports a stressed/non-stressed split
        split = {
            'stressed': [key for key in self.labels if self.labels[key]['type'] == 'S'],
            'non_stressed': [key for key in self.labels if self.labels[key]['type'] == 'NS'],
        }
        assert set(self.labels.keys()) == set(split['stressed'] + split['non_stressed'])
        return split