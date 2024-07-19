from .dataset_constructor import DatasetConstructor
from .config import Config
from glob import glob
import os
import re
import pickle

class MuSEDatasetConstructor(DatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None):
        self.dataset_name = 'MuSE'
        self.muse_directory = Config()['muse_directory']
        self.use_whisper = Config()['use_whisper_for_muse']
        super().__init__(3, filter_fn, dataset_save_location)

    def read_labels(self):
        labels = {}
        # Versions of MuSE on different machines have different names 
        if os.path.exists(os.path.join(self.muse_directory, "Survey Information (Questions, Data etc)")):
            survey_folder_name = "Survey Information (Questions, Data etc)"
        else:
            survey_folder_name = "SurveyInformation"
        lab_file = os.path.join(self.muse_directory, survey_folder_name, "Emotion data from crowdsourcing (C is when annotators had access to all previous sentences).csv")
        get_utt_id = re.compile(r'(?P<utt_id>.*).wav')
        with open(lab_file, 'r') as r:
            for i, line in enumerate(r.readlines()):
                if i == 0: # Skip the first line with headers
                    continue
                line=line.rstrip()
                Sentence_name,Duration,Gender,Type,Ques_Type,Activation_Mean,Activation_SD,Valence_Mean,Valence_SD,Activation_Annotation,Valence_Annotation,C_Activation_Mean,C_Activation_SD,C_Valence_Mean,C_Valence_SD,C_Activation_Annotation,C_Valence_Annotation,S_Activation,S_Valence = line.split(',')
                utt_id = get_utt_id.match(Sentence_name).group('utt_id')
                transcript = self.read_transcript(utt_id, Type)
                if transcript is None:
                    continue
                if utt_id not in labels:
                    labels[utt_id] = {}
                else:
                    raise IOError(f'Encountered duplicate label {utt_id}')
                labels[utt_id]['act'] = float(Activation_Mean)
                labels[utt_id]['self-report-act'] = float(S_Activation)
                labels[utt_id]['soft_act_labels'] = [int(x) for x in Activation_Annotation.split(';')]
                labels[utt_id]['val'] = float(Valence_Mean)
                labels[utt_id]['self-report-val'] = float(S_Valence)
                labels[utt_id]['soft_val_labels'] = [int(x) for x in Valence_Annotation.split(';')]
                labels[utt_id]['transcript_text'] = transcript
                labels[utt_id]['gender'] = Gender
                labels[utt_id]['type'] = Type # S == stressed, NS == non_stressed
                labels[utt_id]['speaker_id'] = utt_id[:2]

        return labels

    def read_transcript(self, utt_id, stress_type):
        if self.use_whisper:
            if stress_type == 'NS':
                directory = 'Nonstressed_Segments'
            elif stress_type == 'S':
                directory = 'Stressed_Segments'
            else:
                raise ValueError(f'Unknown stress type: {stress_type}')
            transcript_file = os.path.join(self.muse_directory, 'Whisper-Transcriptions', directory, f'{utt_id}.txt')
        else:
            transcript_file = os.path.join(self.muse_directory, 'Combined', 'Utterance Transcripts', f'{utt_id}.txt')
        transcript = None
        if not os.path.isfile(transcript_file):
            print(f'Warning - no transcript found for {utt_id}, skipping.')
            return None
        with open(transcript_file, 'r') as f:
            transcript = f.read()
        
        return transcript

    def prepare_labels(self):
        return ['act', 'val']

    def get_wavs(self):
        non_stressed_wavs = glob(os.path.join(self.muse_directory, 'NotStressed', 'Audio', 'Non Stressed Question Monologue Audio*', '*.wav'))
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

    def get_dataset_splits(self, data_split_type, perception_of_self_only=False):
        split_type = super().get_dataset_splits(data_split_type)
        if type(split_type) == dict:
            return split_type
        elif type(split_type) == str:
            labels_to_use = None
            if perception_of_self_only: 
                labels_to_use = []
                for label in self.labels.keys():
                    if 'self-report-act' in self.labels[label]:
                        assert 'self-report-val' in self.labels[label]
                        labels_to_use.append(label)
            else:
                labels_to_use = list(self.labels.keys())

            if split_type == 'stress_type_split':
                types = {k: self.labels[k]['type'] for k in self.labels if 'type' in self.labels[k]}
                split = {
                    'stressed': [key for key in labels_to_use if types[key] == 'S'],
                    'non_stressed': [key for key in labels_to_use if types[key] == 'NS'],
                    'full': list(labels_to_use),
                }
                assert set(labels_to_use) == set(split['stressed'] + split['non_stressed'])
                return split
            elif split_type == 'speaker-independent':
                speakers = {k: self.labels[k]['speaker_id'] for k in self.labels if 'speaker_id' in self.labels[k]}
                # Randomly selected speaker-independent split kept static for repeat runs 
                train_f_speakers = ['20', '17', '25', '22', '08'] # 5 f 11 m speakers
                val_f_speakers = ['27', '11'] # 2 f 4 m speakers 
                test_f_speakers = ['19', '23'] # 2 f 4 m speakers 
                train_m_speakers = ['24', '12', '26', '21', '01', '04', '07', '03', '16', '06', '13']
                val_m_speakers = ['15', '05', '09', '10']
                test_m_speakers = ['14', '02', '18', '28']

                train_speakers = train_f_speakers + train_m_speakers
                val_speakers = val_f_speakers + val_m_speakers
                test_speakers = test_f_speakers + test_m_speakers
                splits = {
                    'train': [key for key in labels_to_use if speakers[key] in train_speakers],
                    'val': [key for key in labels_to_use if speakers[key] in val_speakers],
                    'test': [key for key in labels_to_use if speakers[key] in test_speakers],
                }

                assert set(labels_to_use) == set(splits['train'] + splits['val'] + splits['test'])
                return splits