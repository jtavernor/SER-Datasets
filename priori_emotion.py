import pandas
from tqdm import tqdm
from .dataset_constructor import DatasetConstructor

class PrioriEmotionConstructor(DatasetConstructor):
    def __init__(self, filter_fn=None, dataset_save_location=None):
        self.assessment_calls = pandas.read_csv('/nfs/turbo/McInnisLab/sandymn/ICASSP2023_mismatch/data/data_assess_arr.csv')
        self.personal_calls = pandas.read_csv('/nfs/turbo/McInnisLab/sandymn/ICASSP2023_mismatch/data/data_personal_arr.csv')
        self.assessment_calls['transcriptions_azure'] = self.assessment_calls['transcriptions_azure'].fillna('')
        self.personal_calls['transcriptions_azure'] = self.personal_calls['transcriptions_azure'].fillna('')
        self.samples = self.assessment_calls['id'].tolist() + self.personal_calls['id'].tolist()
        super().__init__(0, filter_fn, dataset_save_location)

    def read_labels(self):
        labels = {}
        self.label_id_to_audio_path = {}
        for sample_id in tqdm(self.samples, desc='Loading priori emotion pandas into labels'):
            call_csv = self.assessment_calls
            if not call_csv['id'].str.fullmatch(sample_id).any():
                call_csv = self.personal_calls
            pd_row = call_csv[call_csv['id'] == sample_id]
            audio_path = pd_row['path'].item()
            transcript = pd_row['transcriptions_azure'].item()
            activation = pd_row['activation'].item()
            valence = pd_row['valence'].item()
            speaker_id = pd_row['sub_id'].item()
            labels[sample_id] = {
                'act': activation, 'val': valence, 'transcript': transcript, 'speaker_id': speaker_id
            }
            self.label_id_to_audio_path[sample_id] = audio_path
        return labels

    def prepare_labels(self):
        return ['act', 'val']

    def get_wavs(self):
        return self.label_id_to_audio_path

    def get_dataset_splits(self, data_split_type):
        # Skip call to super in this case 
        # split_type = super().get_dataset_splits(data_split_type, expected_type=list)
        train_speakers = data_split_type['train']
        val_speakers = data_split_type['val']
        test_speakers = data_split_type['test']
        train_sample_ids = self.assessment_calls[self.assessment_calls['sub_id'].isin(train_speakers)]['id'].tolist() + self.personal_calls[self.personal_calls['sub_id'].isin(train_speakers)]['id'].tolist()
        val_sample_ids = self.assessment_calls[self.assessment_calls['sub_id'].isin(val_speakers)]['id'].tolist() + self.personal_calls[self.personal_calls['sub_id'].isin(val_speakers)]['id'].tolist()
        test_sample_ids = self.assessment_calls[self.assessment_calls['sub_id'].isin(test_speakers)]['id'].tolist() + self.personal_calls[self.personal_calls['sub_id'].isin(test_speakers)]['id'].tolist()
        return {
            'train': train_sample_ids,
            'val': val_sample_ids,
            'test': test_sample_ids
        }
