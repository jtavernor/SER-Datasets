### Set of functions to load various audio/text features
import librosa
from .config import Config

def read_wav(wav_path):
    SR = Config()['SR']
    y, sr = librosa.load(wav_path, sr=SR)
    return y, sr

def get_mfb(y, sr):
    mfb_config = Config()['mfb_settings']
    n_fft, n_mels, hop_length, fmin, fmax, clamp_values = mfb_config['n_fft'], mfb_config['n_mels'], mfb_config['hop_length'], mfb_config['fmin'], mfb_config['fmax'], mfb_config['clamp_values']
    y = librosa.effects.preemphasis(y, coef=0.97)
    mfb = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, fmin=fmin, fmax=fmax, htk=False)
    mfb = mfb.T
    if clamp_values:
        mfb = clamp_mfb(mfb)
    return mfb

def clamp_mfb(mfb):
    mfb[mfb>clampVal] = clampVal
    mfb[mfb<-clampVal] = -clampVal
    return mfb

def get_w2v2(y, sr, w2v2_model='facebook/wav2vec2-base'):
    pass

def get_bert_embedding(transcript, bert_model='google/bert_uncased_L-4_H-512_A-8'):
    pass