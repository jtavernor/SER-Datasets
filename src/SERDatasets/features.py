### Set of functions to load various audio/text features
import librosa
import torch
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
    clampVal = Config()['mfb_settings']['clamp_val']
    mfb[mfb>clampVal] = clampVal
    mfb[mfb<-clampVal] = -clampVal
    return mfb

def get_w2v2(y, sr, w2v2_extractor, w2v2_model, upsample=False):
    with torch.no_grad():
        if sr != 16000 and upsample:
            # print(f'Resampling audio from {sr} to 16000')
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        audio_features = w2v2_extractor(y, sampling_rate=16000, return_tensors='pt')
        if torch.cuda.is_available():
            audio_features = audio_features.to('cuda')
        assert len(audio_features) == 1
        audio_features = w2v2_model(**audio_features)['last_hidden_state']

        return audio_features.cpu().numpy()

def pool_w2v2(unpooled_audio_features):
    with torch.no_grad():
        audio_features_pooled = torch.mean(torch.as_tensor(unpooled_audio_features), dim=1)
        audio_features_pooled = audio_features_pooled.squeeze(dim=0).cpu().numpy()
    return audio_features_pooled

def get_bert_embedding(transcript, bert_tokenizer, bert_model, return_cls=True):
    with torch.no_grad():
        bert_tokens = bert_tokenizer.encode_plus(transcript, add_special_tokens=True, return_tensors='pt')
        # if torch.cuda.is_available():
            # bert_tokens = bert_tokens.to('cuda')
        bert_tokens = bert_tokens.to(bert_model.device)
        out = bert_model(**bert_tokens).last_hidden_state
        if return_cls:
            return out.squeeze()[0].cpu()
        else:
            raise NotImplementedError('todo')