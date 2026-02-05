import numpy as np
import librosa
import soundfile as sf
import io

def extract_features(audio_bytes, sr=16000):
    # ✅ Decode bytes → numpy array
    y, sr = sf.read(io.BytesIO(audio_bytes))

    # Convert stereo → mono
    if len(y.shape) == 2:
        y = np.mean(y, axis=1)

    # Resample if needed
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))

    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y), sr=sr
    )
    features.extend(np.mean(tonnetz, axis=1))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))

    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))

    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    pitch = pitch[np.isfinite(pitch)]

    features.extend([
        np.mean(pitch),
        np.std(pitch),
        np.min(pitch),
        np.max(pitch),
        np.percentile(pitch, 25),
        np.percentile(pitch, 50),
        np.percentile(pitch, 75),
        np.ptp(pitch)
    ])

    return np.array(features)
