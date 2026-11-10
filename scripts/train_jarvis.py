"""Train the INT8 'Hello Jarvis' wake-phrase spotter for the ESP32 firmware.

Training data comes from two places:

1. Device recordings (data/device/*.bin + *.csv) made with
   scripts/play_dataset.py and scripts/device_capture.py: clips played through a
   speaker and recorded by the ESP32's own microphone/ADC chain. These are the
   ground truth for what the firmware hears, and a held-out part of them is the
   validation set.
2. Clean WAVs (TTS wake phrases, Speech Commands words, negative phrases) passed
   through the speaker->ADC transfer function measured from the device recordings,
   with real device background noise added.

Features must match firmware/main/feature_extractor.cc bit for bit; see
extract_features().
"""

import csv
import os
import struct
import wave
from pathlib import Path

import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Audio & feature configuration (keep in sync with firmware/main/config.h)
SAMPLE_RATE = 10_000
WINDOW_SAMPLES = 28_000  # 2.8 seconds
SLIDE_SAMPLES = 3_000    # firmware evaluates one window every 300 ms
FRAME_SAMPLES = 300      # 30 ms
HOP_SAMPLES = 200        # 20 ms
MEL_BINS = 40
FFT_SIZE = 512
FMIN = 60.0
FMAX = 4800.0
FRAME_COUNT = 1 + (WINDOW_SAMPLES - FRAME_SAMPLES) // HOP_SAMPLES  # 139
SPEECH_MIN_PEAK = 5000   # firmware silence gate (KWS_SPEECH_MIN_PEAK)

LABELS = ("hello_jarvis", "unknown")
DEVICE_DIR = Path("data/device")


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank():
    bins = np.floor((FFT_SIZE + 1) * mel_to_hz(np.linspace(
        hz_to_mel(FMIN), hz_to_mel(FMAX), MEL_BINS + 2
    )) / SAMPLE_RATE).astype(int)
    bank = np.zeros((MEL_BINS, FFT_SIZE // 2 + 1), dtype=np.float32)
    for band in range(MEL_BINS):
        left, center, right = bins[band:band + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for i in range(left, min(center, bank.shape[1])):
            bank[band, i] = (i - left) / (center - left)
        for i in range(center, min(right, bank.shape[1])):
            bank[band, i] = (right - i) / (right - center)
    return bank


MEL_BANK = build_mel_filterbank()
HAMMING = np.hamming(FRAME_SAMPLES).astype(np.float32)


def extract_features(audio_10k):
    """(139, 40, 1) float32 log-mel features for one 28,000-sample window @ 10 kHz."""
    audio = np.asarray(audio_10k, dtype=np.float32).reshape(-1)
    if audio.size < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - audio.size))
    audio = audio[:WINDOW_SAMPLES]
    frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME_SAMPLES)[::HOP_SAMPLES] * HAMMING
    spec = np.abs(np.fft.rfft(frames, n=FFT_SIZE)) ** 2
    features = np.log(np.maximum(spec @ MEL_BANK.T, 1e-10)).astype(np.float32)
    features -= features.mean()
    features /= max(float(features.std()), 1e-6)
    return features[..., None]


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        channels, rate = w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def to_10k(data, rate):
    if rate == SAMPLE_RATE:
        return data
    g = np.gcd(SAMPLE_RATE, rate)
    return scipy.signal.resample_poly(data, SAMPLE_RATE // g, rate // g).astype(np.float32)


def load_wav_10k(path):
    return to_10k(*read_wav(path))


def speech_bounds(audio, rel=0.04):
    """(start, end) of the part of `audio` above rel * peak 10 ms RMS."""
    env = np.sqrt(np.convolve(audio ** 2, np.ones(100) / 100, mode="same"))
    idx = np.where(env > rel * env.max())[0]
    return (int(idx[0]), int(idx[-1]) + 1) if idx.size else (0, len(audio))


def trim(audio):
    s, e = speech_bounds(audio)
    return audio[s:e]


# ---------------------------------------------------------------------------
# Device recordings
# ---------------------------------------------------------------------------

def load_session(bin_path):
    """Return (stream, valid-slide mask, (a, b)) with slide i ending at host time a + b*i."""
    raw = Path(bin_path).read_bytes()
    rec = 12 + SLIDE_SAMPLES * 2
    n = len(raw) // rec
    seqs = np.array([struct.unpack_from("<I", raw, k * rec)[0] for k in range(n)])
    recv = np.array([struct.unpack_from("<d", raw, k * rec + 4)[0] for k in range(n)])
    first = seqs[0]
    count = seqs[-1] - first + 1
    stream = np.zeros(count * SLIDE_SAMPLES, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    for k in range(n):
        i = seqs[k] - first
        stream[i * SLIDE_SAMPLES:(i + 1) * SLIDE_SAMPLES] = np.frombuffer(
            raw, dtype="<i2", count=SLIDE_SAMPLES, offset=k * rec + 12)
        valid[i] = True
    # receive time = slide end time + polling delay (>= 0): fit a line, shift it
    # down to the fastest arrivals.
    idx = seqs - first
    b, a = np.polyfit(idx, recv, 1)
    a += np.percentile(recv - (a + b * idx), 2)
    return stream, valid, (a, b)


ENV_BLOCK = 100  # 10 ms envelope blocks for alignment


def log_env(x):
    x = np.pad(x, (0, -len(x) % ENV_BLOCK))
    return np.log(x.reshape(-1, ENV_BLOCK).std(axis=1) + 30.0)


def best_match(env_stream, env_clip, center, radius):
    """(correlation, block index) of the best envelope match of env_clip near block `center`."""
    best = (-1.0, None)
    n = len(env_clip)
    c = env_clip - env_clip.mean()
    norm_c = np.sqrt(np.dot(c, c)) + 1e-9
    for j in range(max(0, center - radius), min(len(env_stream) - n, center + radius) + 1):
        seg = env_stream[j:j + n]
        seg = seg - seg.mean()
        corr = float(np.dot(c, seg) / (norm_c * (np.sqrt(np.dot(seg, seg)) + 1e-9)))
        if corr > best[0]:
            best = (corr, j)
    return best


def align_rows(stream, rows, slide_period, t_stream0, min_corr=0.6):
    """Locate every played clip in the recorded stream.

    Stream position is linear in host time (device sample clock, missing slides
    are zero-filled in place), but HTTP receive times are a poor clock, so the
    rate and offset are fitted from the audio itself: a coarse grid search that
    maximises envelope agreement over all clips, then a +-0.4 s per-clip
    refinement for playback-latency jitter. Returns [(row, clip, start_sample)]
    for clips whose envelope correlation is >= min_corr.
    """
    env_stream = log_env(stream)
    played = [r for r in rows if r["label"] not in ("silence", "noise")]
    clips = {id(r): load_wav_10k(r["source"]) for r in played}
    envs = {k: log_env(c) for k, c in clips.items()}
    t_first = float(played[0]["start"])
    base = (t_first - t_stream0) / slide_period * SLIDE_SAMPLES  # receive-time estimate of the first clip

    def total(rate, offset):
        score = 0.0
        for r in played[::3]:
            e = envs[id(r)]
            j = int((base + offset + (float(r["start"]) - t_first) * rate) // ENV_BLOCK)
            if 0 <= j and j + len(e) <= len(env_stream):
                seg = env_stream[j:j + len(e)]
                score += float(np.dot(e - e.mean(), seg - seg.mean()))
        return score

    _, rate, offset = max((total(r, o), r, o) for r in np.arange(9900, 10061, 10)
                          for o in np.arange(-100000, 100001, 1000))
    _, rate, offset = max((total(r, o), r, o) for r in np.arange(rate - 10, rate + 11, 2)
                          for o in np.arange(offset - 1500, offset + 1501, 100))
    out = []
    for r in played:
        guess = base + offset + (float(r["start"]) - t_first) * rate
        corr, block = best_match(env_stream, envs[id(r)], int(guess // ENV_BLOCK), 40)
        if block is not None and corr >= min_corr:
            out.append((r, clips[id(r)], block * ENV_BLOCK))
    for r in rows:  # calibration noise clip: flat envelope, place it by the fitted clock
        if r["label"] == "noise":
            pos = int(base + offset + (float(r["start"]) - t_first) * rate)
            if pos >= 0:
                out.insert(0, (r, load_wav_10k(r["source"]), pos))
    print(f"  fitted device rate {rate} samples per host second")
    return out


def device_windows(session):
    """Windows of one recorded session, as the firmware would see them.

    Returns (windows, channel, stream, valid) where windows is a list of
    (audio, label, group, peak): label 0 = window holds (nearly) the whole wake
    phrase, 1 = no part of a wake phrase. Windows cutting a wake phrase, and
    windows near clips that could not be aligned, are skipped.
    """
    stream, valid, (a, b) = load_session(DEVICE_DIR / f"{session}.bin")
    rows = list(csv.DictReader(open(DEVICE_DIR / f"{session}.csv")))
    t_stream0 = a - b  # host time at stream sample 0 (start of slide 0)
    aligned = align_rows(stream, rows, b, t_stream0)
    played = [r for r in rows if r["label"] not in ("silence",)]
    print(f"{session}: aligned {len(aligned)} of {len(played)} clips")
    events, channel = [], None
    for row, clip, pos in aligned:
        if row["label"] == "noise":
            channel = (clip[2000:-2000], stream[pos + 2000:pos + len(clip) - 2000])
            continue
        s, e = speech_bounds(clip)
        events.append((pos + s, pos + e, row["label"], f"{session}:{Path(row['source']).name}"))
    # Unaligned clips: block out their rough location so they never become negatives by accident.
    rate = SLIDE_SAMPLES / b
    aligned_ids = {id(r) for r, _, _ in aligned}
    blocked = []
    for row in played:
        if id(row) not in aligned_ids:
            c = (float(row["start"]) - t_stream0) * rate
            blocked.append((c - 20000, c + 60000))
    last_aligned = max((pos for _, _, pos in aligned), default=0) + 40000
    positives = [(s, e, g) for s, e, lab, g in events if lab == "hello_jarvis"]
    others = [(s, e, g) for s, e, lab, g in events if lab != "hello_jarvis"]
    out = []
    for k in range(WINDOW_SAMPLES // SLIDE_SAMPLES + 1, len(valid) + 1):
        end = k * SLIDE_SAMPLES
        start = end - WINDOW_SAMPLES
        if end > last_aligned or not valid[start // SLIDE_SAMPLES:k].all():
            continue
        label, group, ambiguous = 1, f"{session}:bg{end // 60000}", False
        for s, e, g in positives:
            overlap = max(0, min(e, end) - max(s, start))
            if overlap == 0:
                continue
            if overlap >= min(0.92 * (e - s), 0.95 * WINDOW_SAMPLES):
                label, group = 0, g
            else:
                ambiguous = True
        if ambiguous and label == 1:
            continue
        if label == 1:
            if any(min(hi, end) > max(lo, start) for lo, hi in blocked):
                continue
            for s, e, g in others:
                if min(e, end) > max(s, start):
                    group = g
                    break
        window = stream[start:end]
        out.append((window, label, group, float(np.abs(window).max())))
    return out, channel, stream, valid


def measure_channel(played, recorded):
    """Magnitude response |H(f)| (FFT_SIZE//2+1 bins, max 1) of speaker -> mic -> ADC."""
    _, p_in = scipy.signal.welch(played, SAMPLE_RATE, nperseg=FFT_SIZE)
    _, p_out = scipy.signal.welch(recorded, SAMPLE_RATE, nperseg=FFT_SIZE)
    h = np.sqrt(p_out / np.maximum(p_in, 1e-9))
    return (h / h.max()).astype(np.float32)


def apply_channel(audio, h, rng):
    # Random +-4 dB/octave-ish tilt so the model does not overfit one response.
    f = np.linspace(0, 1, len(h))
    h = h * np.exp(rng.uniform(-0.6, 0.6) * (f - 0.2))
    taps = np.roll(np.fft.irfft(h, n=FFT_SIZE), FFT_SIZE // 2) * np.hanning(FFT_SIZE)
    return scipy.signal.fftconvolve(audio, taps, mode="same").astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def place(clip, rng, center_jitter=None):
    out = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    clip = clip[:WINDOW_SAMPLES]
    room = WINDOW_SAMPLES - len(clip)
    if center_jitter is None:
        start = rng.integers(0, room + 1)
    else:
        start = int(np.clip(room // 2 + rng.integers(-center_jitter, center_jitter + 1), 0, room))
    out[start:start + len(clip)] = clip
    return out


def finish(audio, noise_bank, rng, peak):
    audio = audio / max(np.abs(audio).max(), 1.0) * peak
    noise = noise_bank[rng.integers(0, len(noise_bank))]
    off = rng.integers(0, len(noise) - WINDOW_SAMPLES + 1)
    audio = audio + noise[off:off + WINDOW_SAMPLES] * rng.uniform(0.6, 1.6)
    return np.clip(np.round(audio), -32768, 32767).astype(np.float32)


def simulated_set(h, noise_bank, rng):
    def clips(directory, limit=None):
        files = sorted(Path(directory).glob("*.wav"))[:limit]
        return [trim(load_wav_10k(f)) for f in files]

    jarvis = clips("data/raw/hello_jarvis")
    phrases = clips("data/raw/unknown_phrases")
    words = clips("data/raw/unknown", 2500)
    commands = [c for d in ("start", "stop", "next", "previous", "lights_on", "lights_off",
                            "volume_up", "volume_down") for c in clips(f"data/raw/{d}")]
    print(f"clean clips: {len(jarvis)} wake, {len(phrases)} phrases, {len(words)} words, {len(commands)} commands")
    X, y = [], []

    def add(audio, label):
        audio = finish(apply_channel(audio, h, rng), noise_bank, rng, rng.uniform(5000, 30000))
        X.append(extract_features(audio))
        y.append(label)

    def run_of(pool, max_parts):
        parts = []
        for _ in range(rng.integers(1, max_parts + 1)):
            parts += [pool[rng.integers(len(pool))], np.zeros(rng.integers(200, 4000), np.float32)]
        return np.concatenate(parts)

    for i in range(720):
        src = jarvis[i % len(jarvis)]
        speed = rng.uniform(0.9, 1.1)
        src = scipy.signal.resample(src, int(len(src) / speed)).astype(np.float32)
        add(place(src, rng, center_jitter=2500), 0)
    for i in range(400):
        add(place(phrases[i % len(phrases)], rng), 1)
    for _ in range(1000):
        add(place(run_of(words, 4), rng), 1)
    for _ in range(300):
        add(place(run_of(commands, 3), rng), 1)
    for _ in range(250):  # wake phrase fragments: the start only, or the end only
        src = jarvis[rng.integers(len(jarvis))]
        cut = int(len(src) * rng.uniform(0.25, 0.55))
        add(place(src[:cut] if rng.random() < 0.5 else src[-cut:], rng), 1)
    for _ in range(200):  # knocks, coughs, short low-frequency bursts
        burst = rng.normal(0, 1, rng.integers(800, 6000))
        b, a = scipy.signal.butter(2, rng.uniform(0.04, 0.4))
        burst = (scipy.signal.lfilter(b, a, burst) * np.hanning(len(burst))).astype(np.float32)
        add(place(burst, rng), 1)
    return np.stack(X), np.array(y, dtype=np.int32)


def build_model():
    """~2M MAC depthwise-separable CNN, sized for the 300 ms slide budget on an ESP32."""
    def bn_relu(x):  # batch norm folds into the preceding conv at conversion time
        return layers.ReLU()(layers.BatchNormalization()(x))

    inputs = keras.Input(shape=(FRAME_COUNT, MEL_BINS, 1), name="input")
    x = bn_relu(layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", use_bias=False)(inputs))
    for filters, pool in ((32, True), (48, True), (64, False)):
        x = bn_relu(layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x))
        x = bn_relu(layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x))
        if pool:
            x = layers.MaxPool2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(LABELS), activation="softmax", name="output")(x)
    model = keras.Model(inputs, outputs, name="jarvis_spotter")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def quantize_input(features, scale, zero_point):
    return np.clip(np.round(features / scale + zero_point), -128, 127).astype(np.int8)


def tflite_p_wake(tflite_model, X):
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]
    (i_s, i_z), (o_s, o_z) = i_d["quantization"], o_d["quantization"]
    out = np.empty(len(X), dtype=np.float32)
    for k in range(len(X)):
        interp.set_tensor(i_d["index"], quantize_input(X[k], i_s, i_z)[None])
        interp.invoke()
        out[k] = (int(interp.get_tensor(o_d["index"])[0][0]) - o_z) * o_s
    return out


def report(name, p, y, groups, gate_ok, threshold):
    wake_groups = sorted({g for g, lab in zip(groups, y) if lab == 0})
    hit = [np.any(p[(groups == g) & (y == 0)] >= threshold) for g in wake_groups]
    neg = (y == 1) & gate_ok
    print(f"{name}: wake phrases detected {np.mean(hit) * 100:5.1f}% of {len(hit)}, "
          f"wake windows {np.mean(p[y == 0] >= threshold) * 100:5.1f}%, "
          f"false-positive windows {np.mean(p[neg] >= threshold) * 100:5.2f}% of {neg.sum()}")


def main():
    rng = np.random.default_rng(42)
    sessions = sorted(p.stem for p in DEVICE_DIR.glob("*.bin") if (DEVICE_DIR / f"{p.stem}.csv").exists())
    if not sessions:
        raise SystemExit("no device recordings in data/device; run play_dataset.py + device_capture.py first")
    dev, h, noise_bank = [], None, []
    for session in sessions:
        windows, channel, stream, valid = device_windows(session)
        print(f"{session}: {len(windows)} windows ({sum(w[1] == 0 for w in windows)} wake)")
        dev += windows
        if channel is not None and h is None:
            h = measure_channel(*channel)
        quiet = [stream[i * SLIDE_SAMPLES:(i + 1) * SLIDE_SAMPLES] for i in range(len(valid))
                 if valid[i] and np.abs(stream[i * SLIDE_SAMPLES:(i + 1) * SLIDE_SAMPLES]).max() < 0.8 * SPEECH_MIN_PEAK]
        noise_bank += [np.concatenate(quiet[i:i + 12]) for i in range(0, len(quiet) - 12, 12)]
    if h is None or not noise_bank:
        raise SystemExit("recordings need the 'noise' calibration clip and some quiet background")
    print("channel |H| at 250/500/1k/2k/3k/4k Hz:", np.round(h[[13, 26, 51, 102, 154, 205]], 3).tolist(),
          f"| background noise clips: {len(noise_bank)}")

    # Split device windows by source clip so no validation clip is trained on.
    groups = np.array([g for _, _, g, _ in dev])
    unique = sorted(set(groups))
    val_groups = set(rng.permutation(unique)[: len(unique) // 4])
    is_val = np.array([g in val_groups for g in groups])
    Xd = np.stack([extract_features(w) for w, _, _, _ in dev])
    yd = np.array([lab for _, lab, _, _ in dev], dtype=np.int32)
    gate_ok = np.array([pk >= SPEECH_MIN_PEAK for _, _, _, pk in dev])

    Xs, ys = simulated_set(h, noise_bank, rng)
    X_train = np.concatenate([Xd[~is_val]] * 3 + [Xs])  # real device windows weighted x3
    y_train = np.concatenate([yd[~is_val]] * 3 + [ys])
    X_val, y_val = Xd[is_val], yd[is_val]
    print(f"train {len(X_train)} windows (wake {np.sum(y_train == 0)}), "
          f"device validation {len(X_val)} (wake {np.sum(y_val == 0)})")

    counts = np.bincount(y_train, minlength=2)
    class_weight = {c: len(y_train) / (2.0 * counts[c]) for c in range(2)}
    tf.keras.utils.set_random_seed(42)
    model = build_model()
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=64,
              class_weight=class_weight, shuffle=True, verbose=2,
              callbacks=[keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
                         keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])

    def representative_dataset():
        for k in rng.choice(len(X_train), 300, replace=False):
            yield [X_train[k:k + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    p_val = tflite_p_wake(tflite_model, X_val)
    for threshold in (0.5, 0.65, 0.8, 0.9):
        report(f"INT8 device validation @ {threshold:.2f}", p_val, y_val, groups[is_val], gate_ok[is_val], threshold)

    os.makedirs("artifacts", exist_ok=True)
    Path("artifacts/keyword_spotter.tflite").write_bytes(tflite_model)
    print(f"INT8 model: {len(tflite_model)} bytes")
    c_bytes = ", ".join(f"0x{b:02x}" for b in tflite_model)
    Path("firmware/main/model_data.cc").write_text(
        '#include "model_data.h"\n\n'
        f"alignas(16) const unsigned char g_keyword_spotter_model[] = {{\n    {c_bytes}\n}};\n"
        f"const unsigned int g_keyword_spotter_model_len = {len(tflite_model)};\n",
        encoding="utf-8")
    print("wrote firmware/main/model_data.cc")


if __name__ == "__main__":
    main()
