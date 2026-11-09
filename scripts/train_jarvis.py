"""Train full INT8 Keyword Spotter for 'Hello Jarvis Development Mode On'."""

import os
from pathlib import Path
import wave
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Audio & Feature Configuration @ 10 kHz
SAMPLE_RATE = 10_000
WINDOW_SAMPLES = 28_000  # 2.8 seconds
FRAME_SAMPLES = 300      # 30 ms
HOP_SAMPLES = 200        # 20 ms
MEL_BINS = 40
FFT_SIZE = 512
FMIN = 60.0
FMAX = 4800.0
FRAME_COUNT = 1 + (WINDOW_SAMPLES - FRAME_SAMPLES) // HOP_SAMPLES  # 139

LABELS = ("hello_jarvis", "unknown")
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}


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
    """Extract (139, 40, 1) float32 log-mel features from 28,000 samples @ 10kHz."""
    audio = np.asarray(audio_10k, dtype=np.float32).reshape(-1)
    if audio.size < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - audio.size))
    audio = audio[:WINDOW_SAMPLES]

    frames = np.stack([
        audio[s:s + FRAME_SAMPLES]
        for s in range(0, WINDOW_SAMPLES - FRAME_SAMPLES + 1, HOP_SAMPLES)
    ])
    frames *= HAMMING
    spec = np.abs(np.fft.rfft(frames, n=FFT_SIZE)) ** 2
    energies = spec @ MEL_BANK.T
    features = np.log(np.maximum(energies, 1e-10)).astype(np.float32)
    features -= features.mean()
    features /= max(float(features.std()), 1e-6)
    return features[..., None]


def load_wav_10k(path):
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        src_sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        data = data[::n_channels]
    if src_sr == 16000:
        # Resample 16000 -> 10000: 16000 * 5 / 8 = 10000
        data = scipy.signal.resample_poly(data, 5, 8).astype(np.float32)
    elif src_sr != SAMPLE_RATE:
        num_target = int(len(data) * SAMPLE_RATE / src_sr)
        data = scipy.signal.resample(data, num_target).astype(np.float32)
    return data


def generate_dataset():
    rng = np.random.default_rng(42)
    
    # 1. Load raw Hello Jarvis files
    raw_dir = Path("data/raw/hello_jarvis")
    jarvis_files = list(raw_dir.glob("*.wav"))
    print(f"Found {len(jarvis_files)} raw hello_jarvis recordings")
    
    jarvis_audios = [load_wav_10k(f) for f in jarvis_files]

    # Generate 400 augmented samples for hello_jarvis
    X_jarvis = []
    for i in range(400):
        src = jarvis_audios[i % len(jarvis_audios)]
        # Peak normalize
        peak = np.max(np.abs(src))
        if peak > 0:
            src = src / peak * 30000.0
        
        # Slight pitch/speed stretch
        speed_factor = rng.uniform(0.92, 1.08)
        if abs(speed_factor - 1.0) > 0.02:
            num_samples = int(len(src) / speed_factor)
            stretched = scipy.signal.resample(src, num_samples).astype(np.float32)
        else:
            stretched = src.copy()
            
        # Center in 28000 window with realistic speech jitter (+/- 2500 samples = +/- 250ms)
        out = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        if len(stretched) < WINDOW_SAMPLES:
            ideal_start = (WINDOW_SAMPLES - len(stretched)) // 2
            jitter = rng.integers(-2000, 2001)
            start = max(0, min(WINDOW_SAMPLES - len(stretched), ideal_start + jitter))
            out[start:start + len(stretched)] = stretched
        else:
            out = stretched[:WINDOW_SAMPLES]
            
        # Volume variation
        vol = rng.uniform(0.7, 1.0)
        out *= vol
        
        # Add subtle room noise
        noise_level = rng.uniform(200.0, 1500.0)
        out += rng.normal(0, noise_level, size=WINDOW_SAMPLES)
        
        feat = extract_features(out)
        X_jarvis.append(feat)

    # 2. Generate 400 unknown samples
    # From negative phrases
    neg_dir = Path("data/raw/unknown_phrases")
    neg_files = list(neg_dir.glob("*.wav"))
    print(f"Found {len(neg_files)} negative phrase recordings")
    neg_audios = [load_wav_10k(f) for f in neg_files]
    
    # Other single command audio files
    other_cmds = ["lights_on", "lights_off", "start", "stop", "volume_up", "volume_down", "next", "previous"]
    cmd_audios = []
    for c in other_cmds:
        cdir = Path(f"data/raw/{c}")
        if cdir.exists():
            for cf in list(cdir.glob("*.wav"))[:15]:
                cmd_audios.append(load_wav_10k(cf))
    print(f"Found {len(cmd_audios)} other command samples for negative set")

    X_unknown = []
    # 200 phrase negatives
    for i in range(200):
        src = neg_audios[i % len(neg_audios)]
        peak = np.max(np.abs(src))
        if peak > 0:
            src = src / peak * 30000.0
        out = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        if len(src) < WINDOW_SAMPLES:
            start = rng.integers(0, WINDOW_SAMPLES - len(src) + 1)
            out[start:start + len(src)] = src
        else:
            out = src[:WINDOW_SAMPLES]
        vol = rng.uniform(0.7, 1.0)
        out = out * vol + rng.normal(0, rng.uniform(200, 1500), size=WINDOW_SAMPLES)
        X_unknown.append(extract_features(out))

    # 100 concatenated other words
    for i in range(100):
        out = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        # pick 2 random other commands
        c1 = cmd_audios[rng.integers(0, len(cmd_audios))]
        c2 = cmd_audios[rng.integers(0, len(cmd_audios))]
        p1 = np.max(np.abs(c1))
        p2 = np.max(np.abs(c2))
        if p1 > 0: c1 = c1 / p1 * 28000.0
        if p2 > 0: c2 = c2 / p2 * 28000.0
        # insert c1 at ~2000, c2 at ~14000
        l1 = min(len(c1), 10000)
        l2 = min(len(c2), 10000)
        out[2000:2000+l1] = c1[:l1]
        out[14000:14000+l2] = c2[:l2]
        out += rng.normal(0, rng.uniform(200, 1500), size=WINDOW_SAMPLES)
        X_unknown.append(extract_features(out))

    # 50 pure room noises
    for _ in range(50):
        out = rng.normal(0, rng.uniform(800, 3000), size=WINDOW_SAMPLES).astype(np.float32)
        X_unknown.append(extract_features(out))

    # 50 digital silences
    for _ in range(50):
        out = rng.normal(0, 10.0, size=WINDOW_SAMPLES).astype(np.float32)
        X_unknown.append(extract_features(out))

    X = np.concatenate([np.stack(X_jarvis), np.stack(X_unknown)], axis=0)
    y = np.zeros(len(X), dtype=np.int32)
    y[:len(X_jarvis)] = 0
    y[len(X_jarvis):] = 1

    # Shuffle with stratified split: 320 train / 80 val per class
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    rng.shuffle(idx_0)
    rng.shuffle(idx_1)

    train_idx = np.concatenate([idx_0[:320], idx_1[:320]])
    val_idx = np.concatenate([idx_0[320:], idx_1[320:]])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Dataset ready: Train shape {X_train.shape}, Val shape {X_val.shape}")
    return X_train, y_train, X_val, y_val


def build_model():
    inputs = keras.Input(shape=(FRAME_COUNT, MEL_BINS, 1), name="input")
    x = layers.Conv2D(24, (3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = layers.DepthwiseConv2D((3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, (1, 1), padding="same", activation="relu")(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(48, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(48, activation="relu")(x)
    outputs = layers.Dense(len(LABELS), activation="softmax", name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="jarvis_spotter")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    X_train, y_train, X_val, y_val = generate_dataset()
    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Keras Float32 Validation Accuracy: {acc * 100:.2f}%")

    # Quantize to INT8
    def representative_dataset():
        for i in range(40):
            yield [X_val[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant = converter.convert()

    os.makedirs("artifacts", exist_ok=True)
    tflite_path = Path("artifacts/keyword_spotter.tflite")
    tflite_path.write_bytes(tflite_quant)
    print(f"Quantized INT8 model size: {len(tflite_quant)} bytes ({len(tflite_quant)/1024:.1f} KB)")

    # Verify INT8 accuracy on validation set
    interpreter = tf.lite.Interpreter(model_content=tflite_quant)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    correct = 0
    per_class_total = {0: 0, 1: 0}
    per_class_correct = {0: 0, 1: 0}

    for i in range(len(X_val)):
        sample = X_val[i]
        label = y_val[i]
        quant_input = np.round(sample / in_scale + in_zero).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quant_input[None, ...])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        pred = np.argmax(output)
        per_class_total[label] += 1
        if pred == label:
            correct += 1
            per_class_correct[label] += 1

    int8_acc = correct / len(X_val)
    print(f"\nINT8 Validation Accuracy: {int8_acc * 100:.2f}% ({correct}/{len(X_val)})")
    for cls_id in (0, 1):
        c_acc = per_class_correct[cls_id] / max(per_class_total[cls_id], 1)
        print(f"  Class {cls_id} ({LABELS[cls_id]}): {c_acc * 100:.1f}% ({per_class_correct[cls_id]}/{per_class_total[cls_id]})")

    # Embed into firmware/main/model_data.cc
    c_bytes = ", ".join(f"0x{b:02x}" for b in tflite_quant)
    cc_content = f"""#include "model_data.h"

alignas(16) const unsigned char g_keyword_spotter_model[] = {{
    {c_bytes}
}};
const int g_keyword_spotter_model_len = {len(tflite_quant)};
"""
    model_cc_path = Path("firmware/main/model_data.cc")
    model_cc_path.write_text(cc_content, encoding="utf-8")
    print(f"Successfully wrote model to {model_cc_path}")


if __name__ == "__main__":
    main()
