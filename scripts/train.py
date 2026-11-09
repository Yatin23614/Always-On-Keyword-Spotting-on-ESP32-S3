"""Train and export a strictly balanced full-int8 keyword spotter for ESP32."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from kws.features import FeatureConfig, extract_log_mel
from kws.labels import COMMAND_LABELS, LABEL_TO_ID

def align_and_center(audio: np.ndarray, target_len: int = 16000) -> np.ndarray:
    """Pad or window audio to target_len centered around the highest energy region."""
    if len(audio) < target_len:
        pad_left = (target_len - len(audio)) // 2
        pad_right = target_len - len(audio) - pad_left
        return np.pad(audio, (pad_left, pad_right))
    elif len(audio) > target_len:
        energy = audio.astype(np.float64) ** 2
        best_start = 0
        max_energy = -1.0
        for start in range(0, len(audio) - target_len + 1, 160):
            e = float(np.sum(energy[start:start + target_len]))
            if e > max_energy:
                max_energy = e
                best_start = start
        return audio[best_start:best_start + target_len]
    return audio

def load_wav_file(path: Path, config: FeatureConfig) -> tuple[np.ndarray | None, float]:
    """Load, check sample rate, and return (normalized_audio, raw_peak)."""
    rate, audio = wavfile.read(path)
    if rate != config.sample_rate:
        raise ValueError(f"{path}: expected {config.sample_rate} Hz, got {rate}")
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return None, 0.0
    normalized = audio / peak
    aligned = align_and_center(normalized, config.window_samples)
    return aligned, peak

def time_shift(audio: np.ndarray, shift: int) -> np.ndarray:
    """Zero-padded time shift (positive = shift right / later)."""
    shifted = np.zeros_like(audio)
    if shift > 0:
        shifted[shift:] = audio[:-shift]
    elif shift < 0:
        shifted[:shift] = audio[-shift:]
    else:
        shifted = audio.copy()
    return shifted

def speed_pitch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Subtle time-stretch using linear interpolation."""
    n = len(audio)
    indices = np.linspace(0, n - 1, int(n * factor))
    stretched = np.interp(indices, np.arange(n), audio).astype(np.float32)
    return align_and_center(stretched, n)

def augment_sample(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply realistic augmentations: random shift, gain, subtle speed, and noise."""
    out = audio.copy()
    # 1. Time shift (-2400 to +2400 samples, i.e. +/- 150 ms)
    shift = rng.integers(-2400, 2400)
    out = time_shift(out, shift)
    
    # 2. Subtle speed/pitch variation (0.94 to 1.06)
    if rng.random() > 0.4:
        factor = rng.uniform(0.94, 1.06)
        out = speed_pitch(out, factor)
    
    # 3. Gain variation (0.7 to 1.3)
    gain = rng.uniform(0.70, 1.30)
    out = out * gain
    
    # 4. Low-level Gaussian noise
    if rng.random() > 0.3:
        sigma = rng.uniform(0.005, 0.030)
        out = out + rng.normal(0, sigma, size=len(out)).astype(np.float32)
        
    return out

def collect_balanced_dataset(data_dir: Path, config: FeatureConfig, samples_per_class: int = 200, seed: int = 42):
    """Collect an exactly balanced dataset: every class has identical sample count."""
    rng = np.random.default_rng(seed)
    
    features_by_class = {label: [] for label in COMMAND_LABELS}
    discarded_noise_samples = []

    # 1. Process Command Classes (0 through 7)
    for label in COMMAND_LABELS:
        if label == "unknown":
            continue
        folder = data_dir / label
        paths = sorted(folder.glob("*.wav"))
        clean_audios = []
        for p in paths:
            audio, peak = load_wav_file(p, config)
            if audio is None:
                continue
            # Filter out silent/blank user recordings (ambient mic noise floor is < 3000)
            if peak < 3000:
                discarded_noise_samples.append(audio)
                continue
            clean_audios.append(audio)

        if not clean_audios:
            raise ValueError(f"No valid clean audio found for label '{label}' in {folder}")

        print(f"[{label:12s}] Found {len(clean_audios)} clean source files. Target: {samples_per_class}")

        # Add all base clean audios
        for audio in clean_audios:
            features_by_class[label].append(extract_log_mel(audio, config))
            if len(features_by_class[label]) >= samples_per_class:
                break

        # Generate augmented variations until exactly samples_per_class is reached
        cycle_idx = 0
        while len(features_by_class[label]) < samples_per_class:
            base = clean_audios[cycle_idx % len(clean_audios)]
            aug = augment_sample(base, rng)
            features_by_class[label].append(extract_log_mel(aug, config))
            cycle_idx += 1

    # 2. Process Unknown Class (Class 8)
    # Balanced composition: 50% other speech commands, 25% digital silence, 25% room/analog noise
    unknown_speech_target = samples_per_class // 2                  # e.g. 100
    unknown_silence_target = samples_per_class // 4                 # e.g. 50
    unknown_noise_target = samples_per_class - unknown_speech_target - unknown_silence_target # e.g. 50

    # 2a. Real Speech Commands from unknown folder
    unknown_folder = data_dir / "unknown"
    unknown_paths = sorted(unknown_folder.glob("*.wav"))
    if len(unknown_paths) < unknown_speech_target:
        raise ValueError(f"Not enough unknown speech files in {unknown_folder}")
    chosen_paths = rng.choice(unknown_paths, size=unknown_speech_target, replace=False)
    for p in chosen_paths:
        audio, peak = load_wav_file(p, config)
        if audio is not None:
            features_by_class["unknown"].append(extract_log_mel(audio, config))

    # 2b. Digital Silence (zeros and near-zeros)
    for _ in range(unknown_silence_target):
        if rng.random() > 0.5:
            silence = np.zeros(config.window_samples, dtype=np.float32)
        else:
            silence = rng.normal(0, 1e-4, config.window_samples).astype(np.float32)
        features_by_class["unknown"].append(extract_log_mel(silence, config))

    # 2c. Room noise / Analog ADC hiss
    for _ in range(unknown_noise_target):
        if discarded_noise_samples and rng.random() > 0.5:
            noise_audio = rng.choice(discarded_noise_samples)
            noise_audio = noise_audio * rng.uniform(0.1, 0.5)
        else:
            sigma = rng.uniform(0.005, 0.12)
            noise_audio = rng.normal(0, sigma, config.window_samples).astype(np.float32)
        features_by_class["unknown"].append(extract_log_mel(noise_audio, config))

    # Ensure unknown class also has exactly samples_per_class
    while len(features_by_class["unknown"]) < samples_per_class:
        sigma = rng.uniform(0.01, 0.08)
        noise = rng.normal(0, sigma, config.window_samples).astype(np.float32)
        features_by_class["unknown"].append(extract_log_mel(noise, config))
    features_by_class["unknown"] = features_by_class["unknown"][:samples_per_class]

    print(f"[{'unknown':12s}] Generated {len(features_by_class['unknown'])} samples (speech={unknown_speech_target}, silence={unknown_silence_target}, noise={unknown_noise_target})")

    # 3. Stratified Train / Validation Split (80% train, 20% val for each class)
    x_train, y_train, x_val, y_val = [], [], [], []
    train_per_class = int(samples_per_class * 0.8)

    for label in COMMAND_LABELS:
        cls_features = np.stack(features_by_class[label])
        cls_labels = np.full(len(cls_features), LABEL_TO_ID[label], dtype=np.int64)
        perm = rng.permutation(len(cls_features))
        cls_features, cls_labels = cls_features[perm], cls_labels[perm]

        x_train.append(cls_features[:train_per_class])
        y_train.append(cls_labels[:train_per_class])
        x_val.append(cls_features[train_per_class:])
        y_val.append(cls_labels[train_per_class:])

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_val = np.concatenate(x_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    # Shuffle training set
    train_perm = rng.permutation(len(y_train))
    x_train, y_train = x_train[train_perm], y_train[train_perm]

    val_perm = rng.permutation(len(y_val))
    x_val, y_val = x_val[val_perm], y_val[val_perm]

    print(f"\nStrictly Balanced Dataset Ready:")
    print(f"  Training samples:   {len(y_train)} ({train_per_class} per class x {len(COMMAND_LABELS)} classes)")
    print(f"  Validation samples: {len(y_val)} ({samples_per_class - train_per_class} per class x {len(COMMAND_LABELS)} classes)")

    return x_train, y_train, x_val, y_val

def build_model(tf, shape, class_count):
    """High accuracy 3-stage CNN adhering to TFLite Micro registered ops."""
    inputs = tf.keras.Input(shape=shape, name="features")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(48, activation="relu")(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="probabilities")(x)
    model = tf.keras.Model(inputs, outputs)
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0015,
        decay_steps=1800,
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def verify_int8_model(tflite_path: Path, x_val: np.ndarray, y_val: np.ndarray, config: FeatureConfig):
    """Verify quantized model on validation data, silence, and room noise."""
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zero = inp['quantization']
    out_scale, out_zero = out['quantization']

    def predict_features(feat):
        q = np.clip(np.round(feat / in_scale) + in_zero, -128, 127).astype(np.int8)
        interpreter.set_tensor(inp['index'], np.expand_dims(q, 0))
        interpreter.invoke()
        res = interpreter.get_tensor(out['index'])[0]
        probs = (res.astype(np.float32) - out_zero) * out_scale
        return int(np.argmax(probs)), float(np.max(probs))

    # 1. Validation set accuracy
    correct = 0
    per_class_correct = [0] * len(COMMAND_LABELS)
    per_class_total = [0] * len(COMMAND_LABELS)

    for i in range(len(y_val)):
        pred, conf = predict_features(x_val[i])
        true_label = y_val[i]
        per_class_total[true_label] += 1
        if pred == true_label:
            correct += 1
            per_class_correct[true_label] += 1

    print("\n" + "=" * 60)
    print("QUANTIZED INT8 MODEL VERIFICATION")
    print("=" * 60)
    print(f"Overall INT8 Validation Accuracy: {correct / len(y_val) * 100:.2f}% ({correct}/{len(y_val)})")
    print("\nPer-Word Accuracy in INT8:")
    for idx, label in enumerate(COMMAND_LABELS):
        acc = per_class_correct[idx] / per_class_total[idx] * 100
        print(f"  {label:12s}: {acc:5.1f}% ({per_class_correct[idx]}/{per_class_total[idx]})")

    # 2. Silence test
    silence_feat = extract_log_mel(np.zeros(config.window_samples, dtype=np.float32), config)
    s_pred, s_conf = predict_features(silence_feat)
    print(f"\nDigital Silence Test -> '{COMMAND_LABELS[s_pred]}' ({s_conf * 100:.1f}%) [Expected: 'unknown']")

    # 3. Ambient Room Noise test
    for sigma in [0.01, 0.04, 0.08]:
        noise_audio = np.random.normal(0, sigma, config.window_samples).astype(np.float32)
        n_feat = extract_log_mel(noise_audio, config)
        n_pred, n_conf = predict_features(n_feat)
        print(f"Room Noise (sigma={sigma:0.2f}) -> '{COMMAND_LABELS[n_pred]}' ({n_conf * 100:.1f}%) [Expected: 'unknown']")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples-per-class", type=int, default=200, help="Equal samples for every word")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: python -m pip install -e .[train]") from exc

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    config = FeatureConfig()

    # Collect strictly balanced dataset
    x_train, y_train, x_val, y_val = collect_balanced_dataset(
        args.data_dir, config, samples_per_class=args.samples_per_class, seed=args.seed
    )

    # Distribute weight equally across all classes
    class_weights = {i: 1.0 for i in range(len(COMMAND_LABELS))}

    model = build_model(tf, x_train.shape[1:], len(COMMAND_LABELS))
    print("\nModel Architecture:")
    model.summary()

    print(f"\nTraining repeatedly across {args.epochs} epochs with equal class weights...")
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=32,
        class_weight=class_weights,
        verbose=1
    )

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"\nFinal Keras Validation Accuracy: {val_acc * 100:.2f}% (Loss: {val_loss:.4f})")

    # Prepare balanced representative dataset for full-INT8 quantization
    args.output_dir.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Exactly 20 calibration samples per class = 180 samples total for uniform quantization calibration
    rep_samples = []
    for label_idx in range(len(COMMAND_LABELS)):
        idx = np.where(y_train == label_idx)[0][:20]
        rep_samples.extend(x_train[idx])
    rep_samples = np.array(rep_samples, dtype=np.float32)

    def representative_dataset():
        for sample in rep_samples:
            yield [np.expand_dims(sample, axis=0)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("\nQuantizing model to full INT8...")
    model_bytes = converter.convert()
    tflite_path = args.output_dir / "keyword_spotter.tflite"
    tflite_path.write_bytes(model_bytes)
    print(f"Exported INT8 model: {tflite_path} ({len(model_bytes)} bytes)")

    (args.output_dir / "labels.txt").write_text("\n".join(COMMAND_LABELS) + "\n", encoding="utf-8")
    metadata = {
        "sample_rate": config.sample_rate,
        "window_samples": config.window_samples,
        "frame_count": config.frame_count,
        "mel_bins": config.mel_bins,
        "labels": list(COMMAND_LABELS),
        "input_shape": list(x_train.shape[1:]),
        "quantized": True,
        "samples_per_class": args.samples_per_class,
        "val_accuracy": float(val_acc)
    }
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Run verification suite on the INT8 model
    verify_int8_model(tflite_path, x_val, y_val, config)

if __name__ == "__main__":
    main()
