"""Train and export the full-int8 keyword spotter."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from kws.features import FeatureConfig, extract_log_mel
from kws.labels import COMMAND_LABELS, LABEL_TO_ID

def load_wav(path: Path, config: FeatureConfig) -> np.ndarray:
    rate, audio = wavfile.read(path)
    if rate != config.sample_rate:
        raise ValueError(f"{path}: expected {config.sample_rate} Hz, got {rate}")
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio / max(float(np.max(np.abs(audio))), 1.0)

def collect(data_dir: Path, config: FeatureConfig):
    features, targets = [], []
    for label in COMMAND_LABELS:
        for path in sorted((data_dir / label).glob("*.wav")):
            features.append(extract_log_mel(load_wav(path, config), config))
            targets.append(LABEL_TO_ID[label])
    if not features:
        raise ValueError(f"No WAV files found under {data_dir}")
    return np.stack(features), np.asarray(targets, dtype=np.int64)

def build_model(tf, shape, class_count):
    inputs = tf.keras.Input(shape=shape, name="features")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(24, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="probabilities")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: python -m pip install -e .[train]") from exc
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    config = FeatureConfig()
    x, y = collect(args.data_dir, config)
    order = np.random.permutation(len(y))
    x, y = x[order], y[order]
    split, val_split = int(len(y) * .8), int(len(y) * .9)
    model = build_model(tf, x.shape[1:], len(COMMAND_LABELS))
    model.fit(x[:split], y[:split], validation_data=(x[split:val_split], y[split:val_split]),
              epochs=args.epochs, batch_size=32, verbose=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    representative = tf.data.Dataset.from_tensor_slices(x[:split]).batch(1).take(200)
    converter.representative_dataset = lambda: ([sample] for sample in representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    (args.output_dir / "keyword_spotter.tflite").write_bytes(model_bytes)
    (args.output_dir / "labels.txt").write_text("\n".join(COMMAND_LABELS) + "\n", encoding="utf-8")
    metadata = {"sample_rate": config.sample_rate, "window_samples": config.window_samples,
                "frame_count": config.frame_count, "mel_bins": config.mel_bins,
                "labels": list(COMMAND_LABELS), "input_shape": list(x.shape[1:]), "quantized": True}
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir / 'keyword_spotter.tflite'} ({len(model_bytes)} bytes)")

if __name__ == "__main__":
    main()
