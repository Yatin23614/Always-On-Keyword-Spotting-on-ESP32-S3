"""Interactive microphone recorder for collecting custom voice samples."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from scipy.io import wavfile

try:
    import sounddevice as sd
except ImportError:
    raise SystemExit("Please install sounddevice: python -m pip install sounddevice")

from kws.labels import COMMAND_LABELS

SAMPLE_RATE = 16000
DURATION = 1.2  # Record 1.2s to easily capture the full utterance

def record_sample():
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()
    
    # Peak normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0.05:
        audio = audio / max_val
    
    # Center or pad/crop to exactly 1.0 second (16000 samples)
    if len(audio) > SAMPLE_RATE:
        # Find peak energy center
        energy = audio ** 2
        window_size = SAMPLE_RATE
        best_start = 0
        max_energy = 0
        for i in range(0, len(audio) - window_size, 320):
            e = np.sum(energy[i:i + window_size])
            if e > max_energy:
                max_energy = e
                best_start = i
        audio = audio[best_start:best_start + window_size]
    elif len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
        
    return (audio * 32767).astype(np.int16)

def main():
    parser = argparse.ArgumentParser(description="Record your own voice for keyword spotting")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--count", type=int, default=15, help="Number of samples to record per label")
    parser.add_argument("--label", type=str, choices=list(COMMAND_LABELS), help="Record a specific label only")
    args = parser.parse_args()

    labels_to_record = [args.label] if args.label else [l for l in COMMAND_LABELS if l != "unknown"]

    print("=" * 60)
    print("Voice Sample Recorder for ESP32 Keyword Spotter")
    print(f"Sample Rate: {SAMPLE_RATE} Hz | Samples per label: {args.count}")
    print("=" * 60)
    print("Press Enter to begin recording, then say the prompt clearly!\n")

    for label in labels_to_record:
        target_dir = args.data_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Find existing user files count
        existing = len(list(target_dir.glob("user_*.wav")))
        print(f"\n>>> Preparing to record: [{label.upper()}] (currently has {existing} user recordings)")
        input(f"Press Enter when ready to start recording '{label}'...")

        for i in range(1, args.count + 1):
            idx = existing + i
            print(f"\n[{i}/{args.count}] Get ready to say: '{label}' in 2 seconds...")
            time.sleep(1.0)
            print("  3...")
            time.sleep(0.5)
            print("  2...")
            time.sleep(0.5)
            print("  1...")
            time.sleep(0.5)
            print(f"  >>> SPEAK NOW: '{label}' <<<")
            
            samples = record_sample()
            
            # Check volume
            peak = np.max(np.abs(samples))
            if peak < 3000:
                print(f"  [Warning: Very quiet! Peak={peak}/32767. Try speaking closer to mic]")
            else:
                print(f"  [OK! Recorded clean audio, Peak={peak}]")

            save_path = target_dir / f"user_{idx:03d}.wav"
            wavfile.write(save_path, SAMPLE_RATE, samples)
            print(f"  Saved -> {save_path.name}")
            time.sleep(0.4)

    print("\n" + "=" * 60)
    print("Recording complete! You can now retrain the model with:")
    print("  python scripts/train.py --data-dir data/raw --epochs 30")
    print("  python scripts/embed_model.py artifacts/keyword_spotter.tflite firmware/main/model_data.cc")
    print("  idf.py flash monitor")
    print("=" * 60)

if __name__ == "__main__":
    main()

