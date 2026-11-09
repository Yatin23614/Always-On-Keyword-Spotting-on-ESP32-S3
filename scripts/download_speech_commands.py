"""Download Speech Commands v0.02 and prepare it as unknown/background audio.

The public dataset does not contain the project's custom command labels.
It is used here only to seed the unknown class. Record custom command examples
yourself under data/raw/<label>/ before training.
"""
import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
KNOWN_PROJECT_LABELS = {
    "start", "stop", "next", "previous", "volume_up", "volume_down",
    "lights_on", "lights_off", "unknown",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--limit", type=int, default=5000,
                        help="maximum unknown WAV files to copy; use 0 for all")
    parser.add_argument("--url", default=URL, help="dataset archive URL")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "speech_commands_v0.02.tar.gz"
        print(f"Downloading {args.url}")
        urllib.request.urlretrieve(args.url, archive)
        extract_dir = Path(temp) / "extracted"
        extract_dir.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_dir)
        source = extract_dir
        wavs = sorted(
            path for path in source.rglob("*.wav")
            if path.parent.name not in KNOWN_PROJECT_LABELS
            and path.parent.name != "_background_noise_"
        )
        if args.limit:
            wavs = wavs[:args.limit]
        target = args.output_dir / "unknown"
        target.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(wavs):
            shutil.copy2(path, target / f"speech_commands_{index:06d}.wav")
        print(f"Copied {len(wavs)} WAV files to {target}")
        print("Record custom command examples before training.")

if __name__ == "__main__":
    main()
