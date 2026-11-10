"""Play wake phrases and negatives through the PC speaker and check the ESP32's detections.

Reads the detection counter from /api/status before and after each clip.
"""
import argparse, json, random, time, urllib.request, winsound
from pathlib import Path


def status(host):
    with urllib.request.urlopen(f'http://{host}/api/status', timeout=3) as r:
        return json.loads(r.read().decode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='192.168.1.2')
    p.add_argument('--positives', type=int, default=15)
    p.add_argument('--negatives', type=int, default=15)
    p.add_argument('--seed', type=int, default=99)
    args = p.parse_args()
    rng = random.Random(args.seed)
    jobs = [('hello_jarvis', f) for f in rng.sample(sorted(Path('data/raw/hello_jarvis').glob('*.wav')), args.positives)]
    jobs += [('unknown', f) for f in rng.sample(sorted(Path('data/raw/unknown_phrases').glob('*.wav')), args.negatives)]
    rng.shuffle(jobs)
    hits = {'hello_jarvis': 0, 'unknown': 0}
    time.sleep(3.5)  # let any earlier detection's cooldown expire
    for label, path in jobs:
        before = status(args.host)['count']
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        shown = ''
        for _ in range(12):  # the phrase stays in the 2.8 s window for a while
            time.sleep(0.3)
            s = status(args.host)
            if s['id'] == 0: shown = f"dashboard: \"{s['command']}\" ({s['confidence'] * 100:.0f}%)"
        fired = status(args.host)['count'] - before
        hits[label] += fired > 0
        print(f"{label:13s} {path.name:16s} detections={fired} {shown}", flush=True)
    print(f"\nwake phrases detected: {hits['hello_jarvis']}/{args.positives}")
    print(f"negatives falsely detected: {hits['unknown']}/{args.negatives}")


if __name__ == '__main__':
    main()
