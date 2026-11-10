"""Play wake-phrase and negative clips through the PC speaker for device_capture.py.

Writes a CSV log of (label, source, host start time, host end time).
"""
import argparse, csv, random, tempfile, time, wave, winsound
from pathlib import Path
import numpy as np


def write_wav(path, audio, rate):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(np.clip(audio, -32768, 32767).astype('<i2').tobytes())


def read_wav(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32), w.getframerate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log')
    p.add_argument('--positives', type=int, default=120)
    p.add_argument('--triplets', type=int, default=120)
    p.add_argument('--phrases', type=int, default=1000)
    p.add_argument('--command-triplets', type=int, default=30)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.seed)
    tmp = Path(tempfile.mkdtemp())
    jobs = []
    jarvis = sorted(Path('data/raw/hello_jarvis').glob('*.wav'))
    jobs += [('hello_jarvis', f) for f in rng.sample(jarvis, min(args.positives, len(jarvis)))]
    phrases = sorted(Path('data/raw/unknown_phrases').glob('*.wav'))
    jobs += [('unknown', f) for f in rng.sample(phrases, min(args.phrases, len(phrases)))]
    speech = sorted(Path('data/raw/unknown').glob('*.wav'))
    commands = [f for d in ('start', 'stop', 'next', 'previous', 'lights_on', 'lights_off', 'volume_up', 'volume_down')
                for f in sorted(Path('data/raw', d).glob('*.wav'))]
    for i in range(args.triplets + args.command_triplets):
        pool = speech if i < args.triplets else commands
        parts = []
        for f in rng.sample(pool, 3):
            a, rate = read_wav(f)
            parts += [a / max(np.abs(a).max(), 1) * 26000, np.zeros(int(rate * rng.uniform(0.05, 0.4)))]
        path = tmp / f'triplet_{i:03d}.wav'; write_wav(path, np.concatenate(parts), rate)
        jobs.append(('unknown', path))
    # broadband noise for measuring the speaker->ADC transfer function
    noise = tmp / 'noise.wav'; write_wav(noise, np.random.default_rng(0).normal(0, 6000, 16000 * 6), 16000)
    rng.shuffle(jobs)
    jobs = [('silence', None), ('noise', noise)] + jobs
    with open(args.log, 'w', newline='') as f:
        log = csv.writer(f); log.writerow(['label', 'source', 'start', 'end'])
        for n, (label, path) in enumerate(jobs):
            start = time.time()
            if path is None: time.sleep(30)
            else: winsound.PlaySound(str(path), winsound.SND_FILENAME)
            log.writerow([label, str(path), f'{start:.3f}', f'{time.time():.3f}']); f.flush()
            if n % 25 == 0: print(f'{n}/{len(jobs)}', flush=True)
            time.sleep(rng.uniform(0.9, 1.6))
    print('done')


if __name__ == '__main__':
    main()
