"""Stream the ESP32's microphone audio from /api/audio to disk.

Each record in the output file is: uint32 seq, float64 host receive time, then
KWS_SLIDE_SAMPLES int16 samples. Stop with Ctrl+C or --seconds.
"""
import argparse, struct, time, urllib.request
import numpy as np

SLIDE = 3000
HEADER = struct.Struct('<IfhBB')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output')
    p.add_argument('--host', default='192.168.1.2')
    p.add_argument('--seconds', type=float, default=1e9)
    args = p.parse_args()
    since, got, missing, t0 = 0, 0, 0, time.time()
    with open(args.output, 'ab') as out:
        while time.time() - t0 < args.seconds:
            try:
                with urllib.request.urlopen(f'http://{args.host}/api/audio?since={since}', timeout=3) as r:
                    raw = r.read()
            except Exception as e:  # keep going through Wi-Fi hiccups
                print('fetch error:', e, flush=True); time.sleep(0.3); continue
            now = time.time(); rec = HEADER.size + SLIDE * 2
            for k in range(len(raw) // rec):
                seq = HEADER.unpack_from(raw, k * rec)[0]
                if seq <= since: continue
                if since and seq != since + 1: missing += seq - since - 1
                out.write(struct.pack('<Id', seq, now)); out.write(raw[k * rec + HEADER.size:(k + 1) * rec])
                since = seq; got += 1
            out.flush()
            if got and got % 200 == 0: print(f'{got} slides, {missing} missing', flush=True)
            time.sleep(0.05)
    print(f'done: {got} slides, {missing} missing')


if __name__ == '__main__':
    main()
