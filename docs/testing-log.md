# "Hello Jarvis" on-device testing log

This is a record of the debugging and testing session that got the ESP32 DevKit V1 to reliably detect "Hello Jarvis" and show it on the web dashboard. It covers what was wrong, what changed, how it was tested, and the results.

## Setup

- **Board:** ESP32 DevKit V1 on COM3 (CP210x) with an analog microphone on the ADC.
- **Toolchain:** ESP-IDF v6.1. Build and flash with `idf.py -p COM3 build flash`.
- **Audio:** ADC continuous mode at 20 kHz, decimated to 10 kHz.
- **Windowing:** each inference window is 28,000 samples (2.8 s). The window slides every 3,000 samples (300 ms).
- **Features:** 300-sample frames with a 200-sample hop. 40 log-mel bands from 60 to 4800 Hz, 512-point FFT, 139 frames per window. Each window is normalized by its own mean and standard deviation.
- **Model:** TFLite Micro, INT8, two classes (`hello_jarvis`, `unknown`).
- **Dashboard:** `http://<device-ip>/`. The board prints this URL on the serial log after it joins Wi-Fi.

## Symptoms at the start

- The dashboard reported "Hello Jarvis" for almost any sound loud enough to pass the speech gate. Confidence was 0.87–0.99, even for unrelated speech.
- Inference took about 356 ms, but a new window arrives every 300 ms. The worker kept falling behind and losing audio.

## Root causes

1. **The model had never heard the device's microphone.** It was trained only on clean text-to-speech clips. The ESP32's microphone and ADC change the sound so much that every device window looked like the positive class. The ~5,000 real speech clips in `data/raw/unknown` were never used.
2. **Inference was too slow.** The old model, plus re-computing the features for a whole window on every slide, did not fit in the 300 ms budget.
3. **Smaller defects:**
   - The firmware did not check the model's input size against the features it wrote.
   - The detection cooldown reset whenever confidence dipped, so one utterance could fire several times.
   - Samples already read were thrown away when a read returned an error.
   - A Wi-Fi or dashboard startup failure crashed the board.
   - The Python feature settings did not match the firmware.

## Fixes

### Firmware (`firmware/main/`)

- **Incremental features** (`feature_extractor.cc`): each 300 ms of audio adds its new log-mel frames to a ring buffer, instead of re-computing the whole 2.8 s window. This removed a 56 KB window buffer and takes 8–9 ms per slide. Frame alignment between slides is handled by a carry of `kCarry = 200` samples. With it, the features match the Python training code within one quantization step.
- **Model I/O checks** (`keyword_inference.cc`): at startup the firmware checks that the model takes `139*40` input bytes and returns 2 output bytes, and fails cleanly otherwise. The tensor arena uses 71,296 of 98,304 bytes.
- **Cooldown:** 3,000 ms, longer than the 2.8 s window, and no reset when confidence dips. One utterance now produces one detection.
- **Audio capture:** a read error keeps the samples already read. An interrupt callback counts ADC buffer overflows.
- **Dashboard** (`web_dashboard.cc`):
  - `GET /api/audio?since=N` streams the last few raw 300 ms audio slides. It copies them while holding the lock and sends them after releasing it, so slow HTTP clients don't stall audio capture.
  - `/api/status` now also reports `uptime_ms`, `audio_samples`, `adc_overflows`, `slides` and `count` (detections since boot).
  - The page shows "✅ Hello Jarvis detected" with the confidence and the detection count.
- **Startup:** a Wi-Fi or dashboard failure is logged and the board keeps running instead of aborting.
- **Wi-Fi credentials** now live in the gitignored file `firmware/main/wifi_credentials.h`. Copy `wifi_credentials.example.h` to create it.

### Data collection through the real device

- `scripts/play_dataset.py` plays the clips through the PC speaker in this order: 30 s of silence, a 6 s noise burst used to measure the speaker-to-microphone response, then shuffled wake phrases, negative phrases and runs of Speech Commands words. It writes a CSV with the time each clip was played.
- `scripts/device_capture.py` polls `/api/audio` and saves everything the board's microphone heard to a `.bin` file.
- `scripts/train_jarvis.py` lines up each played clip with the recording by matching their loudness envelopes. It fits a clock rate (about 9,974 samples per host second) and an offset, then refines each clip within ±0.4 s. The PC clock alone was not accurate enough.
- **Recording sessions:**
  - Session 1 was discarded. The board lost audio while an HTTP send held the lock, which caused drift and jumps. This was fixed by the copy-then-send change above.
  - Session 2 was discarded. It was recorded while training ran on the same PC, which caused capture lag.
  - Session 3 is the good one: 22 minutes, recorded with the PC otherwise idle.

### Training (`scripts/train_jarvis.py`)

- **Training data:**
  - Session 3 device windows, repeated three times.
  - A simulated set passed through the measured speaker-to-microphone response, with a random spectral tilt, real device noise, and peak levels of 5,000–30,000. It has 720 wake phrases, 400 negative phrases, 1,000 Speech Commands word runs, 300 command runs, 250 partial wake phrases used as negatives, and 200 noise bursts.
- **Model:** a 16-filter stride-2 conv, then three depthwise-separable blocks (32, 48 and 64 channels), global average pooling, dropout 0.3, and a dense layer with 2 outputs. It uses about a third of the old model's compute, and the INT8 model is 22,472 bytes.
- **INT8 evaluation** now clips quantized inputs the same way the firmware does.

## Results

### Offline: INT8 model on held-out device recordings

| Threshold | Wake phrases detected | Wake windows | False-positive windows |
|-----------|-----------------------|--------------|-----------------------|
| 0.50 | 20 / 20 (100%) | 98.9% | 0 / 470 (0.00%) |
| 0.65 | 20 / 20 (100%) | 98.9% | 0 / 470 (0.00%) |
| 0.80 | 20 / 20 (100%) | 98.9% | 0 / 470 (0.00%) |
| 0.90 | 20 / 20 (100%) | 98.9% | 0 / 470 (0.00%) |

### Live: speaker-to-board test (`scripts/test_live_trigger.py`)

This test played the clips through the PC speaker and counted detections using `count` from `/api/status`.

- **Wake phrases detected: 15 / 15**, each with exactly one detection. The dashboard showed "Hello Jarvis detected".
- **Negatives falsely detected: 2 / 15** (`neg_0108`, `neg_0087`), at 71–73% confidence.

### Serial log during the live test

```
I (8000)  kws_app: timing: features 9 ms, inference 0 ms (budget 300 ms) | audio 0 Hz, ADC overflows 0 | free heap 25528
I (14016) kws_app: timing: features 8 ms, inference 0 ms (budget 300 ms) | audio 9973 Hz, ADC overflows 0 | free heap 26972
I (20032) kws_app: timing: features 8 ms, inference 0 ms (budget 300 ms) | audio 9973 Hz, ADC overflows 0 | free heap 23964
```

When three more wake phrases were played, all three fired (`command=0`) with confidence 0.996, 0.980 and 0.996.

### Host tests

`python -m pytest -q`: 4 passed.

## Known limitations and next steps

- **Your own voice is untested.** Every test so far used synthetic voices played through a PC speaker. To fix weak detection on real voices, record yourself with `play_dataset.py` and `device_capture.py`, then retrain.
- **False alarms.** About 2 in 15 negative phrases fired at 71–73% confidence. Raising `KWS_CONFIDENCE_THRESHOLD` in `config.h` from 0.65 to about 0.85 would block those. Some live wake detections were only 80–83%, so test after changing it.
- **Inference time is unmeasured.** The serial log reports `inference 0 ms`, probably because the timing sample usually lands on silent windows that skip inference. Zero ADC overflows during detections shows the board keeps up.

## Reproducing

```powershell
# 1. Flash the firmware (after creating firmware/main/wifi_credentials.h)
cd firmware; idf.py -p COM3 build flash monitor; cd ..

# 2. Record a device session: start the capture, then play the dataset (PC otherwise idle)
python scripts/device_capture.py data/device/session4.bin --host <device-ip>
python scripts/play_dataset.py data/device/session4.csv

# 3. Retrain (writes artifacts/keyword_spotter.tflite and firmware/main/model_data.cc)
python scripts/train_jarvis.py

# 4. Re-flash, then run the live test
python scripts/test_live_trigger.py --host <device-ip>
```
