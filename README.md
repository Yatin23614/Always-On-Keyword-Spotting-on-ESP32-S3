# Always-On Keyword Spotting on ESP32-S3

This project trains a small quantized CNN and runs it continuously on an ESP32-S3 with an I2S microphone.

The default model recognizes:

`start`, `stop`, `next`, `previous`, `volume_up`, `volume_down`, `lights_on`, `lights_off`, and `unknown`.

## 1. Prerequisites

### Host computer

Install:

- Python 3.10 or newer
- Git
- An ESP-IDF installation with the `idf.py` command available
- An ESP32-S3 board
- A 16-bit I2S digital microphone

For model training, install the optional TensorFlow dependency. For running the host tests, TensorFlow is not required.

### ESP-IDF TensorFlow Lite Micro

The firmware includes TensorFlow Lite Micro headers, but the library must be available as an ESP-IDF component. Add the Espressif TensorFlow Lite Micro component to the project using the ESP-IDF Component Manager or your organization’s approved TFLite Micro component, then ensure the component is visible to the `firmware/main` component.

## 2. Install the Python environment

From the repository root in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,dev]"
```

If PowerShell blocks activation, run the commands from a Python-enabled shell or use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train,dev]"
```

## 3. Run the host tests

```powershell
python -m pytest -q
```

The tests verify feature shape/finite output and command confidence/cooldown behavior.

## 4. Prepare the audio dataset

Create one directory per label under `data\raw`:

```text
data/raw/
  start/*.wav
  stop/*.wav
  next/*.wav
  previous/*.wav
  volume_up/*.wav
  volume_down/*.wav
  lights_on/*.wav
  lights_off/*.wav
  unknown/*.wav
```

Every file must be:

- WAV format
- 16,000 Hz
- Mono or stereo; stereo is averaged to mono
- Any duration; the loader pads or truncates to one second

Use multiple speakers, recording distances, microphones, and background-noise conditions. Put silence, unrelated speech, and noise in `unknown`. Keep speakers separated between training and evaluation recordings when measuring real accuracy.

## 5. Train and export the model

From the repository root:

```powershell
python scripts/train.py `
  --data-dir data/raw `
  --output-dir artifacts `
  --epochs 30
```

The command creates:

```text
artifacts/keyword_spotter.tflite
artifacts/labels.txt
artifacts/model_metadata.json
```

The exported model is fully int8 quantized and expects a feature tensor with shape `99 x 40 x 1`.

## 6. Embed the model in the firmware

Convert the model to a C++ byte array:

```powershell
python scripts/embed_model.py `
  artifacts/keyword_spotter.tflite `
  firmware/main/model_data.cc
```

Do this again after every model retraining. Do not edit the generated `model_data.cc` manually.

## 7. Configure the microphone

Edit [firmware/main/config.h](firmware/main/config.h) and set the pins for your board:

```cpp
#define KWS_I2S_BCLK GPIO_NUM_4
#define KWS_I2S_WS   GPIO_NUM_5
#define KWS_I2S_DATA GPIO_NUM_6
```

The current configuration assumes an I2S microphone providing 16-bit mono samples. Check your microphone breakout’s wiring and whether its left/right-select pin must be tied low or high.

## 8. Build and flash the ESP32-S3

Open an ESP-IDF PowerShell or command prompt, then run:

```powershell
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p COM5 flash monitor
```

Replace `COM5` with the serial port for your board. If the port is omitted, `idf.py` may detect it automatically.

On successful startup, the serial monitor should show the application booting. Recognized commands are logged in this form:

```text
I (...) kws_app: command=0 confidence=0.91
```

Command IDs follow the order in [kws/labels.py](kws/labels.py): `start` is 0 and `unknown` is 8.

## 9. Tune detection behavior

The main runtime controls are in [firmware/main/config.h](firmware/main/config.h):

```cpp
#define KWS_CONFIDENCE_THRESHOLD 0.80f
#define KWS_COOLDOWN_MS 800
```

- Increase the threshold to reduce false activations.
- Decrease it only if valid commands are being rejected.
- Increase the cooldown to prevent repeated commands.
- The gate resets after an `unknown` prediction, allowing the same command to be emitted again.

Tune these values with held-out recordings and real background noise, not with training samples.

## 10. Troubleshooting

### `No WAV files found`

Confirm that the command labels are directory names directly under `data/raw`, and that files use the `.wav` extension.

### `expected 16000 Hz`

Resample the recordings to 16 kHz before training.

### TensorFlow import error

Install the training extras:

```powershell
python -m pip install -e ".[train]"
```

### Model initialization fails on the ESP32

Confirm that:

1. `scripts/embed_model.py` was run after the latest training run.
2. The TensorFlow Lite Micro ESP-IDF component is installed.
3. The model is fully int8 quantized.
4. The ESP32-S3 has enough tensor-arena memory.

### No audio or unstable detections

Check BCLK, WS, DATA, ground, microphone power, and the microphone’s I2S channel-selection pin. Verify that the GPIO definitions match the physical wiring.

## Runtime contract

| Property | Value |
|---|---:|
| Sample rate | 16,000 Hz mono |
| Rolling window | 1,000 ms / 16,000 samples |
| Frame | 20 ms / 320 samples |
| Hop | 10 ms / 160 samples |
| Mel bands | 40 |
| Feature tensor | 99 x 40 x 1 |
| Model classes | 8 commands + `unknown` |
| Default confidence threshold | 0.80 |
| Default cooldown | 800 ms |
