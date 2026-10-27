# DHVANI – Mind-to-Voice
Developer: **T.T.S**

A limited-command EEG classification assistive-communication prototype (YES / NO / WATER / HELP).
It is **not** unrestricted mind reading. A model trained on one person's EEG may not generalise to another person.

Pipeline: `BioAmp EXG -> Arduino UNO R4 Minima -> USB Serial -> backend (preprocess -> 2 s window -> features -> model) -> translation -> browser voice`

## Layout
- `firmware/dhvani_eeg/dhvani_eeg.ino` - Arduino IDE sketch (separate from the app)
- `backend/` - FastAPI + Python: `pipeline.py` (shared by training AND live), `model_service.py`, `app.py` (serial, WebSocket, REST)
- `frontend/index.html` - UI (served by the backend at http://localhost:8000). Set `window.DHVANI_API` to use another backend URL.

## Hardware and Arduino IDE
1. BioAmp EXG Pill / EEG hardware per Upside Down Labs docs (electrode placement, power, battery use are YOUR configuration; verify against their guide).
2. Connect each BioAmp OUT to an analog pin. Firmware default: channel 1 = A0, channel 2 = A1 (**configuration point, check your wiring**), common GND.
3. Arduino IDE -> Boards Manager -> install "Arduino UNO R4 Boards" -> select UNO R4 Minima -> open the .ino -> Upload.
4. Edit `NUM_CHANNELS`, `SAMPLE_RATE`, `BAUD` if needed; they must match DHVANI Settings. Close Serial Monitor/Plotter before using DHVANI (only one program can own the port).
5. Serial format: `value1,value2\n` per sample, 115200 baud, no other text.

The live waveform display uses a streaming 0.5–40 Hz band-pass and 50 Hz mains notch filter. Recordings remain raw ADC samples; training and prediction apply the shared offline preprocessing pipeline (DC removal, band-pass, notch, and per-channel z-score) to keep model inputs consistent.

## Run
```
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --port 8000
```
Open http://localhost:8000.

<!-- cd "C:\Users\user\OneDrive\Desktop\DHVANI SOFTWARE"

# Agar .venv pehle se hai, yeh step chhod sakte hain
py -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install fastapi "uvicorn[standard]" numpy scipy scikit-learn joblib pyserial pydantic

python -m uvicorn app:app --host 127.0.0.1 --port 8000 -->

## AI Assistant
The Assistant answers open-ended questions using the visible browser page, recent chat, live DHVANI status, and the model's general knowledge. It requires an OpenAI API key, which is read only by the backend and must not be placed in `index.html`.

Set the key in the same terminal before starting Uvicorn:
```
# PowerShell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_MODEL="gpt-4o-mini"  # optional; defaults to gpt-4o-mini
uvicorn app:app --port 8000
```
If the key is not configured, the assistant reports that setup is required. The page text and recent conversation are sent to OpenAI when a question is asked.

## Workflow
1. **Settings**: choose port, baud, channels, sampling rate, then **EEG Monitor -> Connect**.
2. **Recordings**: pick a command, countdown -> record -> auto-save `YES_01.csv` etc. to `backend/data/prod/recordings/` (+ JSON metadata: label, id, time, duration, fs, channels, samples, device). Record at least 2 per command (more is better).
3. **AI Model -> Prepare + Train + Evaluate**: preprocessing (DC removal, 0.5-40 Hz band-pass, 50 Hz notch, z-score), 2 s windows (1 s hop), features (stats + band powers, fixed order), RandomForest. The last recording of each command is held out for testing, so metrics are not inflated by window leakage. The final model is refit on all data and saved to `backend/models/prod/model.joblib`.
4. **Live Mind-to-Voice -> Start live prediction**. A full 2-second window is needed for the first prediction; after that, predictions update every 1 second. The browser speaks only after the configured number of stable, above-threshold windows (default: 2), so a stable command normally takes about 3 seconds from start to confirm. Low-confidence results stay silent; cooldown prevents repeated speech (Settings).

## Demo vs Production
Demo uses simulated waveform/predictions, every one marked SIMULATED, stored under `demo:*`. Production uses only real serial data and a real model, stored under `prod:*`. The app never switches modes automatically. "Clear data" affects only the current mode.

## Status honesty
Software Ready: yes. Hardware Tested: **no** until you have run it on real hardware. The signal-quality indicator is a heuristic (flat/saturated detection), not EEG validation.

## Troubleshooting
- *Backend unavailable*: start uvicorn; check port 8000.
- *Serial connection failed on Windows*: select the Arduino's USB serial port in **Settings**; do not select a `Standard Serial over Bluetooth link` port. Use a data-capable USB cable, install the UNO R4 board driver via Arduino IDE if needed, and refresh the port list. Close Arduino IDE Serial Monitor/Plotter and any other serial terminal before connecting because only one app can own a COM port at a time. If Windows reports `Access is denied`, also close any other DHVANI backend window/process and retry.
- *Serial connection failed on Linux*: check the USB cable/port, close Arduino Serial Monitor/Plotter, and add the user to `dialout` if needed.
- *Many invalid lines*: baud/channels/delimiter mismatch with firmware.
- *No signal / flat line*: check electrodes, BioAmp power, wiring to A0/A1.
- *Model Not Trained / training error*: record every command, at least 2 recordings each, at the same fs/channels as current Settings.
- *Voice unavailable*: use Chrome/Edge and install a system voice for the chosen language.

## Future
All hardware code is in `EEGService` (app.py). A Raspberry Pi reader (e.g. SPI ADC) can replace it without frontend changes.
## Not yet done
Vite/React split, backend unit tests, mains-frequency auto-detect, multi-user datasets, WebSocket-based training progress.
