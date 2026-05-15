# ASL Teckenspråkstolk

Realtidsigenkänning av amerikanskt fingerspråk (ASL) med MediaPipe och en tränad SVM-modell.

Håll upp handen mot kameran och forma en bokstav — appen identifierar bokstaven och bygger upp ord allteftersom.

**Demo:** [kunskapskontroll.biscuitlab.com](https://kunskapskontroll.biscuitlab.com/)

---

## Hur det fungerar

1. **Landmarkextraktion** — MediaPipe detekterar 21 handlandmarks (x/y-koordinater) i varje kamerabild.
2. **Klassificering** — Koordinaterna skalas och matas in i en SVM-modell som förutsäger vilken bokstav handen formar.
3. **Ordsbyggnad** — En bokstav läggs till i ordet när den hålls stabil i 10 frames i rad. `del` raderar senaste bokstaven, `space` lägger till mellanslag.

---

## Appar

| Fil | Beskrivning |
|-----|-------------|
| `app_stream.py` | Huvudapp — realtidsstreaming via WebRTC (`streamlit-webrtc`). Ritar landmarks direkt på videoflödet. |
| `app.py` | Alternativ — snapshot-baserad kamera via `st.camera_input`. Enklare, fungerar i de flesta miljöer. |

---

## Kom igång

### Lokalt med venv

```bash
# Skapa och aktivera virtuell miljö
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# Installera beroenden
pip install -r requirements.txt

# Kör appen
streamlit run app_stream.py
# eller snapshot-versionen:
streamlit run app.py
```

### Med Docker

```bash
docker build -t asl-app .
docker run -p 8501:8501 asl-app
```

Öppna sedan `http://localhost:8501` i webbläsaren.

---

## Projektstruktur

```
kunskapskontroll_ai/
├── app.py                    # Streamlit-app (snapshot-kamera)
├── app_stream.py             # Streamlit-app (WebRTC-streaming)
├── asl_training_extended.ipynb  # Träningsnotebook för SVM-modellen
├── landmarkl_extrahering.ipynb  # Landmarkextraktion från träningsbilder
├── asl_landmarks.csv         # Extraherade landmarks (hela datasetet)
├── asl_landmarks_700.csv     # Balanserat urval (700 per klass)
├── asl_modell.pkl            # Tränad SVM-modell
├── asl_scaler.pkl            # StandardScaler
├── asl_label_encoder.pkl     # LabelEncoder (bokstav ↔ klass-ID)
├── hand_landmarker.task      # MediaPipe-modell för handdetektering
├── stora_modeller/           # Alternativa modellversioner
├── asl_alphabet_train/       # Träningsbilder (en mapp per bokstav)
├── asl_alphabet_test/        # Testbilder (en bild per bokstav)
├── Dockerfile
└── requirements.txt
```

---

## Modell

- **Algoritm:** Support Vector Machine (SVM)
- **Features:** 42 värden per bild (x och y för 21 landmarks)
- **Klasser:** A–Z samt `space` och `del`
- **Förbehandling:** StandardScaler

---

## Källor

- [Sign Language Recognition using Python & MediaPipe](https://medium.com/@m.rafaymct/sign-language-recognition-using-python-mediapipe-1de7638d2c1f)
- [MediaPipe Hand Landmarker](https://www.npmjs.com/package/@mediapipe/tasks-vision)
- YouTube-handledningar: [1](https://www.youtube.com/watch?v=MJCSjXepaAM) · [2](https://www.youtube.com/playlist?list=PLCC34OHNcOtoC6GglhF3ncJ5rLwQrLGnV) · [3](https://www.youtube.com/watch?v=01sAkU_NvOY&t=15425s)
