"""
var i venv (source venv/bin/activate linux,venv\Scripts\activate win)
streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

## vi labbar lite här med olika storla modeller..
MAPP = './'


# --- Ladda modell, scaler och label encoder ---
@st.cache_resource
def ladda_modell():
    modell = joblib.load(f'{MAPP}/new_asl_modell.pkl')
    scaler = joblib.load(f'{MAPP}/new_asl_scaler.pkl')
    le     = joblib.load(f'{MAPP}/new_asl_label_encoder.pkl')
    return modell, scaler, le

modell, scaler, le = ladda_modell()

# --- Sidlayout ---
st.title('ASL Teckenspråkstolk')
st.write('Håll upp handen för bokstav.')

if 'bokstav' not in st.session_state:
    st.session_state.bokstav = ''
if 'ord' not in st.session_state:
    st.session_state.ord = ''
if 'konfidens' not in st.session_state:
    st.session_state.konfidens = 0
if 'raknar' not in st.session_state:
    st.session_state.raknar = 0
if 'foreg_bokstav' not in st.session_state:
    st.session_state.foreg_bokstav = ''



# Denna var för kjomplicerad.. klipp och klistrat utifrån det jag hittat online, saknar TTS, den är dålig på ORD, mellans slag ettc.
#vi använder opencv för att plocka ut bild
class Processor(VideoProcessorBase):
    def __init__(self):
        self.ord = '' 
        self.frame_count = 0
        # Egen detektor per instans — thread-safe
        base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        self.detektor = vision.HandLandmarker.create_from_options(options)
        self.senaste_bokstav = ''
        self.senaste_konfidens = 0
        self.raknar = 0
        self.foreg_bokstav = ''

    def recv(self, frame):
        bild = frame.to_ndarray(format='bgr24')
        self.frame_count += 1
        if self.frame_count % 5 == 0:
            bild_rgb = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bild_rgb)
            resultat = self.detektor.detect(mp_image)

            if resultat.hand_landmarks:
                landmarks = resultat.hand_landmarks[0]
                rad = []
                for lm in landmarks:
                    rad.append(lm.x)
                    rad.append(lm.y)

                X = scaler.transform([rad])
                pred = modell.predict(X)[0]
                bokstav = le.inverse_transform([pred])[0]

                # Konfidens
                try:
                    proba = modell.predict_proba(X)[0]
                    konfidens = int(max(proba) * 100)
                except AttributeError:
                    konfidens = 0

                self.senaste_bokstav = bokstav
                self.senaste_konfidens = konfidens

                # Bygg upp ord — håll bokstav stabil i 5 frames
                if bokstav == self.foreg_bokstav:
                    self.raknar += 1
                else:
                    self.raknar = 0
                    self.foreg_bokstav = bokstav
                ##.. vi kör var 10..5 för snabbt, 15 för långsamt
                if self.raknar == 10:
                    if bokstav == 'del':
                        self.ord = self.ord[:-1]
                    elif bokstav == 'space':
                        self.ord += ' '
                    else:
                        self.ord += bokstav
                    self.raknar = 0
                # Rita landmarks
                for lm in landmarks:
                    h, w = bild.shape[:2]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(bild, (cx, cy), 5, (0, 255, 0), -1)

                # Visa text på bild
                cv2.putText(bild, f'{bokstav} {konfidens}%',
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1.5, (0, 255, 0), 2)
            else:
                self.senaste_bokstav = ''
                cv2.putText(bild, 'Ingen hand',
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1.0, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(bild, format='bgr24')


#video kamrea streaming
ctx = webrtc_streamer(
    key='asl',
    video_processor_factory=Processor,
    media_stream_constraints={'video': True, 'audio': False},
    async_processing=True
)
if ctx.video_processor:
    st.metric('Bokstav', ctx.video_processor.senaste_bokstav or '—')
    st.metric('Konfidens', f"{ctx.video_processor.senaste_konfidens}%")
    st.markdown(f'### {ctx.video_processor.ord or "—"}')

col1, col2 = st.columns(2)

with col1:
    st.metric('Bokstav', st.session_state.bokstav or '—')
    st.metric('Konfidens', f"{st.session_state.konfidens}%")

with col2:
    st.subheader('Ord')
    st.markdown(f'### {st.session_state.ord or "—"}')

col3, col4 = st.columns(2)
with col3:
    if st.button('Mellanslag'):
        st.session_state.ord += ' '
with col4:
    if st.button('Rensa'):
        st.session_state.ord = ''

if ctx.state.playing:
    import time
    time.sleep(0.5)
    st.rerun()

# --- ASL-teckentabell ---
st.divider()
st.subheader('ASL-teckentabell')

bokstaver = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['space', 'del']
bildmapp = './asl_alphabet_test'
kolumner = st.columns(7)
for i, tecken in enumerate(bokstaver):
    filnamn = f'{bildmapp}/{tecken}_test.jpg'
    with kolumner[i % 7]:
        try:
            st.image(filnamn, caption=tecken, use_container_width=True)
        except Exception:
            st.write(tecken)

