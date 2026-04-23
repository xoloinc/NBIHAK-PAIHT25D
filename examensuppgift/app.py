"""
ASL Teckenspråkstolk — Streamlit-app
Använder MediaPipe för att detektera handlandmarks i realtid
och en tränad SVM-modell för att klassificera bokstäver.

Kör med: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
st.set_page_config(page_title='ASL App')


# --- Ladda modell, scaler och label encoder ---
@st.cache_resource
def ladda_modell():
    modell = joblib.load('asl_modell.pkl')
    scaler = joblib.load('asl_scaler.pkl')
    le     = joblib.load('asl_label_encoder.pkl')
    return modell, scaler, le



# --- MediaPipe setup ---
@st.cache_resource
def ladda_detektor():
    try:
        base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        detektor = vision.HandLandmarker.create_from_options(options)
        return detektor
    except Exception as e:
        st.error(f'Fel vid laddning av detektor: {e}')
        st.stop()


st.write('Laddar modell...')
modell, scaler, le = ladda_modell()
st.write('Modell laddad!')
st.write('Laddar detektor...')
try:
    detektor = ladda_detektor()
    st.write('Detektor laddad!')
except Exception as e:
    st.error(f'Fel: {e}')
    st.stop()
st.write('Detektor laddad!')


# --- Sidlayout ---
st.title('ASL Teckenspråkstolk')
st.write('Håll upp en hand mot kameran och forma en bokstav i amerikanskt fingerspråk.')

col1, col2 = st.columns([2, 1])

with col1:
    kamera_bild = st.camera_input('Kamera')

with col2:
    st.subheader('Prediktion')
    bokstav_placeholder = st.empty()
    ord_placeholder     = st.empty()
    konfidens_placeholder = st.empty()

# --- Ord-buffer ---
if 'ord' not in st.session_state:
    st.session_state.ord = ''

if 'foreg_bokstav' not in st.session_state:
    st.session_state.foreg_bokstav = ''

if 'raknar' not in st.session_state:
    st.session_state.raknar = 0

# --- Knappar ---
col3, col4 = st.columns(2)
with col3:
    if st.button('Lägg till mellanslag'):
        st.session_state.ord += ' '
with col4:
    if st.button('Rensa ord'):
        st.session_state.ord = ''

# --- Bearbeta kamerabild ---
if kamera_bild is not None:
    # Läs in bilden
    fil_bytes = np.asarray(bytearray(kamera_bild.read()), dtype=np.uint8)
    bild = cv2.imdecode(fil_bytes, cv2.IMREAD_COLOR)
    bild_rgb = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)

    # Kör MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bild_rgb)
    resultat = detektor.detect(mp_image)

    if resultat.hand_landmarks:
        # Extrahera landmarks
        landmarks = resultat.hand_landmarks[0]
        rad = []
        for lm in landmarks:
            rad.append(lm.x)
            rad.append(lm.y)

        # Skala och prediktera
        X = scaler.transform([rad])
        pred = modell.predict(X)[0]
        bokstav = le.inverse_transform([pred])[0]

        # Konfidens (om modellen stödjer det)
        try:
            proba = modell.predict_proba(X)[0]
            konfidens = round(max(proba) * 100, 1)
        except AttributeError:
            konfidens = None

        # Visa bokstav
        bokstav_placeholder.markdown(f'## {bokstav}')
        if konfidens:
            konfidens_placeholder.write(f'Konfidens: {konfidens}%')

        # Bygg upp ord — lägg till bokstav när den hålls stabil
        if bokstav == st.session_state.foreg_bokstav:
            st.session_state.raknar += 1
        else:
            st.session_state.raknar = 0
            st.session_state.foreg_bokstav = bokstav

        if st.session_state.raknar == 10:
            if bokstav == 'del':
                st.session_state.ord = st.session_state.ord[:-1]
            elif bokstav != 'space':
                st.session_state.ord += bokstav
            st.session_state.raknar = 0

    else:
        bokstav_placeholder.write('Ingen hand detekterad')
        konfidens_placeholder.empty()

# Visa nuvarande ord
ord_placeholder.markdown(f'**Ord:** {st.session_state.ord}')
