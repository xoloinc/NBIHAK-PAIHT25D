"""
Extraherar hand-landmarks från ASL-bilder med MediaPipe.
Spottar ut en CSV-fil redo att använda i notebooken.

Kör med: python extrahera_landmarks.py
"""

import os
import cv2
import csv
import mediapipe as mp
from tqdm import tqdm  # progressbar

# --- Inställningar ---
DATA_MAPP   = '/media/macke/D/school/NBIHAK-PAIHT25D/examensuppgift/asl_alphabet_train/asl_alphabet_train/'
OUTPUT_CSV  = 'asl_landmarks.csv'
MAX_BILDER  = 500  # antal bilder per bokstav (87k totalt är onödigt mycket, 500 räcker)

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # vi kör på stillbilder, inte video
    max_num_hands=1,              # vi letar bara efter en hand
    min_detection_confidence=0.5
)

# --- Bygg CSV-header ---
# 21 landmarks x 2 koordinater (x, y) = 42 features + label
header = []
for i in range(21):
    header.append(f'x{i}')
    header.append(f'y{i}')
header.append('label')

# --- Loopa igenom alla bokstavsmappar ---
rader        = []
missade      = 0
totalt       = 0
bokstavsmapp = sorted(os.listdir(DATA_MAPP))

print(f'Hittade {len(bokstavsmapp)} mappar: {bokstavsmapp}\n')

for bokstav in bokstavsmapp:
    mapp_path = os.path.join(DATA_MAPP, bokstav)

    if not os.path.isdir(mapp_path):
        continue

    bilder = os.listdir(mapp_path)[:MAX_BILDER]

    for filnamn in tqdm(bilder, desc=f'{bokstav} ({len(bilder)} bilder)'):
        fil_path = os.path.join(mapp_path, filnamn)

        # Läs in bilden
        bild = cv2.imread(fil_path)
        if bild is None:
            missade += 1
            continue

        # Konvertera BGR → RGB (MediaPipe vill ha RGB)
        bild_rgb = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)

        # Kör MediaPipe
        resultat = hands.process(bild_rgb)

        if not resultat.multi_hand_landmarks:
            # Ingen hand hittades i bilden — hoppa över
            missade += 1
            continue

        # Extrahera x, y för alla 21 landmarks
        landmarks = resultat.multi_hand_landmarks[0].landmark
        rad = []
        for lm in landmarks:
            rad.append(round(lm.x, 6))
            rad.append(round(lm.y, 6))
        rad.append(bokstav)

        rader.append(rad)
        totalt += 1

# --- Spara CSV ---
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rader)

hands.close()

print(f'\nKlart!')
print(f'Sparade {totalt} rader till {OUTPUT_CSV}')
print(f'Missade {missade} bilder (ingen hand detekterad eller läsfel)')
