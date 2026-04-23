"""
Konvertera ASL Alphabet bildern, till csv med landmarks för att träna modellen.
curl -o https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

import os
import csv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

#Bildmapp och output CSV
DATA_MAPP  = 'C:\\workspace\\privat\\skola\\examensprojekt\\asl_alphabet_train'
OUTPUT_CSV = 'asl_landmarks.csv'
MAX_BILDER = 500  # antal bilder per bokstav

""" --- Initiera handlandmarker --- Använd MediaPipe's hand landmarker för att extrahera 21 landmarks per hand."""
base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# --- Bygg CSV-header ---
# 21 landmarks x 2 koordinater (x, y) = 42 features + label
#header = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y')]
header =  []
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
#Varje mapp i bokstavavmpapen
for bokstav in bokstavsmapp:
    ##hämta pathc
    mapp_path = os.path.join(DATA_MAPP, bokstav)
    ## validera är mapp..
    if not os.path.isdir(mapp_path):
        continue
    
    bilder = [f for f in os.listdir(mapp_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_BILDER]

    for filnamn in tqdm(bilder, desc=f'{bokstav} ({len(bilder)} bilder)'):
        file_path = os.path.join(mapp_path, filnamn)

        try:
            #här kör konvertering
            mp_image = mp.Image.create_from_file(file_path) #läser in bilden
            resultat = detector.detect(mp_image) #här detekterar den en hand med hjälpa av hand_landmarker.task och får ut "landmarks" som är typ koordinater.
        except Exception:
            missade += 1
            continue

        if not resultat.hand_landmarks:
            missade += 1
            continue
        
        landmarks = resultat.hand_landmarks[0]
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

detector.close()

print(f'\nKlart!')
print(f'Sparade {totalt} rader till {OUTPUT_CSV}')
print(f'Missade {missade} bilder (ingen hand detekterad eller läsfel)')
