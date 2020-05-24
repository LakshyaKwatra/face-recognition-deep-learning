import face_recognition as fr
import os
import cv2
import time
import numpy as np
import pandas as pd

KNOWN_FACES_DIR = 'known'

known_faces = []
known_names = []


for name in os.listdir(KNOWN_FACES_DIR):
    print(f"caching {name}'s data")
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print(f"looking at {filename}")
        try:
            image = fr.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
            encoding = fr.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)
            
        except:
            print("no face detected")
                
kf = pd.DataFrame(np.array(known_faces))
kn = pd.DataFrame(np.array(known_names))
kf.to_csv('known_encode.csv')
kn.to_csv('known_name.csv')
