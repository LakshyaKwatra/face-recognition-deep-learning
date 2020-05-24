import face_recognition as fr
import os
import cv2
import numpy as np
import pandas as pd

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


UNKNOWN_FACES_DIR = 'unknown'
TOLERANCE = 0.5
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = 'hog' #cnn

print('loading known faces...')


known_faces = list(np.array(pd.read_csv('known_encode.csv').iloc[:,1:])) 
known_names = list(np.array(pd.read_csv('known_name.csv').iloc[:,1:]))

print('processing unknown faces')
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = fr.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    image = ResizeWithAspectRatio(image, width = 500)
    locations = fr.face_locations(image,model = MODEL)
    encodings = fr.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    for face_encoding, face_location in zip(encodings, locations):
        results = fr.compare_faces(known_faces,face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            top_left = (face_location[3],face_location[0])
            bottom_right = (face_location[1],face_location[2])
            color = [0,255,0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            top_left = (face_location[3],face_location[2])
            bottom_right = (face_location[1],face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image,match[0],(face_location[3]+10,face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),FONT_THICKNESS)
        else:
            top_left = (face_location[3],face_location[0])
            bottom_right = (face_location[1],face_location[2])
            color = [255,0,0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
    cv2.imshow(filename,image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)

