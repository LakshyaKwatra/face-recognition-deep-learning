# face-recognition-deep-learning

# Tools and Libraries required
numpy, pandas, dlib, openCV, Pillow, scipy, Click, face_recognition

# File and Directory description
cache.py: used to cache the train image data, so that you don't have to scan the whole train directory time and again.
facial_rec.py: used to train the data with CNN/Hog model and display the labels and boxes around the recognized faces.
known_encode.csv: stores the encodings data gathered from train images.
known_name.csv: stores the name labels corresponding to the encodings in known_encode.csv
known: this directory contains subdirectories with the same name as the name label of the images inside those subdirectories.
unknown: this directory contains images to test our face recognition program on.

# Instructions to run:
To cache the train image data present in directory 'known', first run cache.py

>python cache.py

After the above code run has completed, to train the data and see face recognition happening simultaneously, run facial_rec.py

>python facial_rec.py
