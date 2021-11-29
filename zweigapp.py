import cv2
import face_recognition
import os
from tkinter import *
from tkinter import filedialog

DIR_KNOWN_IMGS = "known_images"
print("choose folder with unknown pictures:")
DIR_UNKNOWN_IMGS = filedialog.askdirectory()

TOLERANCE = 0.5
FRAME_THICKNESS = 1
FONT_THICKNESS = 1
MODEL = 'hog'

known_imgs = []
known_names = []

print("known Image names: ")
#going trough known_images and assign names
for name in os.listdir(DIR_KNOWN_IMGS):
    for filename in os.listdir(f'{DIR_KNOWN_IMGS}/{name}'):
        image    = face_recognition.load_image_file(f'{DIR_KNOWN_IMGS}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]

        known_imgs.append(encoding)
        known_names.append(name)

for filename in os.listdir(DIR_UNKNOWN_IMGS):
    image     = face_recognition.load_image_file(f'{DIR_UNKNOWN_IMGS}/{filename}')
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image     = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f', found {len(encodings)} face(s)')

    #compare found faces in unknown_images with known_images
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_imgs, face_encoding, TOLERANCE)

        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')
                
            #drawing rectangle at facelocation
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [200, 0, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)