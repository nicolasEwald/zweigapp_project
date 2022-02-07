# Zweigapp 

Zweigapp is a project of Yasmin Tourk, Natalia Vizintini and Nicolas Ewald. The project was started while we attended the subject "Software Praktikum" at the University of Salzburg in Austria. Our goal was to create a programm which can detect and recognize faces of persons. 

## General Installation Information
face_recognition can be installed via pip, but there are quite a few dependencies, which we'll be running through.

### Windows
Microsoft Visual Studio 2015 or newer (check if build tools are enough): https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/ - download and install.

Download and install CMake for Windows: https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2-win64-x64.msi (other options available at: https://cmake.org/download/) During installation check to add CMake to PATH for all users (leave everything else default).

Restart Windows (simply starting new cmd window so it will include changes in PATH should be enough, restart is a safer option though)

pip install --upgrade numpy scipy

Install git client from: https://git-scm.com/download/win


## Code explanation
    import os
    import cv2
    import face_recognition
    import tkinter as tk                
    from tkinter import font  as tkfont 
    from tkinter import *
    from tkinter.ttk import *
    from tkinter import filedialog
    import shutil as shutil

We are using os for working with directories and cv2 for labeling/drawing on our images.
Furthermore we use tkinter for our UI and to choose folders we want to work with.
At last we use shutil to copy images and save it in other folders.

### class SampleApp(tk.Tk):


### class StartPage(tk.Frame):


### class EditPerson(tk.Frame):


### class DetectPerson(tk.Frame):

    
### class PictureAdd(tk.Frame):


### class AddPic(tk.Frame):
   

### class AddMultPic(tk.Frame):
       

### class NotExisting(tk.Frame):
    

### class AddPerson(tk.Frame):


### class AlreadyExisting(tk.Frame):
   

### class SuccessfulAdd(tk.Frame):
   

### class SuccessfulAddedPic(tk.Frame):
  
   
### def name_to_color(name):
   
This method is the core programm which detects and recognizises the images and encases the faces of people being recognized. 

    def myClick():

    DIR_KNOWN_IMGS = "known_images"
    DIR_UNKNOWN_IMGS = filedialog.askdirectory()

    TOLERANCE = 0.5
    FRAME_THICKNESS = 1
    FONT_THICKNESS = 1
    MODEL = 'hog'

In the Programm we use the above constants. The first two constants are just the names of our known and unknown directories.
Next we have TOLERANCE. This is a value from 0 to 1 that will allow you to tweak the sensitivity of labeling/predictions. The default value here in the face_recognition package is 0.6. The lower the tolerance, the more "strict" the labels will be.

If you're getting a bunch of labels of some identity on a bunch of faces that aren't correct, you may want to lower the TOLERANCE. If you're not getting any labels at all, then you might want to raise the TOLERANCE.

The FRAME_THICKNESS value is how many pixels wide do you want the rectangles that encase a face to be and FONT_THICKNESS is how thick you want the font with the label to be.

For the model we are using the hog (histogram of oriented gradients), but you can also use cnn (convolutional neural network).

    known_imgs = []
    known_names = []

Than we start with a couple of lists. One for the images we know, the other for the names associated with these images. Next thing we do is iterate over our known faces directory, which contains possibly many directories of identities, which then contain one or more images with that person's face. From here, we want to load in this image with the face_recognition package. Continuing along in this same loop, we will encode each of these faces, then store the encodings and the associated identity to our lists:

    for name in os.listdir(DIR_KNOWN_IMGS):
            for filename in os.listdir(f'{DIR_KNOWN_IMGS}/{name}'):
                image    = face_recognition.load_image_file(f'{DIR_KNOWN_IMGS}/{name}/{filename}')
                encoding = face_recognition.face_encodings(image)[0]

                known_imgs.append(encoding)
                known_names.append(name)

In this part we create a .txt file with recognized persons in the chosen folder:

    save_path = 'recognized_images/'+ f'{name_of_chosen_folder}_recog'
    file_name = f'{name_of_chosen_folder}_recognized.txt'
    completeName = os.path.join(save_path, file_name)
    file1 = open(completeName, "w")
    file1.close()
    file1 = open(completeName, "a")

After that we check unknown images for faces, and then try to identify those faces!

    for filename in os.listdir(DIR_UNKNOWN_IMGS):
            image     = face_recognition.load_image_file(f'{DIR_UNKNOWN_IMGS}/{filename}')
            locations = face_recognition.face_locations(image, model=MODEL)
            encodings = face_recognition.face_encodings(image, locations)
            image     = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

Also in this for loop we compare found faces in unknown_images with known_images and draw the rectangles at the recognized face locations.

    for face_encoding, face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(known_imgs, face_encoding, TOLERANCE)

                match = None
                if True in results:
                    match = known_names[results.index(True)]
                    print(match)
                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])
                    color = name_to_color(match)
                    cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                    top_left = (face_location[3], face_location[2])
                    bottom_right = (face_location[1], face_location[2] + 22)
                    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                    cv2.putText(image, match, (face_location[3] + 5, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), FONT_THICKNESS)
                    file1.write(f'{match}'+ '\n')

At the end we save the created image with the marked faces into a new file and close the file:

    cv2.imwrite(f'recognized_images/{name_of_chosen_folder}_recog/'+ filename + '_recognized.jpg', image)

        file1.close()