import tkinter as tk                
from tkinter import font  as tkfont 
from tkinter import *
import os
import sys
from tkinter.ttk import *
from tkinter import filedialog
import shutil as shutil
import cv2
import face_recognition

application_path = os.path.dirname(sys.executable)

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        self.geometry("850x600")
        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, EditPerson, DetectPerson,AddPic,AddPerson,AlreadyExisting,SuccessfulAdd,SuccessfulAddedPic,PictureAdd,AddMultPic):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Willkommen in der Zweig App!", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        label1 = tk.Label(self, text="       ", font=controller.title_font)
        label1.pack()

        button1 = tk.Button(self, text="Personen verwalten",
                            command=lambda: controller.show_frame("EditPerson"))
        button2 = tk.Button(self, text="Unbekannte erkennen",
                            command=lambda: controller.show_frame("DetectPerson"))
        button1.pack()
        button2.pack()


class EditPerson(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Bitte wählen Sie aus:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
            
        button1 = tk.Button(self, text="Porträt zu bestehender Person hinzufügen",
                          command=lambda: controller.show_frame("PictureAdd"))
        button1.pack()

        
        button2 = tk.Button(self, text="Person hinzufügen",
        command=lambda: controller.show_frame("AddPerson"))
        button2.pack()
        button3 = tk.Button(self, text="zurück",
                           command=lambda: controller.show_frame("StartPage"))
        button3.pack()



class DetectPerson(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Bitte wählen Sie aus ob Sie mehrere Bilder untersuchen möchten", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button= tk.Button(self, text="Ordner auswählen",
                           command = myClick
                        )
        button.pack()

        button1 = tk.Button(self, text="zurück",
                           command=lambda: controller.show_frame("StartPage"))
        button1.pack()

class PictureAdd(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        button = tk.Button(self, text="Ein Porträt hinzufügen",
        command=lambda: controller.show_frame("AddPic"))
        button.pack()
        button1 = tk.Button(self, text="Mehrere Bilder hinzufügen",
                           command=lambda: controller.show_frame("AddMultPic"))
        button1.pack()
        button1 = tk.Button(self, text="zurück",
                           command=lambda: controller.show_frame("StartPage"))
        button1.pack()


class AddPic(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Name der Person:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        my_list = os.listdir('known_images')
        chosen=StringVar()
        drop = OptionMenu(self,chosen,*my_list )
        drop.pack()
        def kontr():
            text = chosen.get()
            def fotohinzu():
                print(text)
                src_dir = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=(("jpg files", "*.jpg"),("all files", "*.*")))
                dst_dir = 'known_images/'+text
                shutil.copy(src_dir, dst_dir)
                controller.show_frame("SuccessfulAddedPic")              
            if os.path.isdir('known_images/'+text) and text!="":

                button1 = tk.Button(self, text="Foto hinzufügen",
                        command=fotohinzu)
                button1.pack()   
            else:
                controller.show_frame("NotExisting")
            button.configure(state=DISABLED)
        button=tk.Button(self,text="weiter",command=kontr)
        button.pack()
        button2=tk.Button(self,text="zurück",command=lambda: controller.show_frame("PictureAdd"))
        button2.pack()
        

class AddMultPic(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Name der Person:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        my_list = os.listdir('known_images')
        chosen=StringVar()
        drop = OptionMenu(self,chosen,*my_list )
        drop.pack()
        def kontr():
            text = chosen.get()
            def fotohinzu():
                print(text)
                source_folder = filedialog.askdirectory()
                destination_folder = 'known_images/'+text
                files = os.listdir(source_folder)
                for fname in files:
                    shutil.copy2(os.path.join(source_folder,fname), destination_folder)

                controller.show_frame("SuccessfulAddedPic")              
            if os.path.isdir('known_images/'+text) and text!="":
                button1 = tk.Button(self, text="Foto hinzufügen",
                        command=fotohinzu)
                button1.pack()   
            else:
                pass
            button.configure(state=DISABLED)
            
        button=tk.Button(self,text="weiter",command=kontr)
        button.pack()
        button2=tk.Button(self,text="zurück",command=lambda: controller.show_frame("PictureAdd"))
        button2.pack()
        


        

class AddPerson(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Bitte geben Sie den Namen der Person an, die Sie hinzufügen möchten", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        e = Entry(self,width=50)
        e.pack()
        def add():
            text = e.get()
            if os.path.isdir('known_images/'+text):
                controller.show_frame("AlreadyExisting")
            else:
                os.mkdir('known_images/'+text)
                controller.show_frame("SuccessfulAdd")



        button1 = tk.Button(self, text="hinzufügen",
                        command=add)
        button1.pack()
        button2 = tk.Button(self, text="zurück",
                           command=lambda: controller.show_frame("EditPerson"))
        button2.pack()



class AlreadyExisting(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Person existiert bereits", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button1 = tk.Button(self, text="zurück",
                    command=lambda: controller.show_frame("AddPerson"))
        button1.pack()


class SuccessfulAdd(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Person wurde erfolgreich hinzugefügt", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Foto hinzufügen",
                            command=lambda: controller.show_frame("AddPic"))
        button.pack()
        button1 = tk.Button(self, text="zurück",
                    command=lambda: controller.show_frame("StartPage"))
        button1.pack()

class SuccessfulAddedPic(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Foto wurde hinzugefügt!", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Startseite",
                            command=lambda: controller.show_frame("StartPage"))
        button.pack()
        button1 = tk.Button(self, text="zurück",
                            command=lambda: controller.show_frame("AddPic"))
        button1.pack()
   
#giving different names different colors
def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

def myClick():
    DIR_KNOWN_IMGS = "known_images"
    DIR_UNKNOWN_IMGS = filedialog.askdirectory()

    TOLERANCE = 0.5
    FRAME_THICKNESS = 1
    FONT_THICKNESS = 1
    MODEL = 'hog'

    known_imgs = []
    known_names = []

    name_of_chosen_folder = f'{DIR_UNKNOWN_IMGS}'.rsplit('/', 1)[-1]
    
    #creating recognized image folder if not there already
    if os.path.isdir('recognized_images/'):
        print("already here")
    else:
        os.mkdir('recognized_images/') 

    #creating specific folder for the chosen folder
    if os.path.isdir('recognized_images/'+ f'{name_of_chosen_folder}_recog'):
        print("already here")
    else:
        os.mkdir('recognized_images/'+ f'{name_of_chosen_folder}_recog')      

    print("Bekannte Personen: ")
    #going trough known_images and assign names
    for name in os.listdir(DIR_KNOWN_IMGS):
        for filename in os.listdir(f'{DIR_KNOWN_IMGS}/{name}'):
            image    = face_recognition.load_image_file(f'{DIR_KNOWN_IMGS}/{name}/{filename}')
            encoding = face_recognition.face_encodings(image)[0]

            known_imgs.append(encoding)
            known_names.append(name)

    #create .txt file with recognized persons in chosen folder
    save_path = 'recognized_images/'+ f'{name_of_chosen_folder}_recog'
    file_name = f'{name_of_chosen_folder}_recognized.txt'
    completeName = os.path.join(save_path, file_name)
    file1 = open(completeName, "w")
    file1.close()
    file1 = open(completeName, "a")

    for filename in os.listdir(DIR_UNKNOWN_IMGS):
        image     = face_recognition.load_image_file(f'{DIR_UNKNOWN_IMGS}/{filename}')
        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)
        image     = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #print(f'{len(encodings)} Gesicht(er) gefunden')
        file1.write(f'Im Bild: "{filename}" wurden {len(encodings)} Gesichter und folgende Personen erkannt:'+ '\n')
        #compare found faces in unknown_images with known_images
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_imgs, face_encoding, TOLERANCE)

            match = None
            if True in results:
                match = known_names[results.index(True)]
                print(match)
                #drawing rectangle at facelocation
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                color = name_to_color(match)
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                #writing names at facelocation
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match, (face_location[3] + 5, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), FONT_THICKNESS)
                file1.write(f'{match}'+ '\n')
        file1.write('\n')

        cv2.imwrite(f'recognized_images/{name_of_chosen_folder}_recog/'+ filename + '_recognized.jpg', image)

    file1.close()


app = SampleApp()
app.mainloop()