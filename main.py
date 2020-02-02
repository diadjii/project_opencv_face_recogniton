from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk,Image
from functions import find_person


def openFile():
    name = askopenfilename(initialdir=".",
        filetypes =(("Text File", "*.jpg"),("All Files","*.*")),
        title = "Choisir une Image.")

    label = Label(fenetre, text="En cours de traitement.....")

    label.place(x=20,y=150)

    find_person(name)

#creation d'une fenetre
fenetre = Tk()

title = Label(fenetre, text="Programme de Reconnaissance Faciale ESP",font=("Helvetica", 16))
title.place(x=20,y=50)

btn = Button(fenetre,text="Choisir une photo", command=openFile)
btn.place(x=200, y=120)

fenetre.title("Reconnaissance Faciale")
fenetre.geometry("500x300")
fenetre.resizable(False, False)
fenetre.mainloop()
