import customtkinter
from matplotlib import pyplot as plt
from logging import root
from tkinter import filedialog
from tkinter.tix import IMAGE
import tensorflow as tf
from keras.models import load_model
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from numpy import asarray
from numpy import expand_dims
from numpy import squeeze
from matplotlib import pyplot
from numpy import ndarray
from matplotlib import pyplot as plt
from tkinter import ttk
import numpy as np
import time
from io import BytesIO
root_tk = customtkinter.CTk()
root_tk.geometry(f"{600}x{500}")
root_tk.title("Satellite2Map")
photo = Image.open('Desktop\samples\icon-png.png')
img = ImageTk.PhotoImage(photo)
root_tk.iconphoto(False, img)
customtkinter.set_appearance_mode("dark")

label = customtkinter.CTkLabel(master=root_tk,
                               text="Satellite2Map",
                               width=120,
                               height=25,
                               corner_radius=8,
                               text_color='white',
                               text_font=("bold", "38"))
label.place(relx=0.35, rely=0.1)



def translate():
    frame1 = customtkinter.CTkFrame(master=root_tk,
                                fg_color='#cccccc',
                                width=200,
                                height=200,
                                corner_radius=10)

    frame1.place(relx=0.2, rely=0.6, anchor='center')
    frame2 = customtkinter.CTkFrame(master=root_tk,
                                fg_color='#cccccc',
                                width=200,
                                height=200,
                                corner_radius=10)

    frame2.place(relx=0.8, rely=0.6, anchor='center')

    path = filedialog.askopenfilename(title="Select an Image", filetype=(
        ('image    files', '*.jpg'), ('all files', '*.*')))
    raw_image = Image.open(path)
    # print the source image

    resized_image1 = raw_image.resize((350, 350))
    src_img = ImageTk.PhotoImage(resized_image1)
    label_img1 = Label(frame1, image=src_img)
    label_img1.pack()

    # feed to the model
    image = raw_image.resize((256, 256))
    # load the model
    model = load_model('Desktop\generator_model.h5')  # put it outside this mtd
    # transform the image into numpy array
    numpydata = asarray(image)
    # normalize the image
    numpydata = (numpydata - 127.5) / 127.5
    numpydata = expand_dims(numpydata, 0)
    # generate an image
    gen_data = model.predict(numpydata)
    # resize
    squeezed = squeeze(gen_data, axis=None)
    # upscaling
    up_scaled = (squeezed * 127.5) + 127.5
    rounded = np.rint(up_scaled)
    int_array = rounded.astype(int)
    # plot images

    plt.imshow(int_array, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    reloadedPILImage = Image.open(buffer)

    left = 144
    top = 53
    right = 513
    bottom = 421
    img_res = reloadedPILImage.crop((left, top, right, bottom))

    resized_image2 = img_res.resize((350, 350))
    img = ImageTk.PhotoImage(resized_image2)
    label_img2 = Label(frame2, image=img)
    label_img2.pack()


button1 = customtkinter.CTkButton(master=root_tk,

                                  text="Translate",

                                  command=translate)
button1.place(relx=0.5, rely=0.9, anchor='center')

root_tk.mainloop()
