import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import euclidean

#clases a elegir
CLASES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

# Cargar modelo CNN
model = tf.keras.models.load_model("modelo_retina_final.keras")

# para cargar la base CBIR
db_descriptores = np.load("db_descriptores.npy")
db_etiquetas = np.load("db_etiquetas.npy")
db_rutas = np.load("db_rutas.npy")


# Características GLCM

def extraer_glcm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)

    return np.array([
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
        graycoprops(glcm, "ASM")[0, 0]
    ])


# Predcciones de la CNN

def predecir(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # CNN
    img_cnn = cv2.resize(img_rgb, (224, 224))
    img_cnn = img_cnn / 255.0
    img_cnn = np.expand_dims(img_cnn, axis=0)

    pred = model.predict(img_cnn, verbose=0)[0]

    texto = "Diagnóstico:\n"
    for c, p in zip(CLASES, pred):
        texto += f"{c}: {p*100:.2f}%\n"

    resultado.config(text=texto)

    # CBIR
    query_desc = extraer_glcm(img_rgb)

    distancias = [
        euclidean(query_desc, d) for d in db_descriptores
    ]

    idxs = np.argsort(distancias)[:5]

    mostrar_similares(idxs)


# mostrar imagenes similares

def mostrar_similares(idxs):
    for widget in frame_similares.winfo_children():
        widget.destroy()

    for i, idx in enumerate(idxs):
        try:
            img = Image.open(db_rutas[idx])
            img = img.resize((120, 120))
            img = ImageTk.PhotoImage(img)

            lbl = tk.Label(frame_similares, image=img)
            lbl.image = img
            lbl.grid(row=0, column=i, padx=5)

            tk.Label(
                frame_similares,
                text=db_etiquetas[idx]
            ).grid(row=1, column=i)

        except:
            continue


# se carga la imagen del usuairo

def cargar_imagen():
    path = filedialog.askopenfilename(
        filetypes=[("Imágenes", "*.jpg *.png *.jpeg")]
    )

    if path:
        img = Image.open(path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)

        panel.config(image=img)
        panel.image = img

        predecir(path)


# confihuracion de la interfaz

root = tk.Tk()
root.title("Sistema de Detección de Enfermedades Retinales")
root.geometry("900x700")

btn = tk.Button(root, text="Cargar imagen", command=cargar_imagen)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

resultado = tk.Label(root, text="Carga una imagen", justify="left")
resultado.pack(pady=10)

tk.Label(root, text="Top 5 imágenes similares").pack()

frame_similares = tk.Frame(root)
frame_similares.pack()

root.mainloop()