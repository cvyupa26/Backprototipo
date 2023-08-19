# -*- coding: utf-8 -*-
"""
Datalle: Pruebas de clasificación con CNN MiniVGG
@author: cvyupasilva
"""
import cv2
import numpy as np
import os
from imutils import paths
from tensorflow.keras.utils import img_to_array

dir_imagenes = "dataset/test/"
print(os.listdir("dataset/test/"))
imagePaths = np.array(list(paths.list_images(dir_imagenes)))
# Coger aleatoriamente 10 imagenes para cargar en el disco
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# Todas las imagenes tendran un tamaño de 32x32
SIZE = 32
data = []

# Cargar imagenes
for (i, imagePath) in enumerate(imagePaths):
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Redimensionar las imágenes para un procesamiento más rápido:
    img = cv2.resize(img, (SIZE, SIZE), interpolation = cv2.INTER_AREA)
    #Convertir a array la imagen con el pre-procesador de keras
    img = img_to_array(img)
    # Agregar imagenes a la lista
    data.append(img)

data = np.array(data)
# Normalizar imagenes escalando pixeles en (0-1) dividir entre 255
data = data = data.astype("float") / 255.0

# Cargar modelo entrenado:
dir_model = "logs/modelo3.h5"
from keras.models import load_model
model = load_model(dir_model)

#predicciones en las imagenes
preds = model.predict(data, batch_size=32).argmax(axis=1)

classLabels = ["psoriasis", "sinenfermedad", "vitiligo"]
for(i, imagePath) in enumerate(imagePaths):
	# cargar la imagen de ejemplo, dibujar la predicción,y la desplegarla en la pantalla
	image = cv2.imread(imagePath)
	cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)







