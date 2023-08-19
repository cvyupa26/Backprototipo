# -*- coding: utf-8 -*-
"""
Datalle: Implementación de la CNN MiniVGG
@author: cvyupa
"""
from minivgg_model import MiniVGGNet
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import img_to_array

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
#from tensorflow.keras.models import Sequential, save_model

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob

# Definir ruta del dataset
dataset_dir = "dataset/enfermedad/*"
print(os.listdir("dataset/enfermedad/"))
# Todas las imagenes tendran un tamaño de 32x32
SIZE = 32
data = []
labels = [] 
# Cargar dataset
verbose = 250
i = 0
number_images = 1541
for directory_path in glob.glob(dataset_dir):
    #capturar label/etiqueta
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):

        #leer imagen:
        img = cv2.imread(img_path)
        
        #pasar de BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (SIZE, SIZE), interpolation = cv2.INTER_AREA)

        img = img_to_array(img)
        data.append(img)
        labels.append(label)
        
        # Mostrar alertas/logs
        if verbose > 0 and i > 0 and (i+1)%verbose == 0:
            print("[INFO] Procesado {}/{}".format(i+1, number_images))
        i +=1

data = np.array(data)
labels =  np.array(labels)


data = data = data.astype("float") / 255.0


(x_train, x_test, y_train, y_test) =  train_test_split(data,
                                                       labels,
                                                       test_size=0.25,
                                                       random_state=42)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# Definir el optimizador del modelo:
from keras.optimizers import SGD
epocas = 20
opt = SGD(lr=0.01, decay=0.01 / epocas, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Definir callback para guardar solo el mejor modelo con menor perdida
carpeta_logs = "logs/modelo3.h5"
checkpoint = ModelCheckpoint(carpeta_logs, monitor="val_loss", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# Entrenar Red:
H = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
			  batch_size=64,
			  epochs=epocas,
			  callbacks=callbacks,
			  verbose=2)

# Evaluar la red
# Nota: Los labels estan en orden alfabético:
print(os.listdir("dataset/enfermedad"))
labelNames = ["psoriasis", "sinenfermedad", "vitiligo"] 

predictions = model.predict(x_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames))

#Guardar nuestra perdida y precision en la figura
plot_dir = "logs/graficoenf.png"
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epocas), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epocas), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epocas), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epocas), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Piel")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(plot_dir)


# Cargar el model entrenado y lo guardar el modelo en formato tflite:
keras_model = tf.keras.models.load_model('logs/modelo3.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tfmodel = converter.convert()
file = open('enfermedad.tflite','wb') 
file.write(tfmodel)







