import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import Recall
import tensorflow.keras.backend as K
import os
import numpy as np

# Configurar TensorFlow para usar solo la GPU 0
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Establecer el límite de memoria a 4 GB
    except RuntimeError as e:
        print(e)


# Verificar la configuración
print("GPUs visibles:", tf.config.get_visible_devices('GPU'))

# Verificar si se detecta alguna GPU
gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) == 0:
    print("No se detectó ninguna GPU. TensorFlow está utilizando la CPU.")
else:
    print("Se detectó la siguiente GPU:")
    for device in gpu_devices:
        print(device)

from tensorflow.keras.layers import Dropout
def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
        # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c1 = Dropout(0.3)(c1)  # Añadir Dropout
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    c2 = Dropout(0.3)(c2)  # Añadir Dropout
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    c3 = Dropout(0.4)(c3)  # Añadir Dropout
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    c4 = Dropout(0.4)(c4)  # Añadir Dropout
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = Dropout(0.5)(c5)  # Añadir Dropout en el Bottleneck

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4], axis=-1)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    c6 = Dropout(0.4)(c6)  # Añadir Dropout

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3], axis=-1)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    c7 = Dropout(0.4)(c7)  # Añadir Dropout

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2], axis=-1)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    c8 = Dropout(0.3)(c8)  # Añadir Dropout

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1], axis=-1)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    c9 = Dropout(0.3)(c9)  # Añadir Dropout

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def cargar_imagenes_y_mascaras(ruta_imagenes, ruta_mascaras, tamaño=(256, 256)):
    imagenes = []
    mascaras = []
    
    # Procesar las imágenes y máscaras
    archivos_imagenes = [archivo for archivo in os.listdir(ruta_imagenes) if archivo.lower().endswith(".jpg")]
    archivos_mascaras = {archivo.lower(): archivo for archivo in os.listdir(ruta_mascaras) if archivo.lower().endswith(".jpg")}
    
    for archivo in archivos_imagenes:
        nombre_mascara = archivos_mascaras.get(archivo.lower())
        
        if not nombre_mascara:
            print(f"No se encontró la máscara correspondiente para {archivo}. Se omite este archivo.")
            continue
        
        # Cargar y redimensionar imagen
        imagen = Image.open(os.path.join(ruta_imagenes, archivo)).convert("RGB")
        imagen = imagen.resize(tamaño)
        imagen_array = np.array(imagen, dtype=np.float32) / 255.0  # Convertir a float32
        imagenes.append(imagen_array)
        
        # Cargar y redimensionar máscara
        mascara = Image.open(os.path.join(ruta_mascaras, nombre_mascara)).convert("L")
        mascara = mascara.resize(tamaño)
        mascara_array = np.array(mascara, dtype=np.float32) / 255.0
        mascara_array = (mascara_array > 0.5).astype(np.float32)
        mascaras.append(mascara_array)
    
    imagenes = np.array(imagenes, dtype=np.float32)
    mascaras = np.expand_dims(np.array(mascaras, dtype=np.float32), axis=-1)
    
    print(f"Imágenes cargadas: {len(imagenes)}, Máscaras cargadas: {len(mascaras)}")
    return imagenes, mascaras

# Probar la función
ruta_imagenes_entrenamiento = r"datos\Entrenamiento\imagenes_procesadas_3"
ruta_mascaras_entrenamiento = r"datos\Entrenamiento\mascaras_procesadas_3"

# Cargar imágenes y máscaras de entrenamiento
X_train_full, y_train_full = cargar_imagenes_y_mascaras(ruta_imagenes_entrenamiento, ruta_mascaras_entrenamiento)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

def iou(y_true, y_pred):
    y_pred = K.cast(y_pred > 0.5, dtype='float32')  # Binarización de predicciones
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = K.mean((intersection + K.epsilon()) / (union + K.epsilon()), axis=0)
    return iou

def f1_m(y_true, y_pred):
    y_pred = K.cast(y_pred > 0.5, dtype='float32')
    
    # Calcular precisión
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    
    # Calcular recall
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    
    # Calcular F1
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

# Crear y compilar el modelo UNet
model = unet()

# Compilar el modelo incluyendo IoU y F1-Score como métricas
model.compile(optimizer=Adam(), 
              loss=BinaryCrossentropy(), 
              metrics=[BinaryAccuracy(), Recall(), iou, f1_m])

# Entrenar el modelo con early stopping
history = model.fit(
    X_train, y_train, 
    epochs=500, 
    batch_size=8, 
    validation_data=(X_val, y_val)
)

# Guardar el modelo entrenado
model.save('./modelos/model500_256_8.h5')

# Crear la carpeta de gráficos si no existe
ruta_graficos = r'./gráficos'
os.makedirs(ruta_graficos, exist_ok=True)

# Evaluar el modelo
loss, accuracy, recall_metric, iou_metric, f1_metric = model.evaluate(X_val, y_val)

print(f"Pérdida: {loss}")
print(f"Precisión: {accuracy}")
print(f"Recall: {recall_metric}")
print(f"IoU: {iou_metric}")
print(f"F1-Score: {f1_metric}")

def guardar_graficos(history):
    # Gráfico de pérdida (loss)
    plt.figure()
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title('Pérdida vs Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig(os.path.join(ruta_graficos, 'perdida_vs_epocas.png'))  
    plt.close()

    # Gráfico de precisión (binary_accuracy)
    if 'binary_accuracy' in history.history and 'val_binary_accuracy' in history.history:
        plt.figure()
        plt.plot(history.history['binary_accuracy'], label='Precisión de entrenamiento')
        plt.plot(history.history['val_binary_accuracy'], label='Precisión de validación')
        plt.title('Precisión vs Épocas')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
        plt.savefig(os.path.join(ruta_graficos, 'precision_vs_epocas.png'))
        plt.close()

    # Gráfico de IoU
    if 'iou' in history.history and 'val_iou' in history.history:
        plt.figure()
        plt.plot(history.history['iou'], label='IoU de entrenamiento')
        plt.plot(history.history['val_iou'], label='IoU de validación')
        plt.title('IoU vs Épocas')
        plt.xlabel('Épocas')
        plt.ylabel('IoU')
        plt.legend()
        plt.savefig(os.path.join(ruta_graficos, 'iou_vs_epocas.png'))
        plt.close()

    # Gráfico de F1-Score
    if 'f1_m' in history.history and 'val_f1_m' in history.history:
        plt.figure()
        plt.plot(history.history['f1_m'], label='F1-Score de entrenamiento')
        plt.plot(history.history['val_f1_m'], label='F1-Score de validación')
        plt.title('F1-Score vs Épocas')
        plt.xlabel('Épocas')
        plt.ylabel('F1-Score')
        plt.legend()
        plt.savefig(os.path.join(ruta_graficos, 'f1_score_vs_epocas.png'))
        plt.close()


guardar_graficos(history)