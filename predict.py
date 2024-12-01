import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model

# Crear carpetas de salida si no existen
ruta_comparaciones = r"comparaciones"
ruta_resultados = r"predicciones"
os.makedirs(ruta_comparaciones, exist_ok=True)
os.makedirs(ruta_resultados, exist_ok=True)

# Función para cargar todas las imágenes de una vez
def cargar_imagenes(ruta_imagenes, size=(256, 256)):
    archivos_imagenes = [f for f in os.listdir(ruta_imagenes) if f.endswith(".jpg")]
    imagenes = [
        np.array(Image.open(os.path.join(ruta_imagenes, archivo)).convert("RGB").resize(size)) / 255.0
        for archivo in archivos_imagenes
    ]
    return np.array(imagenes), archivos_imagenes

# Cargar el modelo entrenado
model = load_model(r'modelos\model500_256_8.h5', compile=False)

# Definir la ruta de las imágenes de prueba
ruta_imagenes_prueba = r"datos\Prueba\imagenes"

# Función para combinar imagen original y predicción en una sola imagen
def combinar_imagenes(imagen_original, prediccion):
    # Convertir la imagen original a formato uint8 y escalarla
    imagen_original_uint8 = (imagen_original * 255).astype(np.uint8)
    
    # Convertir la predicción a formato uint8 (binaria)
    prediccion_uint8 = (prediccion.squeeze() * 255).astype(np.uint8)
    
    # Convertir ambas imágenes a objetos PIL
    imagen_pil = Image.fromarray(imagen_original_uint8)
    prediccion_pil = Image.fromarray(prediccion_uint8)

    # Asegurarse de que ambas imágenes tengan el mismo modo (RGB o L)
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    if prediccion_pil.mode != 'L':  # Predicción en escala de grises
        prediccion_pil = prediccion_pil.convert('L')

    # Crear una nueva imagen que combine ambas
    ancho_total = imagen_pil.width + prediccion_pil.width
    alto_maximo = max(imagen_pil.height, prediccion_pil.height) + 30  # Espacio extra para los títulos
    
    # Crear una imagen vacía donde se combinarán ambas
    imagen_combined = Image.new('RGB', (ancho_total, alto_maximo), color=(0, 0, 0))
    
    # Pegar la imagen original y la predicción en la imagen combinada
    imagen_combined.paste(imagen_pil, (0, 30))  # Añadir 30px de espacio para el título
    imagen_combined.paste(prediccion_pil.convert('RGB'), (imagen_pil.width, 30))

    # Añadir títulos a las imágenes
    draw = ImageDraw.Draw(imagen_combined)
    font = ImageFont.load_default()  # Usar fuente por defecto
    
    # Añadir títulos en la parte superior
    draw.text((10, 10), "Imagen Original", fill="white", font=font)
    draw.text((imagen_pil.width + 10, 10), "Prediccion", fill="white", font=font)
    
    return imagen_combined

# Función para guardar la imagen combinada
def guardar_comparaciones_combinadas(imagen_original, prediccion, nombre_archivo):
    imagen_combined = combinar_imagenes(imagen_original, prediccion)
    imagen_combined_path = os.path.join(ruta_comparaciones, f"{nombre_archivo}_comparacion.png")
    imagen_combined.save(imagen_combined_path)

# Función para guardar solo la predicción
def guardar_resultado(prediccion, nombre_archivo):
    prediccion_pil = Image.fromarray((prediccion.squeeze() * 255).astype(np.uint8))
    resultado_path = os.path.join(ruta_resultados, f"{nombre_archivo}.png")
    prediccion_pil.save(resultado_path)

# Procesar todas las imágenes y guardar comparaciones combinadas y resultados
def procesar_imagenes():
    X_test, nombres_imagenes = cargar_imagenes(ruta_imagenes_prueba)
    predicciones = model.predict(X_test)
    predicciones_binarizadas = (predicciones > 0.5).astype(np.uint8)

    for i in range(len(X_test)):
        nombre_archivo = os.path.splitext(nombres_imagenes[i])[0]
        guardar_comparaciones_combinadas(X_test[i], predicciones_binarizadas[i], nombre_archivo)
        guardar_resultado(predicciones_binarizadas[i], nombre_archivo)  # Guardar solo la predicción

    print(f'Todas las imágenes han sido procesadas y guardadas en la carpeta de comparaciones y resultados.')

# Procesar y guardar las predicciones
procesar_imagenes()