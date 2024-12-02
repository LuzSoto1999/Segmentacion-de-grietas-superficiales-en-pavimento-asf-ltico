# Segmentacion-de-grietas-superficiales-en-pavimento-asf-ltico
En esta carpeta encontraras los archivos relacionados al proyecto final de grado titulado "Segmentación de grietas superficiales en pavimento asfaltico utilizando técnicas de visión artificial" incluyendo las base de datos, el modelo entrenado, los algoritmos de entrenamiento, de predicción, de comparación y de medición de grietas.
# El archivo "datos.zip"
Contiene los datos utilizados tanto para el entrenamiento del modelo así como los datos utilizados para la prueba
# train_unet.py
Contiene la arquitectura utilizada para el entrenamiento y llama a las imagenes en la carpeta de datos
# predict.py
Utiliza las imágenes de prueba para verificar el funcionamiento del programa
# model500_256_8.h5
Es el modelo ya entrenado con la base de datos proporcionada, se puede utilizar directamente con las imagenes de prueba sin necesidad de ejecutar train_unet.py
Observación: El programa solo lee archivos .jpg
# Medir.py 
Categoriza las grietas pintando aquellas menores o iguales a 3mm de color verde y aquellas mayores a 3mm de color rojo. Esto se hace para identificar aquellas que presentan un mayor riesgo para el pavimento de aquellas que no. Funciona llamando a una imagen particular por vez.
