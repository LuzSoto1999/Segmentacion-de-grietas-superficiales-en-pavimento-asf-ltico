import matplotlib.pyplot as plt
import numpy as np
from skimage import io, morphology
from scipy.ndimage import distance_transform_edt

# Cargar la imagen (asegúrate de que esté en escala de grises)
BW = io.imread(r'predicciones\212_1.png', as_gray=True)

# Asegurarse de que la imagen sea binaria
BW = BW > 0.5

# Obtener el esqueleto
BW_skel = morphology.skeletonize(BW)

# Calcular la distancia del esqueleto a los bordes de las grietas
distancia = distance_transform_edt(~BW_skel)

# Ancho de la grieta en píxeles
ancho_grieta = 2 * distancia

# Convertir de píxeles a milímetros (según tu factor de conversión)
ancho_grieta_mm = ancho_grieta * 0.924

# Crear una imagen de colores para las categorías
predic_pintada = np.zeros((BW.shape[0], BW.shape[1], 3))  # Imagen RGB vacía

# Definir categorías:
# Verde para grietas <= 3 mm, Rojo para grietas > 3 mm
predic_pintada[ancho_grieta_mm <= 3] = [0, 1, 0]  # Verde
predic_pintada[ancho_grieta_mm > 3] = [1, 0, 0]   # Rojo

# Mantener el color blanco donde no hay grietas (en la máscara original)
predic_pintada[BW == 1] = [1, 1, 1]  # Blanco

# Mostrar el resultado
plt.imshow(predic_pintada)
plt.title('Ancho de las Grietas (verde <= 3mm, rojo > 3mm)')
plt.axis('off')  # Ocultar ejes
plt.show()

# Calcular y mostrar el tamaño mínimo de grieta detectada en mm
min_grieta_mm = np.min(ancho_grieta_mm[ancho_grieta_mm > 0])  # Excluyendo ceros
print(f"Tamaño mínimo de grieta detectada: {min_grieta_mm:.2f} mm")
