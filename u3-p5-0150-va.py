## Zavala Medina Alexia Paola   Fecha 20/11/2026

import os
import cv2

# Crear carpeta para guardar los resultados
os.makedirs("resultados", exist_ok=True)

# 1. Leer la imagen original
ruta_imagen = "Dark.jpg"
imaDark = cv2.imread(ruta_imagen)

if imaDark is None:
  raise FileNotFoundError(
      f"No se encontró '{ruta_imagen}'. Verifica el nombre y la ruta."
  )

# 2. Convertir a escala de grises
gray_image = cv2.cvtColor(imaDark, cv2.COLOR_BGR2GRAY)
cv2.imwrite("resultados/1_escala_grises.jpg", gray_image)

# 3. Redimensionar (300x200 píxeles)
resized_image = cv2.resize(imaDark, (300, 200))
cv2.imwrite("resultados/2_redimensionada.jpg", resized_image)

# 4. Rotar 90 grados
(h, w) = imaDark.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_image = cv2.warpAffine(imaDark, matrix, (w, h))
cv2.imwrite("resultados/3_rotada_90.jpg", rotated_image)

# 5. Desenfoque Gaussiano
blurred_image = cv2.GaussianBlur(imaDark, (15, 15), 0)
cv2.imwrite("resultados/4_desenfoque.jpg", blurred_image)

# 6. Detección de bordes (Canny)
edges = cv2.Canny(gray_image, 100, 200)
cv2.imwrite("resultados/5_bordes_canny.jpg", edges)

# 7. Dibujar rectángulo y texto (sobre una copia para no alterar la original)
annotated_image = imaDark.copy()
cv2.rectangle(annotated_image, (50, 50), (200, 200), (255, 0, 0), 3)
cv2.putText(
    annotated_image,
    "OpenCV",
    (60, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)
cv2.imwrite("resultados/6_anotada.jpg", annotated_image)

# 8. Umbralización binaria (Thresholding)
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("resultados/7_umbralizada.jpg", thresh_image)

# 9. Detección y dibujo de contornos
contours_image = imaDark.copy()
contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
cv2.imwrite("resultados/8_contornos.jpg", contours_image)

print("Procesamiento completado. Revisa la carpeta 'resultados/'.")
print("Programa ejecutado por Zavala Medina Alexia Paola")