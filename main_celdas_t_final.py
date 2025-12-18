import cv2 #manipular imagenes
import numpy as np #operaciones numericas
import os
import time


# Definir contorno del blister
# __________________________________________________
def find_blister_contour(img):
    # --- Convertir a escala de grises ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Suavizar ---
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # --- Bordes ---
    edges = cv2.Canny(blur, 50, 150)

    # --- Cerrar huecos para que el contorno sea más sólido ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # --- Buscar contornos ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Seleccionar el contorno más grande ---
    blister_contour = max(contours, key=cv2.contourArea)

    return blister_contour

 

# Dividir el blister en celdas
# __________________________________________________
def dividir_en_celdas(blister, filas, columnas):
    h, w = blister.shape[:2] #alto y ancho
    celda_h = h // filas #alto de cada celda
    celda_w = w // columnas #ancho de cada celda
    celdas = []
    for i in range(filas):
        for j in range(columnas):
            y1 = i * celda_h #inicio y de la celda
            y2 = (i + 1) * celda_h #fin y de la celda
            x1 = j * celda_w #inicio x de la celda
            x2 = (j + 1) * celda_w #fin x de la celda
            celda = blister[y1:y2, x1:x2] #recortar la celda
            celdas.append(celda)
    return celdas, celda_h, celda_w


# Cargar valores reales desde archivo
# __________________________________________________
def cargar_n_reales(ruta_reales):
    reales = {}
    with open(ruta_reales, "r") as f:
        for line in f:
            line = line.strip()

            # Ignorar líneas vacías o sin formato correcto
            if not line or ":" not in line:
                continue

            try:
                nombre, valores = line.split(":")
                valores = valores.strip().replace("(", "").replace(")", "")
                partes = valores.split(",")

                # Asegurar que hay exactamente 3 números
                if len(partes) != 3:
                    continue

                # Limpiar y convertir
                vacias = int(partes[0].strip())
                incompletas = int(partes[1].strip())
                rotas = int(partes[2].strip())

                reales[nombre.strip()] = vacias + incompletas + rotas

            except Exception as e:
                # Si hay una línea mal, se ignora sin frenar el programa
                print(f"⚠ Línea ignorada por error: {line}")
                continue

    return reales


# Función para determinar si una celda está llena o vacía
# __________________________________________________
def es_celda_llena(celda):
    gray = cv2.cvtColor(celda, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    mean_val = np.mean(blur)

    # Puedes ajustar 80 si hace falta (70-120 según iluminación)
    return mean_val > 80, mean_val


# Programa principal
# __________________________________________________
ruta_reales = "label_depurado2.txt"
image_folder = "img_depurado/"

filas, columnas = 2, 5
total_cavidades = filas * columnas

reales = cargar_n_reales(ruta_reales)


procesados = 0
correctos = 0
tiempo_procesado_total = 0

for filename in os.listdir(image_folder):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    procesados += 1

    # --- Cargar imagen (NO se mide tiempo) ---
    img = cv2.imread(os.path.join(image_folder, filename))

    # --- Medir SOLO el procesado ---
    t0 = time.time()


    # Encontrar contorno del blister
    blister_contour = find_blister_contour(img)
    x, y, w, h = cv2.boundingRect(blister_contour)
    blister = img[y:y+h, x:x+w]

    # Dividir en celdas
    celdas, _, _ = dividir_en_celdas(blister, filas, columnas)

    # Detectar vacías
    cav_detectadas = sum(1 for celda in celdas if not es_celda_llena(celda))

    t1 = time.time()

    # Tiempo SOLO del procesado
    tiempo_img = t1 - t0
    tiempo_procesado_total += tiempo_img

    vacias_reales = reales.get(filename, None)

    if vacias_reales is not None and vacias_reales == cav_detectadas:
        correctos += 1


# -------------------------------------------------------
# 6. RESULTADOS
# -------------------------------------------------------
precision = (correctos / procesados) * 100
tiempo_promedio = tiempo_procesado_total / procesados

print("\n------------------------------------------")
print(f"Precisión: {precision:.2f}%")
print(f"Tiempo total de procesado puro: {tiempo_procesado_total:.4f} s")
print(f"Tiempo promedio por imagen (solo procesado): {tiempo_promedio:.4f} s")
print("\n------------------------------------------")