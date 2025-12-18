import cv2 #manipular imagenes
import numpy as np #operaciones numericas
import matplotlib.pyplot as plt #visualizar imagenes
from IPython.display import Image #mostrar imagenes en jupyter
import os #Para operaciones del sistema operativo
import random #Para operaciones aleatorias
import time 


# Definicion de funciones
# __________________________________________________
def find_blister_contour(img):
    # --- Convertir a escala de grises ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Suavizar ---
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # --- Bordes ---
    edges = cv2.Canny(blur, 50, 150)

    # --- Cerrar huecos para que el contorno sea más sólido ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)) #
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # --- Buscar contornos ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Seleccionar el contorno más grande ---
    blister_contour = max(contours, key=cv2.contourArea)

    return blister_contour


def recortar_blister(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    blister = img[y:y+h, x:x+w]
    return blister


def nivel_oscuridad(img, centro, radio):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    # Ecualizar el canal Y
    y_eq = cv2.equalizeHist(y)

    # Unir de nuevo
    img_eq = cv2.merge([y_eq, cr, cb])
    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YCrCb2BGR)

    # Crear máscara circular
    mask = np.zeros(img_ycrcb.shape, dtype=np.uint8)
    cv2.circle(mask, centro, radio, 255, -1)  # círculo lleno

    # Extraer solo la región circular
    valores = img_ycrcb[mask == 255]

    # Calcular nivel de oscuridad
    # 0 = negro, 255 = blanco
    promedio = valores.mean()

    return promedio


def detectar_cavidades_contornos(blister, min_area=2500, circularity_thresh=(0.8, 1)):
    gray = cv2.cvtColor(blister, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    # Contraste adaptativo CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_blister_clahe = clahe.apply(blur)
    edges = cv2.Canny(gray_blister_clahe, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # plt.figure()
    # plt.title("Morfología de Cierre")
    # plt.imshow(closed, cmap='gray')
    # plt.axis('off')  # Ocultar los ejes
    # plt.show()
    # Encontrar contornos
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = blister.copy()
    cavidades_props = []
    niveles = []
    cav_blister = []
    for cnt in contours:
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4*np.pi*(area/(perimeter**2))
        if circularity_thresh[0] <= circularity <= circularity_thresh[1] and area >= min_area:
            # Aproximar centro y radio
            (x,y), radius = cv2.minEnclosingCircle(cnt)
            oscuridad = nivel_oscuridad(blister, (int(x), int(y)), int(radius))
            niveles.append(oscuridad)
            cavidades_props.append({'centro': (int(x), int(y)), 'radio': int(radius), 'area': area})

    level=np.mean(niveles)
    for cav,osc in zip(cavidades_props,niveles):
        if osc > level*0.91:
            cav_blister.append(cav)
            # Dibujar contornos
            cv2.circle(img_contours, cav['centro'], cav['radio'], (0,0,255), 3)

    num_blisters = len(cav_blister)

    return img_contours, num_blisters, niveles,cav_blister


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



# Programa principal
# __________________________________________________
ruta_reales = "label_depurado2.txt"
reales = cargar_n_reales(ruta_reales)

image_folder = "img_depurado/"
total_cavidades = 10

procesados = 0
correctos = 0

inicio = time.time()  # medir tiempo

tiempo_procesado_total = 0

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):

        procesados += 1


        # No medir tiempo de carga
        img = cv2.imread(os.path.join(image_folder, filename))
        # Medir solo el procesado
        t0 = time.time()

        blister_contour = find_blister_contour(img)
        blister = recortar_blister(img, blister_contour)
        _, cav_detectadas, _, _ = detectar_cavidades_contornos(blister, min_area=1000)

        t1 = time.time()

        tiempo_img = t1 - t0
        tiempo_procesado_total += tiempo_img

        print(f"{filename}: Procesado puro = {tiempo_img:.3f} s")

        vacias_detectadas = total_cavidades - cav_detectadas
        vacias_reales = reales.get(filename, None)

        if vacias_reales is not None and vacias_reales == vacias_detectadas:
            correctos += 1


# TIEMPOS FINALES
# __________________________________________________________
precision = (correctos / procesados) * 100
tiempo_promedio = tiempo_procesado_total / procesados

print("\n------------------------------------------")
print(f"TIEMPO TOTAL DE PROCESADO PURO: {tiempo_procesado_total:.3f} segundos")
print(f"TIEMPO PROMEDIO POR IMAGEN: {tiempo_promedio:.4f} segundos")