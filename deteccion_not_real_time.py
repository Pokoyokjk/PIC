import cv2
import numpy as np
import time
import os


#  DEFINCIÓN DE FUNCIONES
# _______________________________________________________

def cargar_n_reales(ruta_reales):
    reales = {}
    with open(ruta_reales, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if ":" not in line:
                print(f"⚠ Línea inválida ignorada: {line}")
                continue

            try:
                nombre, valores = line.split(":")

                # quitar paréntesis y dividir todos los valores
                numeros = valores.strip("()").split(",")

                # coger SOLO el primer número como huecos reales
                vacias = int(numeros[0])

                reales[nombre.strip()] = vacias

            except Exception as e:
                print(f"⚠ Error procesando línea: {line}\n{e}")
                continue

    return reales

ruta_reales = "label_depurado2.txt"
reales = cargar_n_reales(ruta_reales)

# -----------------------------------------------------
def find_blister_contour(img):
    """
    Encuentra el contorno externo más grande del blister.
    Devuelve el contorno con mayor área.
    """
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

# -----------------------------------------------------
def recortar_blister(img, contour):
    """
    Recorta la región del blister usando su bounding box.
    """
    x, y, w, h = cv2.boundingRect(contour)
    blister = img[y:y+h, x:x+w]
    return blister

# -----------------------------------------------------
def nivel_oscuridad(img, centro, radio):
    """
    Calcula el nivel de oscuridad promedio dentro de una cavidad circular.
    El valor devuelto se basa en el canal Y (luminancia) ecualizado.
    """
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

# -----------------------------------------------------
def detectar_cavidades_contornos(blister, min_area=2500, circularity_thresh=(0.8, 1)):
    """
    Detecta cavidades circulares dentro del blister.
    Devuelve:
       - Imagen con cavidades marcadas
       - Número de cavidades válidas
       - Lista de niveles de oscuridad
       - Lista de cavidades detectadas correctamente
    """
    # Preprocesamiento: gris + suavizado
    gray = cv2.cvtColor(blister, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    # Contraste adaptativo CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_blister_clahe = clahe.apply(blur)
    # Bordes y morfología
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
    # Filtrar contornos circulares (posibles cavidades)
    for cnt in contours:
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4*np.pi*(area/(perimeter**2))
        # Condición de circularidad + tamaño
        if circularity_thresh[0] <= circularity <= circularity_thresh[1] and area >= min_area:
            # Aproximar centro y radio
            (x,y), radius = cv2.minEnclosingCircle(cnt)
            oscuridad = nivel_oscuridad(blister, (int(x), int(y)), int(radius))
            niveles.append(oscuridad)
            cavidades_props.append({'centro': (int(x), int(y)), 'radio': int(radius), 'area': area})
    # Promedio global
    level=np.mean(niveles)
    print(f'promedio: {level}')
    for cav,osc in zip(cavidades_props,niveles):
        if osc > level*0.91:
            cav_blister.append(cav)
            # Dibujar contornos
            cv2.circle(img_contours, cav['centro'], cav['radio'], (0,255,0), 3)

    num_blisters = len(cav_blister)
    # print(f"Número de blisters: {num_blisters}")
    print (np.uint8(niveles))
    return img_contours, num_blisters, niveles,cav_blister



# PROCESAMIENTO DE TODAS LAS IMÁGENES
# _______________________________________________________

image_folder = "img_depurado"

procesados = 0
correctos = 0
tiempo_total = 0

print("Procesando imágenes...\n")

for filename in os.listdir(image_folder):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    procesados += 1
    path = os.path.join(image_folder, filename)
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: no se pudo cargar {filename}")
        continue

    # Tiempo inicial
    t0 = time.perf_counter()

    try:
        blister_contour = find_blister_contour(img)
        blister = recortar_blister(img, blister_contour)
        _, n_detectadas, niveles, _ = detectar_cavidades_contornos(blister)
    except:
        print(f"{filename}: ERROR procesando")
        continue

    # Tiempo final
    t1 = time.perf_counter()
    tiempo_img = (t1 - t0) * 1000
    tiempo_total += tiempo_img

    # Cavidades reales
    if filename not in reales:
        print(f"{filename}: ⚠ No hay dato real en label_depurado2.txt")
        continue

    n_real = reales[filename]

    # Comprobar acierto
    total_cavidades = 10  # o pon el número correcto si cambia

    pastillas_reales = total_cavidades - n_real  # n_real = huecos reales

    es_correcto = (n_detectadas == pastillas_reales)
    if es_correcto:
        correctos += 1

    print(f"{filename}: detectadas={n_detectadas}, reales={pastillas_reales}, "
          f"correcto={es_correcto}, tiempo={tiempo_img:.1f} ms")


# RESULTADOS FINALES
    # _______________________________________________________

print("\n----------------------------------------------")
print(f"Imágenes procesadas: {procesados}")
print(f"Aciertos: {correctos}")
print(f"Precisión: {(correctos/procesados)*100:.2f}%")
print(f"Tiempo total: {tiempo_total/1000:.2f} s")
print(f"Tiempo promedio por imagen: {tiempo_total/procesados:.1f} ms")
print("----------------------------------------------")
