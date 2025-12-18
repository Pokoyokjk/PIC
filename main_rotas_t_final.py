# Importación de librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


# Función para mostrar imágenes
# __________________________________________________
def imshow(title, img, size=6):
    plt.figure(figsize=(size, size))
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# Detección de bordes / blister
# __________________________________________________
#Detección de círculos y evaluación
def find_edges(img):
    # --- Convertir a escala de grises ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Suavizar ---
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- Bordes ---
    edges = cv2.Canny(blur, 50, 150)
    #imshow("Bordes (Canny) con suavizado", edges)

    # --- Cerrar huecos para que el contorno sea más sólido ---
    kernel = np.ones((5, 5), np.uint8)
    edges_dil_blur = cv2.dilate(edges, kernel, iterations=2)
    #imshow("Bordes dilatados", edges_dil_blur)
    # --- Buscar contornos ---
    contours, _ = cv2.findContours(edges_dil_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Seleccionar el contorno más grande ---
    blister_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(blister_contour)
    
    # Máscara del blister
    blister_mask_full = np.zeros_like(gray)
    cv2.drawContours(blister_mask_full, [blister_contour], -1, 255, -1)
    #imshow("Máscara del blister (en la imagen completa)", blister_mask_full)
    # Recorte del blister
    blister_crop = img[y:y+h, x:x+w]
    gray_blister = blur[y:y+h, x:x+w]

    # Contraste adaptativo CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_blister_clahe = clahe.apply(gray_blister)

    altura_ref = 200  # píxeles, ejemplo
    h, w = gray_blister.shape[:2]
    factor = altura_ref / h
    gray_blister_clahe = cv2.resize(gray_blister, (int(w*factor), altura_ref))
    blister_crop = cv2.resize(blister_crop, (int(w*factor), altura_ref))

    return gray_blister_clahe, blister_crop


# Clasificación (vacía / completa)
# _____________________________________________________________
def clasificar(gray_blister_clahe, circles):
    umbral_frac_claro = 0.22  # fracción mínima de píxeles claros para considerar pastilla completa
    umbral_intensidad_claro = 150  # intensidad a partir de la cual consideramos "claro"

    resultados_2 = []

    for i, (x_c, y_c, r_c) in enumerate(circles):
        mask = np.zeros_like(gray_blister_clahe)
        cv2.circle(mask, (x_c, y_c), r_c - 3, 255, -1)
        
        region = gray_blister_clahe[mask == 255]


        # ---- Histograma ----
        hist, bins = np.histogram(region, bins=256, range=(0,255))
    
        # Calcular fracción de píxeles "claros"
        pix_claros = np.sum(hist[umbral_intensidad_claro:])
        fraccion_claros = pix_claros / region.size

        # ---- Regla ----
        if fraccion_claros > umbral_frac_claro:
            estado = "completa"
        #elif fraccion_claros > 0.2:
            #estado = "pastilla parcial / rota"
        else:
            estado = "vacía"
            

        resultados_2.append({
            "x": int(x_c),
            "y": int(y_c),
            "r": int(r_c),
            "fraccion_claros": float(fraccion_claros),
            "estado": estado
        })
    return resultados_2


# Pastillas ordenadas
# ______________________________________________________________
def pastillas_ordendas(resultados_2):

    num_columnas = 5
    num_filas = 2   # porque son 10 pastillas
    print("\n--- Blister ordenado 5x2 (solo visualización) ---\n")
    index = 0
    for fila in range(num_filas):
        fila_texto = []
        for col in range(num_columnas):
            if index < len(resultados_2):
                estado = resultados_2[index]["estado"]
                fila_texto.append(f"{estado}")
            else:
                fila_texto.append("—")
            index += 1
        
        print(f"Fila {fila+1}: {fila_texto}")


# Detectar rotas/incompletas
# ______________________________________________________________
def rotas_incompletas(resultados_2,gray_blister_clahe):
    # Parámetros HoughLinesP
    min_line_length = 12
    max_line_gap =6
    threshold_hough = 20

    # Recorrer todas las pastillas
    for i, r_dict in enumerate(resultados_2):
        # Saltar pastillas vacías
        if r_dict['estado'] == 'vacía':
            continue

        x_c, y_c, r_c = r_dict["x"], r_dict["y"], r_dict["r"]

        # Máscara circular
        mask = np.zeros_like(gray_blister_clahe, dtype=np.uint8)
        cv2.circle(mask, (x_c, y_c), r_c - 12, 255, -1)

        # Extraer región
        region = cv2.bitwise_and(gray_blister_clahe, gray_blister_clahe, mask=mask)

        # Recortar bounding box
        x1, y1 = max(x_c - r_c, 0), max(y_c - r_c, 0)
        x2, y2 = min(x_c + r_c, gray_blister_clahe.shape[1]-1), min(y_c + r_c, gray_blister_clahe.shape[0]-1)
        roi = region[y1:y2, x1:x2]

        # Bordes con Canny
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        roi_clahe = clahe.apply(roi)
        roi_blur = cv2.GaussianBlur(roi, (3,3), 0)
        roi_thresh = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(roi_clahe, 50, 150)

        # Detectar líneas
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold_hough,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)

        # Convertir ROI a color para dibujar líneas
        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # Dibujar líneas detectadas
        if lines is not None:
            for x1l, y1l, x2l, y2l in lines[:,0]:
                cv2.line(roi_color, (x1l, y1l), (x2l, y2l), (255,0,0), 1)  # azul para que destaque
            r_dict["estado"] = "rota/incompleta"

        # ROTURAS MUY FINAS
        var_roi = np.var(roi[roi>0])
        bordes = np.sum(edges > 0)
        if var_roi > 100 and bordes > 20: # mucha varianza o bordes
            r_dict["estado"] = "rota/incompleta"

        # Mostrar con matplotlib
        #imshow(f"Pastilla {i+1} - {r_dict['estado']}", roi_color, size=4)
    return resultados_2
    


# PROCESAR TODAS LAS IMÁGENES
# ______________________________________________________________
image_folder = "img_depurado"
resumen_imagenes = []

tiempo_procesado_total = 0  # ← tiempo sin lectura de imágenes

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):

        # No medir tiempo de carga
        img = cv2.imread(os.path.join(image_folder, filename))
        # Medir solo el procesado
        t0 = time.time()

        gray_blister_clahe, blister_crop = find_edges(img)

        circles = cv2.HoughCircles(gray_blister_clahe, cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=75, param1=100, param2=10,
                                   minRadius=25, maxRadius=35)

        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
        else:
            circles = []

        resultados_2 = clasificar(gray_blister_clahe, circles)
        resultados_2 = rotas_incompletas(resultados_2, gray_blister_clahe)


        t1 = time.time()
        tiempo_img = t1 - t0
        tiempo_procesado_total += tiempo_img

        # Salida por imagen
        print(f"{filename}: Procesado puro = {tiempo_img:.3f} s")

        vacia = sum(1 for r in resultados_2 if r['estado'] == "vacía")
        rota_incompleta = sum(1 for r in resultados_2 if r['estado'] == "rota/incompleta")
        resumen_imagenes.append((filename, (vacia, rota_incompleta)))


# TIEMPOS FINALES
# _______________________________________________________________
total = len(resumen_imagenes)

print("\n------------------------------------------")
print(f"TIEMPO TOTAL DE PROCESADO PURO: {tiempo_procesado_total:.3f} s")
print(f"TIEMPO PROMEDIO POR IMAGEN: {tiempo_procesado_total/total:.4f} s")
print("------------------------------------------")