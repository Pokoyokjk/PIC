import cv2
import numpy as np
import time

#Definicion de funciones:
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

def recortar_blister(img, contour):
    """
    Recorta la región del blister usando su bounding box.
    """
    x, y, w, h = cv2.boundingRect(contour)
    blister = img[y:y+h, x:x+w]
    return blister

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

# --- Captura de cámara en tiempo real ---
cap = cv2.VideoCapture(0)  # 1 = WEBCAM

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame")
        break

        # Timer inicio de procesamiento
    t0 = time.perf_counter()

    # Reducir tamaño para acelerar procesamiento
    frame_resized = cv2.resize(frame, (960, 720))

    # --- Procesamiento ---
    try:
        blister_contour = find_blister_contour(frame_resized)
        blister = recortar_blister(frame_resized, blister_contour)
        img_circles, num_blisters, niveles, cavidades = detectar_cavidades_contornos(
            blister, min_area=1000, circularity_thresh=(0.8, 1.1)
        )

        # --- Crear imagen de salida del tamaño original ---
        output = frame_resized.copy()

        # Insertar el blister procesado dentro del frame
        h, w = img_circles.shape[:2]
        output[0:h, 0:w] = img_circles

        # --- Información a mostrar ---
        level_prom = np.mean(niveles) if len(niveles) > 0 else 0

        # Timer fin de procesamiento
        t1 = time.perf_counter()
        tiempo_procesamiento = (t1 - t0) * 1000  # en ms    

        
        # Mostrar información
        cv2.putText(output, f"Cavidades: {num_blisters}", (10, h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(output, f"Nivel promedio: {level_prom:.1f}", (10, h + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Mostrar tiempo de procesamiento
        cv2.putText(output, f"Tiempo proc: {tiempo_procesamiento:.1f} ms", (10, h + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Calcular FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(output, f"FPS: {fps:.1f}", (10, h + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Blister Detection", output)


    except Exception as e:
        # En caso de que no se detecte el contorno principal
        cv2.putText(frame_resized, "No se detecto blister", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Blister Detection", frame_resized)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
