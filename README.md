# Detección y Clasificación de Cavidades en Blísteres
## Procesamiento de Imágenes por Computador
### Autores
- **Alexander Solís Quesada**
- **Juan Diego Cruz Salazar**
- **Nicole Delgadillo Sánchez**
- **Mª Gloria Cumia Espinosa de los Monteros**

## Descripción del proyecto
Este proyecto aborda la detección y clasificación automática de cavidades en blísteres farmacéuticos mediante técnicas clásicas de visión por computador. El objetivo principal es identificar cavidades **vacías**, **completas** y **rotas/incompletas**, evaluando distintos enfoques en términos de precisión y coste computacional.
El sistema ha sido diseñado para funcionar tanto en **modo offline** (procesado de imágenes) como en **tiempo real**, evitando el uso de modelos de aprendizaje profundo y priorizando soluciones explicables y eficientes.

## Estructura del repositorio
├── .venv/
│ └── Entorno virtual de Python
│
├── img_depurado/
│ └── Conjunto de imágenes utilizadas para evaluación
│
├── Funcionamiento_t_real.mp4
│ └── Vídeo demostrativo del funcionamiento en tiempo real
│
├── deteccion_real_time.py
│ └── Detección en tiempo real usando cámara
│
├── label_depurado.txt
├── label_depurado2.txt
│ └── Etiquetas reales de cavidades vacías (ground truth)
│
├── label_rotas.txt
│ └── Etiquetas reales de cavidades rotas/incompletas
│
├── main_celdas_t_final.py
│ └── Método final basado en división en celdas
│
├── main_deteccion_t_final.py
│ └── Método final de detección geométrica (Hough)
│
├── main_rotas_t_final.py
│ └── Método final con reglas de área y circularidad
│
├── *.ipynb
│ └── Notebooks de desarrollo, pruebas y análisis intermedios

## Métodos implementados
### Detección basada en celdas
- División del blíster en una rejilla fija (2×5).
- Clasificación por intensidad media de cada celda.
- Muy rápido, pero sensible a iluminación y alineación.

### Detección geométrica
- Detección de cavidades circulares mediante Transformada de Hough.
- Clasificación basada en histogramas de intensidad.
- Mayor robustez geométrica.

### Detección geométrica con reglas adicionales
- Evaluación del área y la circularidad de las cavidades.
- Detección más fiable de cavidades rotas o incompletas.
- Mejor equilibrio entre precisión y coste computacional.


