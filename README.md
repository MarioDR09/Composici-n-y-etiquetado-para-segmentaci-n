# Composici-n-y-etiquetado-para-segmentaci-n
Este directorio cuenta con el archivo de composición y etiquetado; así como las respectivas redes para segmentación de imágenes.

# Crear imágenes y máscaras sintéticas
En esta sección, utilizaremos "image_composition.py" para seleccionar al azar los primeros planos y superponerlos automáticamente en los fondos. Necesitarás varios recortes de primer plano con fondos transparentes. Por ejemplo, puede tener una imagen de un águila con un fondo completamente transparente. Debido a la necesidad de transparencia, estas imágenes deben estar en formato .png (.jpg no tiene transparencia). Corté mi primer plano con [GIMP] (https://www.gimp.org/), que es gratis.

## Configuración del directorio del conjunto de datos personalizado:
- Dentro del directorio "conjuntos de datos", cree una nueva carpeta para su conjunto de datos (por ejemplo, "recibos")
- Dentro de ese directorio de conjunto de datos, cree una carpeta llamada "entrada"
- Dentro de "entrada", cree dos carpetas llamadas "foregrounds" y "backgrounds"
- Dentro de "primer plano", cree una carpeta para cada súper categoría (por ejemplo, "agua", "cfe")
- Dentro de cada carpeta de súper categoría de primer plano, cree una carpeta para cada categoría 
- Dentro de cada carpeta de categoría, agregue todas las fotos de primer plano que desee usar para la categoría respectiva (por ejemplo, todos los recortes de primer plano de águila)
- Dentro de "fondos", agrega todas las fotos de fondo que pretendes usar

Ejecute "img_comp.py" para crear sus imágenes y máscaras.
```
python ./python/img_comp.py --input_dir ./datasets/box_dataset_synthetic/input --output_dir ./datasets/box_dataset_synthetic/output --count 10 --width 512 --height 512
```
