Restauración de fotos antiguas con Pix2Pix

Proyecto de Redes Neuronales y Deep Learning. Una red generativa condicional (Pix2Pix) que recibe una foto antigua y dañada y devuelve una versión restaurada en blanco y negro.

Rayones, manchas, polvo, grano, tono sepia y desenfoque: la red aprende a quitarlos.
<img width="1493" height="2710" alt="image" src="https://github.com/user-attachments/assets/b4aa9f85-a9de-4961-ba22-0a0f42377f3c" />

Resultados

Evaluado en 50 fotos de prueba que la red nunca vio, comparando contra la foto original.

Método	PSNR (dB) ↑	SSIM ↑
Foto dañada (sin procesar)	17,51	0,576
Filtro clásico (mediano + bilateral, sin redes)	17,33	0,592
Red Pix2Pix	21,38	0,660

La red mejora +3,9 dB de PSNR frente a la foto dañada. El filtro clásico no mejora: quita ruido pero borra detalle.

Por categoría (PSNR, dB)
Categoría	Foto dañada	Red Pix2Pix	Mejora
Gatos	17,04	21,51	+4,5
Perros	18,16	22,24	+4,1
Personas	17,06	21,07	+4,0
Casas	17,85	21,03	+3,2

La prueba tiene solo entre 10 y 15 fotos por categoría, así que las diferencias entre categorías deben leerse con cautela.

Cómo funciona
1. Data set personalizado

No existen pares «foto vieja / foto original», así que se fabricaron: se tomaron fotos modernas y se envejecieron con código (envejecer_foto).

497 fotos limpias tras quitar 3 duplicados exactos (150 personas, 147 casas, 100 perros, 100 gatos).
4 versiones dañadas al azar por cada foto de entrenamiento.
1788 pares de entrenamiento (447 fotos × 4) y 50 fotos de prueba (10 % de cada categoría, 1 versión cada una).
La división se hace por foto original y por categoría, antes de envejecer, para evitar fuga de datos.

Daños simulados (todos aleatorios, con intensidad variable): blanco y negro / sepia / color desteñido, pérdida de contraste, desenfoque gaussiano, grano, rayones y dobleces, manchas, polvo, viñeteado y compresión JPEG de baja calidad.

2. Arquitectura
Componente	Descripción	Parámetros
Generador	U-Net: 8 capas de bajada hasta un código 1×1×512 y 8 de subida hasta 256×256, con conexiones de salto. Salida tanh.	54,4 M
Discriminador	PatchGAN condicional: recibe la foto dañada y una limpia (6 canales) y devuelve un mapa 30×30 de veredictos (un parche de ~70×70 px cada uno).	2,8 M
3. Función de pérdida
Generador: adversaria + λ · L1, con λ = 100.
Discriminador: entropía cruzada binaria (real → 1, generada → 0).
4. Entrenamiento
Parámetro	Valor
Tamaño de imagen	256 × 256
Pasos	20 000 (≈ 11 épocas)
Batch size	1
Optimizador	Adam, lr = 0,0002, β₁ = 0,5
Aumento de datos	Redimensionar a 286, recorte aleatorio 256 y volteo horizontal
Hardware	GPU T4 en Google Colab (≈ 41 min)
Semilla	42

La pérdida L1 baja de 0,49 a cerca de 0,13. Las pérdidas adversaria y del discriminador oscilan sin llegar a cero, como es esperable en una GAN.

Estructura del repositorio
.
├── Miniproyecto_3_RedesNeuronales.ipynb   # Notebook completo (datos, modelo, entrenamiento, validación)
└── README.md

El data set (fotos limpias) no está en el repositorio por su tamaño: se descarga desde la sección Releases.

Data set

El data set (Dataset.zip, ~221 MB, 500 imágenes de personas, casas, perros y gatos) es demasiado pesado para el repositorio, así que está publicado como Release:

Descargar Dataset.zip

Al descomprimirlo queda la carpeta fotos_restauracion/ con una subcarpeta por categoría (personas/, casas/, perros/, gatos/). Son 500 fotos; el notebook quita 3 duplicados exactos y trabaja con 497.

Cómo ejecutarlo

El notebook está pensado para Google Colab con GPU.

Abre el notebook en Colab y activa la GPU: Entorno de ejecución → Cambiar tipo de entorno → GPU (T4).
Descarga y descomprime el data set en la primera celda de código:
!wget -q https://github.com/Julian-Montoya-G/Redes-Neuronales/releases/download/dataset-v1/Dataset.zip
!unzip -q Dataset.zip
En la celda 0, apunta DIR_DRIVE a la carpeta descomprimida (en lugar de la carpeta de Drive del grupo):
python
DIR_DRIVE = "/content/fotos_restauracion"

Si prefieres usar Google Drive, sube fotos_restauracion/ a Mi unidad/ y deja la ruta original. 4. Ejecuta las celdas en orden. El notebook monta Drive, quita duplicados, genera el data set de pares, entrena y valida. 5. Los checkpoints se guardan en Mi unidad/checkpoints_restauracion cada 5000 pasos, así que si Colab se desconecta puedes retomar el entrenamiento (por eso se sigue montando Drive). 6. Al final, la sección 8 permite subir fotos antiguas reales para probarlas, y la sección 9 guarda el generador en generador_restauracion.keras.

Dependencias

Colab ya trae casi todo. Las versiones usadas:

Python 3, TensorFlow 2.21
OpenCV (opencv-python)
NumPy, Pandas, Matplotlib
Parámetros principales (celda 0)
Variable	Valor	Para qué sirve
TAM	256	Tamaño de las imágenes
VERSIONES_POR_FOTO	4	Versiones dañadas por cada foto limpia
PORC_TEST	0.10	Fracción de cada categoría reservada para prueba
COLORIZAR	False	False: salida en blanco y negro; True: restaurar y colorizar
BATCH_SIZE	1	Fotos por paso
LAMBDA	100	Peso de L1 frente a la pérdida adversaria
PASOS	20000	Pasos de entrenamiento
Métricas
PSNR (dB): error píxel a píxel respecto a la foto original. Más alto es mejor; cada +3 dB reduce el error cuadrático a la mitad.
SSIM (0 a 1): similitud de estructura que percibe el ojo. 1 = idéntica.

La foto dañada y el resultado del filtro clásico se pasan a gris antes de medir, igual que el objetivo, para que la comparación sea justa.

Limitaciones
Brecha de dominio: la red aprendió con daños simulados. En fotos antiguas reales quita el sepia y el velo amarillento y sube el contraste, pero aparecen texturas tipo grieta.
Los rayones gruesos dejan rastro en la restauración.
λ = 100 y el resto de hiperparámetros son los del artículo original; no se ajustaron.
La validación usa training=True en el generador, como en el artículo de Pix2Pix, lo que añade algo de aleatoriedad.
PSNR y SSIM no miden si la foto «se ve real».
Próximos pasos
Simular daños más realistas.
Entrenar a mayor resolución.
Añadir colorización (COLORIZAR = True).
Referencias
Isola, P., Zhu, J.-Y., Zhou, T., Efros, A. A. Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix), CVPR 2017.
Tutorial de Pix2Pix de TensorFlow.

Autores
Julián Montoya
Jóse Daniel Gutierrez
Julián Andrés Montoya
Valeria García

Curso de Redes Neuronales y Deep Learning · Universidad / Institución · 2026
