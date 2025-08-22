# VISUALIZA LOS TRAZOS DE LOS DIBUJOS EN UN ARCHIVO BINARIO
import os
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from parte1__procesamiento_binarios import unpack_drawings

def plot_drawings(filename, num_to_show=3):
    count = 0
    for drawing in unpack_drawings(filename):
        image = drawing['image']
        plt.figure()
        for x, y in image:
            plt.plot(x, y)
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.title(f"Dibujo {count + 1} - País: {drawing['country_code'].decode()} - Reconocido: {bool(drawing['recognized'])}")
        count += 1
        if count >= num_to_show:
            break
    plt.show()

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'data', 'flower.bin')
#elegir el archivo .bin que se quiera visualizar 
plot_drawings(file_path, num_to_show=3)