from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from parte1__procesamiento_binarios import unpack_drawings
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'data', 'flower.bin')

def rasterize_drawing(image_data, size=32, padding=5):
    img = Image.new("L", (256, 256), 0)  
    draw = ImageDraw.Draw(img)
    for x, y in image_data:
        draw.line(list(zip(x, y)), fill=255, width=2)
    img = img.resize((size, size), resample=Image.Resampling.LANCZOS)

    return np.array(img)

for i, drawing in enumerate(unpack_drawings(file_path)):
    image_data = drawing['image']
    img_array = rasterize_drawing(image_data, size=32)

    plt.imshow(img_array, cmap='gray')
    plt.title(f'Dibujo {i+1}')
    plt.axis('off')
    plt.show()

    if i == 9:  # mostrar solo los primeros 10
        break
