import os
import numpy as np
from PIL import Image, ImageDraw
from parte1__procesamiento_binarios import unpack_drawings

def rasterize_drawing(image_data, size=32):
    img = Image.new("L", (256, 256), 0)
    draw = ImageDraw.Draw(img)
    for x, y in image_data:
        draw.line(list(zip(x, y)), fill=255, width=2)
    img = img.resize((size, size), resample=Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)

classes = ['apple', 'butterfly', 'cloud', 'eye', 'flower', 'ice cream', 'moon', 'rainbow', 'smiley face', 'star', 'sun', 'tree']

all_images = [] 
all_labels = [] 
class_names = []

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'rasterization')
os.makedirs(output_dir, exist_ok=True)

for class_index, class_name in enumerate(classes):
    class_names.append(class_name)
    
    file_path = os.path.join(script_dir, 'data', f'{class_name}.bin')
    for i, drawing in enumerate(unpack_drawings(file_path)):
        image_data = drawing['image']
        img_array = rasterize_drawing(image_data, size=32)
        all_images.append(img_array)
        all_labels.append(class_index)

np.save(os.path.join(output_dir, 'images_list.npy'), np.array(all_images, dtype=object))
np.save(os.path.join(output_dir, 'labels_list.npy'), np.array(all_labels))
np.save(os.path.join(output_dir, 'class_names.npy'), np.array(class_names))

