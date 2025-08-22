import os
from parte1__procesamiento_binarios import unpack_drawings 

classes = ['apple', 'butterfly', 'cloud', 'eye', 'flower', 'ice cream', 'moon', 'rainbow', 'smiley face', 'star', 'sun', 'tree']

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

conteo_por_clase = {}

for class_name in classes:
    bin_path = os.path.join(data_dir, f'{class_name}.bin')
    count = sum(1 for _ in unpack_drawings(bin_path))
    conteo_por_clase[class_name] = count

print("\nCantidad de dibujos por clase:")
for clase, cantidad in conteo_por_clase.items():
    print(f"{clase}: {cantidad}")

total = sum(conteo_por_clase.values())
print(f"\nTotal de dibujos en todas las clases: {total}")


"""

Cantidad de dibujos por clase
    apple: 144722
    butterfly: 117999
    cloud: 120265
    eye: 125888
    flower: 144818
    ice cream: 123133
    moon: 121661
    rainbow: 126845
    smiley face: 124386
    star: 137619
    sun: 133781
    tree: 144721

    Total de dibujos en todas las clases: 1565838

"""