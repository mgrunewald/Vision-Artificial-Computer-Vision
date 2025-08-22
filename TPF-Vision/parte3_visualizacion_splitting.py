# PARA VER SI A PESAR DE LA DIVISION ALEATORIA DE LOS DATOS, LAS CLASES SIGUEN CORRESPONDIENTO A LAS MISMAS IMAGENES
import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
splitting_dir = os.path.join(script_dir, 'splitting')

X_train = np.load(os.path.join(splitting_dir, 'X_train.npy'), allow_pickle=True)
y_train = np.load(os.path.join(splitting_dir, 'y_train.npy'))
X_val = np.load(os.path.join(splitting_dir, 'X_val.npy'), allow_pickle=True)
y_val = np.load(os.path.join(splitting_dir, 'y_val.npy'))
X_test = np.load(os.path.join(splitting_dir, 'X_test.npy'), allow_pickle=True)
y_test = np.load(os.path.join(splitting_dir, 'y_test.npy'))

class_names = np.load(os.path.join(script_dir, 'rasterization', 'class_names.npy'), allow_pickle=True)

datasets = {
    'Train': (X_train, y_train),
    'Validation': (X_val, y_val),
    'Test': (X_test, y_test)
}

for name, (X, y) in datasets.items():
    print(f"Mostrando ejemplos de: {name}")
    for _ in range(3):  
        idx = np.random.randint(0, len(X))
        img = np.vectorize(int)(X[idx]).astype(np.uint8)

        plt.imshow(img, cmap='gray')
        plt.title(f"{name} - Clase: {class_names[y[idx]]} (label {y[idx]})")
        plt.axis('off')
        plt.show()
