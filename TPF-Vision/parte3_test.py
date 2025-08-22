import os
import numpy as np 
from collections import Counter
import matplotlib.pyplot as plt

# === CARGA DE DATOS ===
script_dir = os.path.dirname(os.path.abspath(__file__))
splitting_dir = os.path.join(script_dir, 'splitting')

X_train = np.load(os.path.join(splitting_dir, 'X_train.npy'), allow_pickle=True)
X_test = np.load(os.path.join(splitting_dir, 'X_test.npy'), allow_pickle=True)
X_val = np.load(os.path.join(splitting_dir, 'X_val.npy'), allow_pickle=True)

y_train = np.load(os.path.join(splitting_dir, 'y_train.npy'))
y_test = np.load(os.path.join(splitting_dir, 'y_test.npy'))
y_val = np.load(os.path.join(splitting_dir, 'y_val.npy'))

# === CUENTAS POR CLASE ===
train_counts = Counter(y_train)
test_counts = Counter(y_test)
val_counts = Counter(y_val)

labels = sorted(set(y_train) | set(y_test) | set(y_val))

train_vals = [train_counts.get(label, 0) for label in labels]
val_vals = [val_counts.get(label, 0) for label in labels]
test_vals = [test_counts.get(label, 0) for label in labels]

y = np.arange(len(labels)) 
height = 0.25

# === PLOTEO HORIZONTAL ===
fig, ax = plt.subplots(figsize=(12, 8))

ax.barh(y - height, train_vals, height, label='Train', color='#ffe008')
ax.barh(y, val_vals, height, label='Validation', color='#fff48d')
ax.barh(y + height, test_vals, height, label='Test', color='#fff9c4')

ax.set_ylabel('Label')
ax.set_xlabel('Cantidad de imágenes')
ax.set_title('Distribución de clases en train, validation y test')
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()
