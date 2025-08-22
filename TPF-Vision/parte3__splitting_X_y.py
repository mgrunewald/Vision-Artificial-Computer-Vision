import numpy as np
from sklearn.model_selection import train_test_split
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

images_path = os.path.join(script_dir, 'rasterization', 'images_list.npy')
labels_path = os.path.join(script_dir, 'rasterization', 'labels_list.npy')

X = np.load(images_path, allow_pickle=True)
y = np.load(labels_path, allow_pickle=True)

X = [np.array(img, dtype=np.uint8) for img in X]
y = np.array(y, dtype=np.int64)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

splitting_dir = os.path.join(script_dir, 'splitting')
os.makedirs(splitting_dir, exist_ok=True)

np.save(os.path.join(splitting_dir, 'X_train.npy'), np.array(X_train, dtype=object))
np.save(os.path.join(splitting_dir, 'X_val.npy'), np.array(X_val, dtype=object))
np.save(os.path.join(splitting_dir, 'X_test.npy'), np.array(X_test, dtype=object))

np.save(os.path.join(splitting_dir, 'y_train.npy'), y_train)
np.save(os.path.join(splitting_dir, 'y_val.npy'), y_val)
np.save(os.path.join(splitting_dir, 'y_test.npy'), y_test)
