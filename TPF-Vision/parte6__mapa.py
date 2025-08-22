import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
split_dir = os.path.join(script_dir, 'splitting')
class_names = np.load(os.path.join(script_dir, 'rasterization', 'class_names.npy'), allow_pickle=True)

X_train = np.load(os.path.join(split_dir, 'X_train.npy'), allow_pickle=True)
y_train = np.load(os.path.join(split_dir, 'y_train.npy'))

X_arr = np.stack([np.array(img, dtype=np.float32) for img in X_train])
y_arr = np.array(y_train)

num_classes = len(class_names)
rows, cols = 3, 4 

fig, axs = plt.subplots(rows, cols, figsize=(12, 9))

for i in range(num_classes):
    row, col = divmod(i, cols)
    class_imgs = X_arr[y_arr == i]
    mean_img = np.mean(class_imgs, axis=0)

    axs[row, col].imshow(mean_img, cmap='hot')
    axs[row, col].set_title(class_names[i])
    axs[row, col].axis('off')

for j in range(num_classes, rows * cols):
    row, col = divmod(j, cols)
    axs[row, col].axis('off')

plt.tight_layout()

plot_path = os.path.join(script_dir, 'mapa_calor.png')
plt.savefig(plot_path)
plt.close()

plt.show()
