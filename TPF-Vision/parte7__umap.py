import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap  

from parte4__CNN import SimpleCNN

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'modelo_cnn.pth')
    split_dir  = os.path.join(script_dir, 'splitting')
    raster_dir = os.path.join(script_dir, 'rasterization')
    umap_file  = os.path.join(script_dir, 'umap_250x12doodles.png')

    class_names = np.load(os.path.join(raster_dir, 'class_names.npy'), allow_pickle=True)
    X_test      = np.load(os.path.join(split_dir, 'X_test.npy'), allow_pickle=True)
    y_test      = np.load(os.path.join(split_dir, 'y_test.npy'))

    X_arr    = np.stack([np.array(img, dtype=np.uint8) for img in X_test])
    X_tensor = torch.tensor(X_arr).unsqueeze(1).float() / 255.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SimpleCNN(num_classes=len(class_names), dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    features = []
    def hook_fn(module, inp, out):
        features.append(out.detach().cpu().numpy())
    model.body[7].register_forward_hook(hook_fn)

    all_feats = []
    for img in X_tensor:
        features.clear()
        _ = model(img.unsqueeze(0).to(device))
        fmap = features[0].squeeze(0)     
        gap  = fmap.mean(axis=(1,2))       
        all_feats.append(gap)
    all_feats = np.stack(all_feats)    

    per_class = 250
    idxs = []
    rng = np.random.default_rng(42)
    for cls in range(len(class_names)):
        locs = np.where(y_test == cls)[0]
        if len(locs) > per_class:
            chosen = rng.choice(locs, size=per_class, replace=False)
        else:
            chosen = locs
        idxs.extend(chosen.tolist())

    feats_sel = all_feats[idxs]          
    labs_sel  = y_test[idxs]              

    scaler       = StandardScaler()
    feats_scaled = scaler.fit_transform(feats_sel)

    n_components = min(20, feats_scaled.shape[0] - 1)
    if n_components > 0:
        pca = PCA(n_components=n_components, random_state=42)
        feats_reduced = pca.fit_transform(feats_scaled)
    else:
        feats_reduced = feats_scaled

    reducer = umap.UMAP(
        n_neighbors=30, 
        min_dist=0.05,    
        metric='cosine',
        random_state=42
    )
    embeds = reducer.fit_transform(feats_reduced)  

    plt.figure(figsize=(10, 8))
    for cls in range(len(class_names)):
        mask = (labs_sel == cls)
        plt.scatter(
            embeds[mask, 0],
            embeds[mask, 1],
            label=class_names[cls],
            s=50,
            alpha=0.7
        )
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.title(f"UMAP de activaciones (hasta {per_class} doodles/clase)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(umap_file, dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
