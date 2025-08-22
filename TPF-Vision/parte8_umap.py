import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap  
import pandas as pd
import seaborn as sns

from parte4__CNN import SimpleCNN

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'modelo_cnn_final.pth')
    split_dir  = os.path.join(script_dir, 'splitting')
    raster_dir = os.path.join(script_dir, 'rasterization')

    class_names = np.load(os.path.join(raster_dir, 'class_names.npy'), allow_pickle=True)
    X_test      = np.load(os.path.join(split_dir, 'X_test.npy'), allow_pickle=True)
    y_test      = np.load(os.path.join(split_dir, 'y_test.npy'))

    X_arr    = np.stack([np.array(img, dtype=np.uint8) for img in X_test])
    X_tensor = torch.tensor(X_arr).unsqueeze(1).float() / 255.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SimpleCNN(num_classes=len(class_names), dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    colors = sns.color_palette("tab10", n_colors=len(class_names))

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

    per_class = 350
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

    df_umap = pd.DataFrame({
        "UMAP1": embeds[:, 0],
        "UMAP2": embeds[:, 1],
        "Clase": [class_names[i] for i in labs_sel]
    })

    sns.set_style("white")

    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    axs = axs.flatten()

    for i, cls in enumerate(class_names):
        ax = axs[i]
        df_focus = df_umap[df_umap["Clase"] == cls]
        
        ax.scatter(df_umap["UMAP1"], df_umap["UMAP2"], s=10, alpha=0.2, color='gray')
        
        ax.scatter(df_focus["UMAP1"], df_focus["UMAP2"], s=20, alpha=0.8, label=cls)
        
        sns.kdeplot(
            data=df_focus,
            x="UMAP1", y="UMAP2",
            fill=True,
            color=colors[i],
            alpha=0.4,
            levels=5,
            linewidths=1,
            ax=ax
        )

        ax.set_title(f"Clase: {cls}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "umap_por_clase_DA.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
