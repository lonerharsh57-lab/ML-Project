import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# put the CICIDS2017 csv files inside data/
files = list(DATA_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError("place the CICIDS2017 CSV files inside the data/ folder")

frames = []
for f in files:
    df = pd.read_csv(f, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
data = data.replace([np.inf, -np.inf], np.nan)
data = data.drop_duplicates()

label_col = next((c for c in ["Label", "Attack Type", "Attack_Type"] if c in data.columns), None)
labels = data[label_col].astype(str).str.strip() if label_col else None

X = data.drop(columns=[label_col], errors="ignore").copy()

# keep numeric columns only for distance-based clustering
X = X.select_dtypes(include=[np.number])
X = X.dropna(axis=1, how="all")
X = X.fillna(X.median(numeric_only=True))

# remove constant columns
nunique = X.nunique()
X = X.loc[:, nunique > 1]

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# basic EDA
summary = X.describe().T
summary.to_csv(OUT_DIR / "feature_summary.csv")

if labels is not None:
    labels.value_counts(dropna=False).to_csv(OUT_DIR / "attack_label_counts.csv")

# PCA visualization
pca = PCA(n_components=2, random_state=42)
Xp = pca.fit_transform(Xs)

plt.figure(figsize=(8, 6))
plt.scatter(Xp[:, 0], Xp[:, 1], s=3, alpha=0.35)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("CICIDS2017 PCA projection")
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_projection.png", dpi=200)
plt.close()

# sample for computationally heavy clustering if needed
max_rows = 100000
rng = np.random.default_rng(42)
idx = rng.choice(len(Xs), size=min(max_rows, len(Xs)), replace=False)
Xs_model = Xs[idx]
labels_model = labels.iloc[idx].reset_index(drop=True) if labels is not None else None

rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    pred = km.fit_predict(Xs_model)
    sil = silhouette_score(Xs_model, pred)
    rows.append({"model": "kmeans", "k": k, "silhouette": sil, "inertia": km.inertia_})

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT_DIR / "kmeans_metrics.csv", index=False)

best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_pred = km.fit_predict(Xs_model)

hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
hier_pred = hier.fit_predict(Xs_model)

results = pd.DataFrame({
    "kmeans_cluster": km_pred,
    "hierarchical_cluster": hier_pred
})
if labels_model is not None:
    results["attack_label"] = labels_model
results.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

print("best k:", best_k)
print("kmeans silhouette:", silhouette_score(Xs_model, km_pred))
print("hierarchical silhouette:", silhouette_score(Xs_model, hier_pred))
