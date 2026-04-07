# Face Recognition using Principal Component Analysis (PCA)
# Dataset: Labeled Faces in the Wild (LFW)
# Author: shricharan17

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.datasets import fetch_lfw_people

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────
print("Loading LFW dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names
n_samples, n_features = X.shape
n_classes = target_names.shape[0]

print(f"Samples  : {n_samples}")
print(f"Features : {n_features}")
print(f"Classes  : {n_classes} → {list(target_names)}")

# ─────────────────────────────────────────
# 2. Train / Test Split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 3. Standardise
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────
# 4. PCA — Extract Eigenfaces
# ─────────────────────────────────────────
n_components = 150
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=42)
pca.fit(X_train_sc)

X_train_pca = pca.transform(X_train_sc)
X_test_pca  = pca.transform(X_test_sc)

print(f"\nVariance explained by {n_components} components: "
      f"{pca.explained_variance_ratio_.sum()*100:.1f}%")

# ─────────────────────────────────────────
# 5. Visualise Eigenfaces
# ─────────────────────────────────────────
h, w = lfw_people.images.shape[1], lfw_people.images.shape[2]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[idx].reshape(h, w), cmap='bone')
    ax.set_title(f'Eigenface {idx+1}', fontsize=9)
    ax.axis('off')
plt.suptitle('Top 10 Eigenfaces', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eigenfaces.png', dpi=150)
plt.show()
print("Saved: outputs/eigenfaces.png")

# ─────────────────────────────────────────
# 6. Explained Variance Curve
# ─────────────────────────────────────────
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 4))
plt.plot(cumvar, color='steelblue', linewidth=2)
plt.axhline(0.95, linestyle='--', color='tomato', label='95% threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA — Explained Variance Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/variance_curve.png', dpi=150)
plt.show()
print("Saved: outputs/variance_curve.png")

# ─────────────────────────────────────────
# 7. Train KNN Classifier
# ─────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_pca, y_train)

# ─────────────────────────────────────────
# 8. Evaluate
# ─────────────────────────────────────────
y_pred = knn.predict(X_test_pca)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score(y_test, y_pred, average='weighted')
f1   = f1_score(y_test, y_pred, average='weighted')

print("\n── Performance Metrics ──────────────────")
print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {prec*100:.2f}%")
print(f"Recall    : {rec*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")
print("\n── Classification Report ────────────────")
print(classification_report(y_test, y_pred, target_names=target_names))

# ─────────────────────────────────────────
# 9. Confusion Matrix
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix — PCA Face Recognition')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.show()
print("Saved: outputs/confusion_matrix.png")

# ─────────────────────────────────────────
# 10. Cross-Validation
# ─────────────────────────────────────────
cv_scores = cross_val_score(knn, X_train_pca, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
