"""
UMAP Visualization: Original vs IFOSS Preprocessing

This script generates a 2×2 UMAP visualization showing:

1) Original training dataset
2) Original testing dataset
3) Training data after Isolation Forest filtering
4) Training data after One-Sided Selection undersampling

Required variables in the session:

- Xl, yl     : original training features and labels
- Xl_t, yl_t : external test features and labels
- X1, y1     : data after Isolation Forest
- X, y       : data after One-Sided Selection

Labels must be encoded as integers (e.g., 0 = Negative, 1 = Positive).

Output:
A 2×2 UMAP figure comparing dataset structure before and after IFOSS.
"""

import numpy as np
import umap
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['font.family'] = 'serif'

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

plot_colors = ['green', 'red', 'blue', 'orange']

fig.suptitle(
    "Comorbidities + Laboratory Information + Symptoms + Vital Sign Dataset (Survival Status)",
    fontsize=14,
    fontweight='bold'
)

reducer1 = umap.UMAP(
    n_neighbors=10,
    n_epochs=100,
    learning_rate=0.0009,
    random_state=42,
    transform_seed=42,
    verbose=False
)
reducer1.fit(Xl, yl)
supervised1 = reducer1.transform(Xl)

for i, label in enumerate(yl):
    axes[0, 0].scatter(
        supervised1[i, 0],
        supervised1[i, 1],
        color=plot_colors[label],
        label=label,
    )
axes[0, 0].set_title("Training Dataset", fontsize=12, fontweight='bold')


reducer2 = umap.UMAP(
    n_neighbors=9,
    n_epochs=100,
    learning_rate=0.0009,
    random_state=42,
    transform_seed=42,
    verbose=False
)
reducer2.fit(Xl_t, yl_t)
supervised2 = reducer2.transform(Xl_t)

for i, label in enumerate(yl_t):
    axes[0, 1].scatter(
        supervised2[i, 0],
        supervised2[i, 1],
        color=plot_colors[label],
        label=label,
    )
axes[0, 1].set_title("Testing Dataset", fontsize=12, fontweight='bold')


reducer3 = umap.UMAP(
    n_neighbors=10,
    n_epochs=100,
    learning_rate=0.009,
    random_state=42,
    transform_seed=42,
    verbose=False
)
reducer3.fit(X1, y1)
supervised3 = reducer3.transform(X1)

for i, label in enumerate(y1):
    axes[1, 0].scatter(
        supervised3[i, 0],
        supervised3[i, 1],
        color=plot_colors[label],
        label=label,
    )
axes[1, 0].set_title("After Isolation Forest", fontsize=12, fontweight='bold')


reducer4 = umap.UMAP(
    n_neighbors=12,
    n_epochs=2000,
    learning_rate=0.0009,
    random_state=42,
    transform_seed=42,
    verbose=False
)
reducer4.fit(X, y)
supervised4 = reducer4.transform(X)

for i, label in enumerate(y):
    axes[1, 1].scatter(
        supervised4[i, 0],
        supervised4[i, 1],
        color=plot_colors[label],
        label=label,
    )
axes[1, 1].set_title("After One-Sided Selection", fontsize=12, fontweight='bold')


legend_handles = [
    plt.Line2D([0], [0], color='green', lw=8, label='Negative Class'),
    plt.Line2D([0], [0], color='red', lw=8, label='Positive Class')
]

fig.legend(
    handles=legend_handles,
    labels=['Negative Class', 'Positive Class'],
    loc='upper right',
    bbox_to_anchor=(1.2, 0.92)
)

plt.tight_layout()
plt.show()
