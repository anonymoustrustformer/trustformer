import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centroids = kmeans.cluster_centers_

# Create a larger shift for centroids
random_shift_large = np.random.uniform(-5, 5, centroids.shape)
new_centroids_large = centroids + random_shift_large

# Move data points according to the centroids' movement
X_moved_large = X.copy()
for i, delta in enumerate(new_centroids_large - centroids):
    X_moved_large[y_kmeans == i] += delta

# Set consistent axis limits
x_min_all = min(X[:, 0].min(), X_moved_large[:, 0].min()) - 5
x_max_all = max(X[:, 0].max(), X_moved_large[:, 0].max()) + 5
y_min_all = min(X[:, 1].min(), X_moved_large[:, 1].min()) - 5
y_max_all = max(X[:, 1].max(), X_moved_large[:, 1].max()) + 5

# Custom colormap for clusters
custom_cmap = list(plt.cm.get_cmap('viridis', 3).colors)

# Create the 3 plots side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Original clusters with local centroids (Local Centroids)
for i in range(3):
    axes[0].scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], c=[custom_cmap[i]], s=50, alpha=0.5)
    axes[0].scatter(centroids[i, 0], centroids[i, 1], c=[custom_cmap[i]], s=200, marker='X', edgecolor='black', linewidth=2)
axes[0].scatter([], [], c='none', edgecolor='black', s=200, marker='X', linewidth=2, label='Local Centroids')
axes[0].set_title('Local Model', fontsize=14)
axes[0].set_xlim(x_min_all, x_max_all)
axes[0].set_ylim(y_min_all, y_max_all)
axes[0].legend(loc='upper right')

# Plot 2: New centroids with delta annotations (Global Centroids)
for i in range(3):
    axes[1].scatter(centroids[i, 0], centroids[i, 1], c=[custom_cmap[i]], s=200, marker='X', edgecolor='black', linewidth=2, alpha=0.5)
    axes[1].scatter(new_centroids_large[i, 0], new_centroids_large[i, 1], c=[custom_cmap[i]], s=200, marker='^', edgecolor='black', linewidth=2)
    delta_label = f"$\\Delta f_{{1}}={new_centroids_large[i, 0] - centroids[i, 0]:.2f}, \\Delta f_{{2}}={new_centroids_large[i, 1] - centroids[i, 1]:.2f}$"
    text_x = (centroids[i, 0] + new_centroids_large[i, 0]) / 2 + 0.5
    text_y = (centroids[i, 1] + new_centroids_large[i, 1]) / 2 + 0.5
    axes[1].arrow(centroids[i, 0], centroids[i, 1], new_centroids_large[i, 0] - centroids[i, 0], new_centroids_large[i, 1] - centroids[i, 1],
                  head_width=0.2, head_length=0.3, fc='black', ec='black')
    axes[1].text(text_x, text_y, delta_label, fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
axes[1].scatter([], [], c='none', edgecolor='black', s=200, marker='X', linewidth=2, label='Local Centroids')
axes[1].scatter([], [], c='none', edgecolor='black', s=200, marker='^', linewidth=2, label='Global Centroids')
axes[1].set_title('Computing the Diff between global and local centroids', fontsize=14)
axes[1].set_xlim(x_min_all, x_max_all)
axes[1].set_ylim(y_min_all, y_max_all)
axes[1].legend(loc='upper right')

# Plot 3: Data points moved to global centroids (Data Points Moved by Global Centroids)
for i in range(3):
    axes[2].scatter(X_moved_large[y_kmeans == i, 0], X_moved_large[y_kmeans == i, 1], c=[custom_cmap[i]], s=50, alpha=0.5)
    axes[2].scatter(new_centroids_large[i, 0], new_centroids_large[i, 1], c=[custom_cmap[i]], s=200, marker='^', edgecolor='black', linewidth=2)
    # Plot arrows showing movement of datapoints, making lines more faded
    for j in range(len(X[y_kmeans == i])):
        axes[2].arrow(X[y_kmeans == i][j, 0], X[y_kmeans == i][j, 1],
                      X_moved_large[y_kmeans == i][j, 0] - X[y_kmeans == i][j, 0],
                      X_moved_large[y_kmeans == i][j, 1] - X[y_kmeans == i][j, 1],
                      head_width=0.1, head_length=0.2, fc='lightgray', ec='lightgray', alpha=0.15)
axes[2].set_title('Data Points Moved by Δ$f_1$ and Δ$f_2$ of Global Centroids', fontsize=14)
axes[2].set_xlim(x_min_all, x_max_all)
axes[2].set_ylim(y_min_all, y_max_all)

# Set shared axis labels
fig.text(0.5, -0.02, '$f_1$', ha='center', fontsize=16)
fig.text(-0.02, 0.5, '$f_2$', va='center', rotation='vertical', fontsize=16)

# Add 'a', 'b', 'c' labels beneath the plots
axes[0].text(0.5, -0.15, 'a', transform=axes[0].transAxes, fontsize=20, ha='center')
axes[1].text(0.5, -0.15, 'b', transform=axes[1].transAxes, fontsize=20, ha='center')
axes[2].text(0.5, -0.15, 'c', transform=axes[2].transAxes, fontsize=20, ha='center')

plt.tight_layout()
plt.show()
