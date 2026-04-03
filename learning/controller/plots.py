
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import TruncatedSVD

def policyEmbeddings2D(self):

    print("Generating 2D Latent Space Plot...")

    # 1. Reduce the raw inaffinity matrix strictly to 2D
    reducer_2d = TruncatedSVD(n_components=2)
    raw_mat_np = np.array(self.inaffinity_matrix)
    coords_2d = reducer_2d.fit_transform(raw_mat_np)
    variance_ratio = sum(reducer_2d.explained_variance_ratio_) * 100
    print(f"SVD preserved {variance_ratio:.2f}% of the variance.")

    # 2. L2 Normalize the 2D coordinates so they lie on a unit circle
    # This makes the plot perfectly reflect the Cosine Distances used by the GP
    norms = np.linalg.norm(coords_2d, axis=1, keepdims=True)
    coords_2d_norm = coords_2d / norms

    # 3. Plotting Setup
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get a distinct colormap for the policies
    colors = cm.get_cmap('tab10', len(self.env_names))

    # 4. Scatter and Annotate each policy
    for i, env_name in enumerate(self.env_names):
        x, y = coords_2d_norm[i, 0], coords_2d_norm[i, 1]
        
        # Draw the point
        ax.scatter(x, y, color=colors(i), s=150, edgecolor='black', zorder=3, label=env_name)
        
        # Add the text label slightly offset from the point
        ax.annotate(env_name, (x, y), xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # 5. Draw the Unit Circle (represents Cosine Distance = 1.0 boundary)
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', alpha=0.5, zorder=1)
    ax.add_patch(circle)

    # 6. Formatting
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, zorder=1)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, zorder=1)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    
    # Force the aspect ratio to be perfectly square so the circle isn't warped
    ax.set_aspect('equal', adjustable='box')
    
    # Add some padding around the circle so labels fit
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)

    plt.title("Policy Embeddings (2D SVD Projection)", fontsize=14, pad=15)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    
    # Move legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Policies")
    plt.tight_layout()
    
    plt.show()

def policyEmbeddings3D(self):
    print("Generating 3D Latent Space Sphere Plot...")

    # 1. Reduce the raw inaffinity matrix strictly to 3D
    # (Make sure we have at least 3 policies to do a 3D projection)
    n_components = min(3, self.inaffinity_matrix.shape[1])
    reducer_3d = TruncatedSVD(n_components=n_components)
    raw_mat_np = np.array(self.inaffinity_matrix)
    coords_3d = reducer_3d.fit_transform(raw_mat_np)
    variance_ratio = sum(reducer_3d.explained_variance_ratio_) * 100
    print(f"SVD preserved {variance_ratio:.2f}% of the variance.")

    # Pad with zeros if we somehow have fewer than 3 dimensions
    if coords_3d.shape[1] < 3:
        coords_3d = np.pad(coords_3d, ((0, 0), (0, 3 - coords_3d.shape[1])), mode='constant')

    # 2. L2 Normalize the 3D coordinates so they lie exactly on a 3D Unit Sphere
    norms = np.linalg.norm(coords_3d, axis=1, keepdims=True)
    coords_3d_norm = coords_3d / (norms + 1e-8)

    # 3. Figure Setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get a distinct colormap for the policies
    colors = cm.get_cmap('tab10', len(self.env_names))

    # 4. Draw the translucent 3D Unit Sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot surface and wireframe for depth perception
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='whitesmoke', alpha=0.15, edgecolor='none')
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1, linewidth=0.5)

    # 5. Scatter and Annotate each policy
    for i, env_name in enumerate(self.env_names):
        x, y, z = coords_3d_norm[i, 0], coords_3d_norm[i, 1], coords_3d_norm[i, 2]
        
        # Draw the point on the sphere surface
        ax.scatter(x, y, z, color=colors(i), s=150, edgecolor='black', depthshade=True, label=env_name)
        
        # Draw a faint line from the origin to the point (shows the vector)
        ax.plot([0, x], [0, y], [0, z], color=colors(i), linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Add the text label, pushed slightly outward (1.1x) from the sphere surface so it doesn't clip
        ax.text(x * 1.1, y * 1.1, z * 1.1, env_name, 
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # 6. Formatting
    # Draw origin axes
    ax.plot([-1.2, 1.2],[0, 0], [0, 0], color='gray', linestyle='-', linewidth=0.5)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='gray', linestyle='-', linewidth=0.5)
    ax.plot([0, 0],[0, 0], [-1.2, 1.2], color='gray', linestyle='-', linewidth=0.5)

    # Force perfectly cubic proportions so the sphere isn't squashed into an ellipsoid
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])

    ax.set_xlabel("Principal Component 1", labelpad=10)
    ax.set_ylabel("Principal Component 2", labelpad=10)
    ax.set_zlabel("Principal Component 3", labelpad=10)
    plt.title("Latent Policy Space (3D SVD Projection on Unit Sphere)", fontsize=14, pad=20)
    
    # Legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Policies")
    plt.tight_layout()
    
    plt.show()