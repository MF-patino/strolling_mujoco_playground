
import matplotlib, os
try:
    matplotlib.use('TkAgg') 
except:
    print("Warning: Running in headless environment.")
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import numpy as np
import jax.numpy as jp
from matplotlib.patches import Patch
from sklearn.decomposition import TruncatedSVD
from matplotlib.lines import Line2D
import umap

PLOT_DATA_DIR = "plotData"
TRAIN_DATA_SUBDIR = "training"

def transferLearningComparison(env_name, pol_name, base_pol_name):
    plot_dir = os.path.join(PLOT_DATA_DIR, TRAIN_DATA_SUBDIR)
    transfer_data = np.load(os.path.join(plot_dir, f"{pol_name}.npz"))
    scratch_data = np.load(os.path.join(plot_dir, f"{base_pol_name}.npz"))
    
    limit_evals = 75
    steps = transfer_data['steps'][:limit_evals]
    means = transfer_data['reward_mean'][:limit_evals]
    stds = transfer_data['reward_std'][:limit_evals]
    plt.figure(figsize=(8, 5))

    # Plot Transfer Learning
    plt.plot(steps, means, label="Fine-tuned from Flat", color='blue')
    plt.fill_between(steps, means - stds, means + stds, alpha=0.2, color='blue')

    # Plot From Scratch
    steps = scratch_data['steps'][:limit_evals]
    means = scratch_data['reward_mean'][:limit_evals]
    stds = scratch_data['reward_std'][:limit_evals]
    plt.plot(steps, means, label="Trained from Scratch", color='red')
    plt.fill_between(steps, means - stds, means + stds, alpha=0.2, color='red')

    plt.title(f"{env_name}: Transfer Learning vs. Training from Scratch")
    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Episode Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

def policyEmbeddings2D(controller):

    print("Generating 2D Latent Space Plot...")

    # 1. Reduce the raw inaffinity matrix strictly to 2D
    reducer_2d = TruncatedSVD(n_components=2)
    raw_mat_np = np.array(controller.inaffinity_matrix)
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
    colors = cm.get_cmap('tab10', len(controller.pol_names))

    # 4. Scatter and Annotate each policy
    for i, env_name in enumerate(controller.pol_names):
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

def policyEmbeddings3D(controller):
    print("Generating 3D Latent Space Sphere Plot...")

    # Reduce the raw inaffinity matrix strictly to 3D
    # (Make sure we have at least 3 policies to do a 3D projection)
    n_components = min(3, controller.inaffinity_matrix.shape[1])
    reducer_3d = TruncatedSVD(n_components=n_components)
    raw_mat_np = np.array(controller.inaffinity_matrix)
    coords_3d = reducer_3d.fit_transform(raw_mat_np)
    variance_ratio = sum(reducer_3d.explained_variance_ratio_) * 100
    print(f"SVD preserved {variance_ratio:.2f}% of the variance.")

    # Pad with zeros if we somehow have fewer than 3 dimensions
    if coords_3d.shape[1] < 3:
        coords_3d = np.pad(coords_3d, ((0, 0), (0, 3 - coords_3d.shape[1])), mode='constant')

    # L2 Normalize the 3D coordinates so they lie exactly on a 3D Unit Sphere
    norms = np.linalg.norm(coords_3d, axis=1, keepdims=True)
    coords_3d_norm = coords_3d / (norms + 1e-8)

    # Figure Setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get a distinct colormap for the policies
    # Extract unique base domains (target domains) from the policy names
    base_names = [name.split("_AdaptedFrom_")[0] for name in controller.pol_names]
    unique_base_names = list(set(base_names))
    cmap = cm.get_cmap('tab10', len(unique_base_names))
    domain_colors = {domain: cmap(i) for i, domain in enumerate(unique_base_names)}

    # Draw the translucent 3D Unit Sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot surface and wireframe for depth perception
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='whitesmoke', alpha=0.15, edgecolor='none')
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1, linewidth=0.5)

    # Scatter and Annotate each policy
    for i, env_name in enumerate(controller.pol_names):
        x, y, z = coords_3d_norm[i, 0], coords_3d_norm[i, 1], coords_3d_norm[i, 2]
        base_name = env_name.split("_AdaptedFrom_")[0]
        c = domain_colors[base_name]
        
        # Draw the point on the sphere surface
        ax.scatter(x, y, z, color=c, s=150, edgecolor='black', depthshade=True, label=env_name)
        
        # Draw a faint line from the origin to the point (shows the vector)
        ax.plot([0, x], [0, y], [0, z], color=c, linestyle='--', alpha=0.6, linewidth=1.5)

    # Formatting
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

def statisticDriftHistory(controller):
    plt.plot(controller.detector.stat_values, label="KS statistic")

    plt.vlines(controller.drift_indices,
        ymin=min(controller.detector.stat_values),
        ymax=max(controller.detector.stat_values),
        color="red", alpha=0.6, label='Drift detection')

    plt.xlabel("Time step")
    plt.title("KS-ADWIN concept drift detector history")
    plt.legend()
    plt.tight_layout()
    plt.show()

def wmErrorHistory(controller, env_change = None):
    if env_change is None:
        env_change = controller.env_changes[-1]

    extra_steps = 250
    last_env_change, env_name = env_change
    start_step = max(last_env_change - extra_steps, 0)
    end_step = last_env_change + extra_steps

    i = controller.env_changes.index(env_change)
    prev_env = 'None' if i == 0 else controller.env_changes[i-1][1]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axvline(x=last_env_change - start_step, color='green', linestyle='--', linewidth=2, zorder=5)

    legend_elements = []
    for wm_name in controller.smooth_errors:
        legend_elements.append(ax.plot(controller.smooth_errors[wm_name][start_step:end_step], label=f"{wm_name} WM errors")[0])

    legend_elements.append(Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f'Change: {prev_env} -> {env_name}'))
    
    # Place legend outside the plot
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), title="Legend")
    plt.xlabel("Time step")
    plt.title("WM error history")
    plt.tight_layout()
    plt.show()

def plotGaitPattern(controller, env_change = None):
    if env_change is None:
        env_change = controller.env_changes[-1]

    extra_steps = 250
    last_env_change, env_name = env_change
    start_step = max(last_env_change - extra_steps, 0)
    end_step = last_env_change + extra_steps

    print("Generating Gait Pattern Plot...")

    # Convert list of boolean arrays to a 2D numpy array (Time x 4)
    contacts = np.array(controller.contact_history)
    
    # The Y-axis will go from 0 (bottom) to 3 (top)
    feet_names = ["RR", "RL", "FR", "FL"] 
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # 1. Plot the foot contacts
    for i in range(len(feet_names)):
        # Invert the index so FR (idx 0) is at the top (y=3)
        y_level = 3 - i 
        contact_bools = contacts[start_step:end_step, i]
        
        # Find the start and end indices of continuous contact segments
        # Padding with False ensures we catch segments that start at 0 or touch the end
        padded = np.pad(contact_bools, (1, 1), mode='constant', constant_values=False)
        diffs = np.diff(padded.astype(int))
        
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        # Create a list of (start_x, width) tuples
        xranges =[(start, end - start) for start, end in zip(starts, ends)]
        
        # Plot solid rectangles. (y_level - 0.2, 0.4) means the bar is centered 
        # on y_level and has a thickness of 0.4.
        ax.broken_barh(xranges, (y_level - 0.2, 0.4), facecolors='black', zorder=4)

    # Get a distinct colormap for the policies
    # Extract unique base domains (target domains) from the policy names
    base_names = [name.split("_AdaptedFrom_")[0] for name in controller.pol_names]
    unique_base_names = list(set(base_names))
    cmap = cm.get_cmap('tab10', len(unique_base_names))
    domain_colors = {domain: cmap(i) for i, domain in enumerate(unique_base_names)}
    
    start_idx = 0
    recent_policy_history = controller.policy_history[start_step:end_step]
    current_pol = recent_policy_history[0]
    
    for t in range(1, len(recent_policy_history)):
        # If the policy changed, or we reached the end of the simulation
        if recent_policy_history[t] != current_pol or t == len(recent_policy_history) - 1:
            c = domain_colors[current_pol.split("_AdaptedFrom_")[0]]
            # Paint the background for that duration
            ax.axvspan(start_idx, t, facecolor=c, alpha=0.6)
            
            start_idx = t
            current_pol = recent_policy_history[t]

    i = controller.env_changes.index(env_change)
    prev_env = 'None' if i == 0 else controller.env_changes[i-1][1]
    ax.axvline(x=last_env_change - start_step, color='green', linestyle='--', linewidth=2, zorder=5)
    # 3. Draw vertical line for drift detection
    drifts = [idx-start_step for idx in controller.drift_indices if idx > start_step and idx < end_step]
    for drift in drifts:
        ax.axvline(x=drift, color='red', linestyle='--', linewidth=2, zorder=5)

    # 4. Formatting
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(feet_names, fontweight='bold')
    ax.set_xlabel("Time step (50Hz)", fontsize=12)
    ax.set_title(f"Gait Contact Pattern During Online Adaptation: {prev_env} -> {env_name}", fontsize=14)
    
    # Create a clean legend for the background colors
    recent_pols = list(set(recent_policy_history))
    legend_elements =[Patch(facecolor=domain_colors[pol.split("_AdaptedFrom_")[0]], alpha=0.6, label=pol) for pol in recent_pols]
    
    # Add a red dashed line to the legend for the drift detector
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Drift Detected'))
    legend_elements.append(Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f'Change: {prev_env} -> {env_name}'))
    
    # Place legend outside the plot so it doesn't cover the gait bars
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), title="Legend")

    plt.tight_layout()
    plt.show()


def plotGPSearch(controller, gp_states=None):
    import warnings

    # Get the sequence of iterations for the most recent drift
    if gp_states is None:
        gp_states = controller.gp_states[-1]

    num_iterations = len(gp_states)
    
    # 1. Collapse the 4D embeddings to a fixed 1D axis using UMAP
    # UMAP preserves both local clustering (redundant policies) AND global distances (domains)
    pol_names_list = list(controller.policy_embeddings.keys())
    emb_matrix = np.array([controller.policy_embeddings[name] for name in pol_names_list])
    
    # n_neighbors dictates how much local vs global structure to preserve. 
    # For a small catalog, a small number (e.g., 3 to 5) works perfectly.
    n_neighbors = min(5, len(pol_names_list) - 1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_components=1, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        coords_1d = reducer.fit_transform(emb_matrix).flatten()
    
    x_coords = {name: coord for name, coord in zip(pol_names_list, coords_1d)}
    
    # Extract unique domains and create a Colormap
    base_names = [name.split("_AdaptedFrom_")[0] for name in pol_names_list]
    unique_base_names = list(set(base_names))
    cmap = cm.get_cmap('tab10', len(unique_base_names))
    domain_colors = {domain: cmap(i) for i, domain in enumerate(unique_base_names)}
    
    # Create a subplot grid: N rows, 1 column. 
    fig, axes = plt.subplots(num_iterations, 1, figsize=(12, 3.5 * num_iterations), 
                             sharex=True, layout="constrained")
    
    if num_iterations == 1: axes = [axes]
    else: axes = axes.flatten()

    for ax_idx, (iteration, base_policy_name, chosen_policy_name, polInfo) in enumerate(gp_states):
        ax = axes[ax_idx]

        xs, means, stds, names = [], [], [], []
        colors, markers, mecs, mews, sizes, zorders = [], [],[], [], [],[]

        # 2. Query GP beliefs for all policies in catalog
        for pol_name in pol_names_list:
            x_val = x_coords[pol_name]
            
            # Find the belief stored in this specific iteration's polInfo
            mean, std =[(m, s) for _, m, s, name in polInfo if pol_name == name][0]
            
            xs.append(float(x_val))
            means.append(float(mean[0]))
            stds.append(float(std[0]))
            names.append(pol_name)
            
            # Color based on Target Domain
            base_domain = pol_name.split("_AdaptedFrom_")[0]
            colors.append(domain_colors[base_domain])
            
            # Shape/Border based on GP Status
            if pol_name == base_policy_name:
                markers.append('D') # Diamond
                mecs.append('black'); mews.append(2.5); sizes.append(100); zorders.append(5)
            elif pol_name == chosen_policy_name:
                markers.append('*') # Star
                mecs.append('red'); mews.append(2.0); sizes.append(300); zorders.append(6)
            else:
                markers.append('o') # Circle
                mecs.append('gray'); mews.append(1.0); sizes.append(60); zorders.append(3)

        # Sort everything by the 1D X-coordinate
        sorted_indices = np.argsort(xs)
        xs, means, stds = np.array(xs)[sorted_indices], np.array(means)[sorted_indices], np.array(stds)[sorted_indices]
        names = [names[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        markers = [markers[i] for i in sorted_indices]
        mecs = [mecs[i] for i in sorted_indices]
        mews = [mews[i] for i in sorted_indices]
        sizes =[sizes[i] for i in sorted_indices]
        zorders = [zorders[i] for i in sorted_indices]

        # 3. Plot the GP's uncertainty bound (Mean ± Std)
        ax.plot(xs, means, color='blue', alpha=0.4, linestyle='--', zorder=1)
        ax.fill_between(xs, means - stds, means + stds, color='blue', alpha=0.1, zorder=0)

        # Scatter the policies individually so we can apply specific markers
        for i in range(len(xs)):
            ax.errorbar(xs[i], means[i], yerr=stds[i], fmt=markers[i], 
                        ecolor='gray', elinewidth=1.5, capsize=3, # Error bar style
                        markerfacecolor=colors[i], markeredgecolor=mecs[i], 
                        markeredgewidth=mews[i], markersize=np.sqrt(sizes[i]), zorder=zorders[i])
            
            # Annotate only the Base and Chosen policies to prevent massive text overlap
            if markers[i] in ['D', '*']:
                # FIX: Push labels strictly UP using fixed screen points (staggered 25 or 45 points high)
                y_offset = 10 if i % 2 == 0 else 45
                
                ax.annotate(names[i], (xs[i], means[i]), xytext=(0, y_offset), 
                            textcoords='offset points', ha='center', fontsize=8, fontweight='bold',
                            # FIX: Set alpha to 0.4 so the box is highly transparent
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=mecs[i], alpha=0.4), zorder=10)

        # Formatting
        ax.set_title(f"Iteration {iteration}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add a little extra top margin so the labels don't get clipped by the subplot boundary
        ax.margins(y=0.2)

    axes[-1].set_xlabel("1D UMAP Projection of Latent Policy Space", fontsize=12)
    fig.supylabel("Predicted Error (GP Mean)", fontsize=14)

    # --- CUSTOM LEGENDS ---
    # Legend 1: Domains (Colors)
    domain_lines =[Line2D([0], [0], marker='o', color='w', markerfacecolor=domain_colors[d], 
                           markersize=10, markeredgecolor='gray', label=d) for d in unique_base_names]
    
    # Legend 2: Status (Shapes)
    status_lines = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='black', markeredgewidth=2.5, markersize=8, label='Base Policy'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white', markeredgecolor='red', markeredgewidth=2, markersize=14, label='Chosen (UCB)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='gray', markeredgewidth=1, markersize=8, label='Other')
    ]
    
    blank_line = Line2D([0], [0], color='w', label=' ')

    axes[0].legend(handles=domain_lines + [blank_line] + status_lines, 
                   loc='upper left', bbox_to_anchor=(1.02, 1.05), fontsize=9, framealpha=0.9)

    plt.show()