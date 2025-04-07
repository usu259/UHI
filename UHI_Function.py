import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from geneticalgorithm import geneticalgorithm as ga
from scipy.ndimage import gaussian_filter, binary_dilation
from sklearn.cluster import DBSCAN
import random



# ===========================
# 1. Parameters
# ===========================

max_possible_trees = 50

num_candidates = 50
tree_species = {
    0: {"base_price": 50, "cooling": 0.8, "decay": 0.35,"name":"tree_1","color":"green"},
    1: {"base_price": 70, "cooling": 0.62, "decay": 0.40,"name":"tree_2","color":"orange"},
    2: {"base_price": 90, "cooling": 0.73, "decay": 0.30,"name":"tree_3","color":"purple"},
    3: {"base_price": 110, "cooling": 0.9, "decay": 0.22,"name":"tree_4","color":"cyan"},
}

# cost factor for trees placed on roads
cost_factor = 1.5

budget_max = 1500

w_mean = 0.25   # Weight for average temperature
w_max  = 0.75   # Weight for maximum temperature (hot spot)


# ===========================
# 2. Data Loading & Preparation
# ===========================
def load_map(filename='sample_grid.csv'):
    df = pd.read_csv(filename)

    # Mapping of types to numerical values
    type_to_num = {'building': 0, 'road': 1, 'green_space': 2, 'water': 3}
    num_to_type = {v: k for k, v in type_to_num.items()}
    df['type_num'] = df['type'].map(type_to_num)

    # Creating pivot tables for types and temperatures
    type_pivot = df.pivot(index='y', columns='x', values='type_num')
    temp_pivot = df.pivot(index='y', columns='x', values='temp_fac')

    # Sort rows and columns by alignment
    type_pivot = type_pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)
    temp_pivot = temp_pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)

    # Conversion to NumPy array for quick access
    type_matrix = type_pivot.values
    temp_matrix = temp_pivot.values

    return type_matrix, temp_matrix, type_pivot, type_to_num, num_to_type

# ===========================
# 3. Cooling Effect Model
# ===========================
def apply_cooling(candidates, temp_matrix, tree_species):
    cooled = temp_matrix.copy()
    for (row, col, species) in candidates:
        mask = np.zeros_like(temp_matrix)
        mask[row, col] = 1
        mask = gaussian_filter(mask, sigma=tree_species[species]['decay'])
        cooled -= mask * 2  # Cooling multiplier
    return np.clip(cooled, 0, None)

# ===========================
# 4. Objective Function
# ===========================

def objective_function(candidates, type_matrix, temp_matrix, tree_species, type_to_num, budget_max,
                       w_mean=1.0, w_max=0.5, penalty_factor=1e6, reward_factor_budget=0.001, reward_factor_count=0.01,
                       cost_factor=1.5):
    """
    Evaluates a multi-criteria objective function for UHI optimization.

    Parameters:
    - candidates: List of tuples [(row, col, type)]
    - type_matrix: Terrain type matrix
    - temp_matrix: Initial temperature matrix
    - tree_species: Tree species dictionary
    - type_to_num: Terrain type dictionary
    - budget_max: Maximum allowed budget
    - w_mean, w_max: Weights for average and max temperature
    - penalty_factor: Penalty multiplier for constraint violation
    - reward_factor_budget, reward_factor_count: Reward factors
    - cost_factor: Cost multiplier for road placement

    Returns:
    - objective_value: Scalar score (lower is better)
    """

    penalty = 0
    total_cost = 0
    H, W = type_matrix.shape
    used_cells = set()

    # --- Evaluate constraints ---
    for (row, col, candidate_type) in candidates:
        # Check: valid position
        if row < 0 or row >= H or col < 0 or col >= W:
            penalty += penalty_factor
            continue

        # Check: building
        if type_matrix[row, col] == type_to_num['building']:
            penalty += penalty_factor
            continue

        # Check: no double placement
        if (row, col) in used_cells:
            penalty += penalty_factor
            continue
        used_cells.add((row, col))

        # Calculate cost
        factor = cost_factor if type_matrix[row, col] == type_to_num['road'] else 1.0
        total_cost += tree_species[candidate_type]['base_price'] * factor

    # Budget constraint
    if total_cost > budget_max:
        penalty += penalty_factor * (total_cost - budget_max)

    # --- Calculate temperature effect ---
    reduced_temp = calculate_reduced_heatmap(candidates, temp_matrix, tree_species)

    avg_temp, hotspot_temp = calculate_objective_stats(reduced_temp, temp_matrix)

    # --- Reward ---
    reward = - reward_factor_budget * total_cost - reward_factor_count * len(candidates)

    # --- Final objective ---
    objective_value = (w_mean * avg_temp + w_max * hotspot_temp + penalty + reward)

    return objective_value



## Visualization of cells type (road, building, green area, water)
def generate_terrain_map(type_pivot, ax=None, show=True):
    terrain_types = {
        0: ('black', 'Building'),
        1: ('tab:gray', 'Road'),
        2: ('tab:green', 'Green Space'),
        3: ('tab:blue', 'Water')
    }
    cmap_colors = [terrain_types[k][0] for k in range(4)]
    cmap = mcolors.ListedColormap(cmap_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    im = ax.imshow(type_pivot.values, origin='lower', cmap=cmap, norm=norm)
    ax.set_title("Terrain Type Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    legend_patches = [mpatches.Patch(color=terrain_types[k][0], label=terrain_types[k][1]) for k in range(4)]
    ax.legend(handles=legend_patches, loc='upper right')
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

def generate_heatmap(temp_matrix, vmin=None, vmax=None, ax=None, show=True):
    """
    Generates a temperature heatmap, returning the figure and axis for further modifications.

    Parameters:
    - temp_matrix: 2D numpy array of temperatures
    - vmin, vmax: Optional color scale limits
    - ax: Matplotlib axis object (optional), allows overlaying on an existing plot
    - show: If True, calls plt.show(). Set to False if additional layers are added.

    Returns:
    - fig, ax: Matplotlib figure and axis objects
    """
    mean_temp = np.mean(temp_matrix)
    max_temp = np.max(temp_matrix)

    if vmin is None:
        vmin = np.min(temp_matrix)
    if vmax is None:
        vmax = np.max(temp_matrix)

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Plot heatmap
    img = ax.imshow(temp_matrix, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_title("Heatmap - Temperature Condition")

    # Add colorbar if not overlaying on an existing axis
    if ax is None:
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Temperature")

    plt.tight_layout()

    # Show the plot if required
    if show:
        plt.show()

    return fig, ax  # Return figure and axis for further modifications

# Detect heat isands through clustering

def detect_heat_islands(temp_matrix, threshold=32):
    """
    Detects heat islands and their edges in a temperature matrix.

    Returns:
        cluster_grid: 2D grid with cluster labels (-1 = no cluster)
        edges: 2D boolean grid where True indicates edge of cluster
    """
    # Get grid dimensions
    H, W = temp_matrix.shape

    # Get all coordinates
    y_coords, x_coords = np.indices((H, W))
    coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))
    temps = temp_matrix.ravel()

    # Select hot regions
    hot_mask = temps >= threshold
    hot_coords = coords[hot_mask]

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=1.5, min_samples=3)
    hot_clusters = dbscan.fit_predict(hot_coords)

    # Create cluster grid
    cluster_grid = np.full((H, W), fill_value=-1)
    for (y, x), cluster_label in zip(hot_coords, hot_clusters):
        cluster_grid[y, x] = cluster_label

    # Identify cluster edges
    edges = np.zeros_like(cluster_grid, dtype=bool)
    for cluster_label in np.unique(hot_clusters[hot_clusters >= 0]):
        cluster_mask = cluster_grid == cluster_label
        dilated_mask = binary_dilation(cluster_mask, structure=np.ones((3, 3)))
        edge = dilated_mask & cluster_mask
        edges |= edge

    return cluster_grid, edges


# Generate heatmap with clusters of heat island

def generate_heat_island_map(reduced_temp, temp_matrix, vmin=None, vmax=None, ax=None, show=True, threshold=32.0, original=True):
    """
    Generates a heatmap with heat island contours.
    """
    if original:
        _, edges = detect_heat_islands(temp_matrix, threshold)
    else:
        _, edges = detect_heat_islands(reduced_temp, threshold)

    # Generate base heatmap
    fig, ax = generate_heatmap(reduced_temp, vmin, vmax, ax=ax, show=False)

    # Overlay temperature statistics
    avg, avg_hotspot = calculate_objective_stats(reduced_temp, temp_matrix)
    ax.text(0.02, 0.98, f"Mean: {avg:.2f}\nMeanHot: {avg_hotspot:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Overlay heat island edges
    ax.contour(edges, colors='yellow', linewidths=1.5, alpha=0.8)

    # Title and display logic
    ax.set_title("Heat Map with Hot Region Cluster Edges")
    ax.axis('off')

    if show:
        plt.show()

    return fig, ax

# Calculate new temperatures inside the grid

def calculate_reduced_heatmap(solution, temp_matrix, tree_species):
    final_temp = np.copy(temp_matrix)
    H, W = temp_matrix.shape
    Y, X = np.indices((H, W))

    for (row_center, col_center, candidate_type) in solution:
        sigma = 1 + (1 - tree_species[candidate_type]["decay"]) * 5
        dist2 = (Y - row_center)**2 + (X - col_center)**2
        effect = tree_species[candidate_type]['cooling'] * np.exp(-dist2 / (2 * sigma**2))
        final_temp -= effect

    return np.clip(final_temp, 0, None)

def calculate_objective_stats(heatmap, temp_matrix, threshold=32.0):
    cluster_grid, _ = detect_heat_islands(temp_matrix, threshold)
    hotspot_mask = cluster_grid >= 0

    avg_temp = np.mean(heatmap)
    mean_hotspot_temp = np.mean(heatmap[hotspot_mask]) if np.any(hotspot_mask) else avg_temp

    return avg_temp, mean_hotspot_temp

def calculate_used_budget(solution, type_matrix, tree_species, type_to_num, cost_factor=1.5):
    """
    Calculates total budget used, considering cost_factor for trees on roads.
    """
    total_cost = 0
    for row, col, species in solution:
        cell_factor = cost_factor if type_matrix[row, col] == type_to_num['road'] else 1.0
        cost = tree_species[species]['base_price'] * cell_factor
        total_cost += cost
    return total_cost

def print_solution_elements(solution, heading='Optimized Solution'):
  print(f"\n--- {heading} ---")
  print("Number of placements:", len(solution))
  for (row, col, candidate_type) in solution:
      print(f"Type: {tree_species[candidate_type]['name']} at Position: ({row}, {col})")


def print_solution_histogram(solution, temp_matrix, tree_species, bins=30):
    """
    Plot histogram comparing temperature distributions before and after cooling.

    Parameters:
    - original_map: 2D array (before cooling)
    - cooled_map: 2D array (after cooling)
    - bins: number of bins for the histogram
    """
    orig_flat = temp_matrix.flatten()
    cool_flat = calculate_reduced_heatmap(solution,temp_matrix, tree_species).flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(orig_flat, bins=bins, alpha=0.6, label='Original', color='red', edgecolor='black')
    plt.hist(cool_flat, bins=bins, alpha=0.6, label='Cooled', color='blue', edgecolor='black')

    plt.axvline(np.mean(orig_flat), color='red', linestyle='dashed', linewidth=1.5)
    plt.axvline(np.mean(cool_flat), color='blue', linestyle='dashed', linewidth=1.5)

    plt.title('Temperature Distribution Before and After Cooling')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_comparison_original_heatmap(solution, temp_matrix, tree_species, vmin=None, vmax=None, use_heat_islands=False, show=True):
    """
    Compares the initial temperature map vs. the optimized one side by side.

    Parameters:
    - solution: List of placed trees [(row, col, species)].
    - temp_matrix: 2D numpy array of initial temperatures.
    - tree_species: Dictionary with tree properties.
    - vmin, vmax: Optional color scale limits.
    - use_heat_islands: If True, overlays heat island edges on both maps.
    - show: If True, calls plt.show(). Set to False to modify further.

    Returns:
    - fig, axs: Matplotlib figure and axes objects.
    """

    # Compute optimized temperature map
    heatislands = detect_heat_islands(temp_matrix)[1]
    sol_temp = calculate_reduced_heatmap(solution, temp_matrix, tree_species)

    # Set consistent color scale
    if vmin is None:
        vmin = min(np.min(temp_matrix), np.min(sol_temp))
    if vmax is None:
        vmax = max(np.max(temp_matrix), np.max(sol_temp))

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot initial heatmap
    if use_heat_islands:
        generate_heat_island_map(temp_matrix, temp_matrix, vmin=vmin, vmax=vmax, show=False, ax=axs[0])
    else:
        generate_heatmap(temp_matrix, vmin, vmax, show=False, ax=axs[0])
    axs[0].set_title("Initial Temperature Map")

    # Plot optimized heatmap
    if use_heat_islands:
        generate_heat_island_map(sol_temp, temp_matrix, vmin=vmin, vmax=vmax, show=False, ax=axs[1])
    else:
        generate_heatmap(sol_temp, vmin, vmax, show=False, ax=axs[1])
    axs[1].set_title("Optimized Temperature Map")

    # Adjust layout
    plt.tight_layout()

    if show:
        plt.show()

    return fig, axs

def plot_comparison_solution_heatmap(solution1, solution2, temp_matrix, tree_species, vmin=None, vmax=None, use_heat_islands=False, show=True):
    """
    Compares two different solutions side by side.

    Parameters:
    - solution1: First list of placed trees [(row, col, species)].
    - solution2: Second list of placed trees [(row, col, species)].
    - temp_matrix: 2D numpy array of initial temperatures.
    - tree_species: Dictionary with tree properties.
    - vmin, vmax: Optional color scale limits.
    - use_heat_islands: If True, overlays heat island edges on both maps.
    - show: If True, calls plt.show(). Set to False to modify further.

    Returns:
    - fig, axs: Matplotlib figure and axes objects.
    """
    # Compute optimized temperature map
    heatislands = detect_heat_islands(temp_matrix)[1]
    sol_temp1 = calculate_reduced_heatmap(solution1, temp_matrix, tree_species)
    sol_temp2 = calculate_reduced_heatmap(solution2, temp_matrix, tree_species)

    # Set consistent color scale
    if vmin is None:
        vmin = min(np.min(sol_temp2), np.min(sol_temp1))
    if vmax is None:
        vmax = max(np.max(sol_temp2), np.max(sol_temp1))

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot solution maps
    if use_heat_islands:
        generate_heat_island_map(sol_temp1, temp_matrix, vmin=vmin, vmax=vmax, show=False, ax=axs[0])
        generate_heat_island_map(sol_temp2, temp_matrix, vmin=vmin, vmax=vmax, show=False, ax=axs[1])
    else:
        generate_heatmap(sol_temp1, vmin, vmax, show=False, ax=axs[0])
        generate_heatmap(sol_temp2, vmin, vmax, show=False, ax=axs[1])
    axs[0].set_title("Optimized Temperature Map 1")
    axs[1].set_title("Optimized Temperature Map 2")

    # Adjust layout
    plt.tight_layout()

    if show:
        plt.show()

    return fig, axs

def plot_solution_histogram_comparison(solution1, solution2, temp_matrix, tree_species, bins=30):
    """
    Plot histogram comparing temperature distributions for two solutions (before and after cooling).

    Parameters:
    - solution1: First solution (list of tree placements [(row, col, species)]).
    - solution2: Second solution (list of tree placements [(row, col, species)]).
    - temp_matrix: 2D array of original temperatures.
    - tree_species: Dictionary with tree species properties.
    - bins: Number of bins for the histogram.
    """
    # Calculate cooled maps for both solutions
    cool_flat_1 = calculate_reduced_heatmap(solution1, temp_matrix, tree_species).flatten()
    cool_flat_2 = calculate_reduced_heatmap(solution2, temp_matrix, tree_species).flatten()

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(cool_flat_1, bins=bins, alpha=0.6, label='Solution 1', color='blue', edgecolor='black')
    plt.hist(cool_flat_2, bins=bins, alpha=0.6, label='Solution 2', color='green', edgecolor='black')

    # Add mean lines
    plt.axvline(np.mean(cool_flat_1), color='blue', linestyle='dashed', linewidth=1.5)
    plt.axvline(np.mean(cool_flat_2), color='green', linestyle='dashed', linewidth=1.5)

    # Add labels and title
    plt.title('Temperature Distribution Comparison for Two Solutions')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_solution_heatmaps(solution1, solution2, temp_matrix, tree_species,
                              title1="Solution 1", title2="Solution 2",
                              use_heat_islands=False):
    """
    Compares temperature maps and histograms of two solutions.

    Parameters:
    - solution1: List of (row, col, species) or [] for initial map.
    - solution2: List of (row, col, species).
    - temp_matrix: Original temperature matrix.
    - tree_species: Dictionary of tree species properties.
    - title1: Title for the first map.
    - title2: Title for the second map.
    - use_heat_islands: If True, overlays heat island contours.
    """
    # Compute reduced maps
    temp1 = temp_matrix if solution1 == [] else calculate_reduced_heatmap(solution1, temp_matrix, tree_species)
    temp2 = calculate_reduced_heatmap(solution2, temp_matrix, tree_species)

    vmin = min(np.min(temp1), np.min(temp2))
    vmax = max(np.max(temp1), np.max(temp2))

    # === Heatmaps ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # First map
    generate_heatmap(temp1, vmin=vmin, vmax=vmax, show=False, ax=axs[0])
    avg1, hotspot1 = calculate_objective_stats(temp1, temp_matrix)
    axs[0].set_title(title1)
    axs[0].text(0.02, 0.98, f"Mean: {avg1:.2f}\nHotspot Mean: {hotspot1:.2f}",
                transform=axs[0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    for (row, col, species) in solution1:
        axs[0].add_patch(plt.Circle((col, row), radius=0.5,
                                    color=tree_species[species]['color'], alpha=0.8))

    # Second map
    generate_heatmap(temp2, vmin=vmin, vmax=vmax, show=False, ax=axs[1])
    avg2, hotspot2 = calculate_objective_stats(temp2, temp_matrix)
    axs[1].set_title(title2)
    axs[1].text(0.02, 0.98, f"Mean: {avg2:.2f}\nHotspot Mean: {hotspot2:.2f}",
                transform=axs[1].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    for (row, col, species) in solution2:
        axs[1].add_patch(plt.Circle((col, row), radius=0.5,
                                    color=tree_species[species]['color'], alpha=0.8))

    plt.tight_layout()
    plt.show()

    # === Histogram ===
    temp1_flat = temp1.flatten()
    temp2_flat = temp2.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(temp1_flat, bins=30, alpha=0.6, label=title1, color='red', edgecolor='black')
    plt.hist(temp2_flat, bins=30, alpha=0.6, label=title2, color='blue', edgecolor='black')

    plt.axvline(np.mean(temp1_flat), color='red', linestyle='dashed', linewidth=1.5)
    plt.axvline(np.mean(temp2_flat), color='blue', linestyle='dashed', linewidth=1.5)

    plt.title('Temperature Distribution Comparison')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def evaluate_multiple_runs(runs, type_matrix, temp_matrix, budget_max, type_to_num, tree_species):
    stats = [[], [], [], []]
    algorithms = ['Random Insertion', 'GA', 'ACO', 'ACO + Cand Ph']

    for run in range(runs):
        print(f"\n--- Run {run + 1} ---")

        # Random
        random_sol = random_insertion(type_matrix, budget_max, type_to_num, tree_species)
        budget = calculate_used_budget(random_sol, type_matrix, tree_species, type_to_num)
        avg, avg_hotspot = calculate_objective_stats(calculate_reduced_heatmap(random_sol, temp_matrix, tree_species), temp_matrix)
        stats[0].append((budget, avg, avg_hotspot))

        # GA
        model = run_ga(type_matrix, temp_matrix, budget_max, type_to_num, tree_species)
        ga_sol = [(int(model.output_dict['variable'][j]), int(model.output_dict['variable'][j+1]), int(model.output_dict['variable'][j+2]))
                  for j in range(0, len(model.output_dict['variable']), 4) if int(model.output_dict['variable'][j+3]) == 1]
        budget = calculate_used_budget(ga_sol, type_matrix, tree_species, type_to_num)
        avg, avg_hotspot = calculate_objective_stats(calculate_reduced_heatmap(ga_sol, temp_matrix, tree_species), temp_matrix)
        stats[1].append((budget, avg, avg_hotspot))

        # ACO
        aco_sol = run_aco(type_matrix, temp_matrix, type_to_num, tree_species, budget_max, iterations=50, num_ants=10)
        budget = calculate_used_budget(aco_sol, type_matrix, tree_species, type_to_num)
        avg, avg_hotspot = calculate_objective_stats(calculate_reduced_heatmap(aco_sol, temp_matrix, tree_species), temp_matrix)
        stats[2].append((budget, avg, avg_hotspot))

        # ACO + Candidate Pheromone
        aco_cand_sol = run_aco(type_matrix, temp_matrix, type_to_num, tree_species, budget_max, iterations=50, num_ants=10,
                               use_cand_pheromone=True)
        budget = calculate_used_budget(aco_cand_sol, type_matrix, tree_species, type_to_num)
        avg, avg_hotspot = calculate_objective_stats(calculate_reduced_heatmap(aco_cand_sol, temp_matrix, tree_species), temp_matrix)
        stats[3].append((budget, avg, avg_hotspot))

    # === Plotting (optional, same as before)

    return stats


def summarize_algorithm_runs(stats, algorithms):
    """
    Summarizes multiple run statistics of optimization algorithms.

    Parameters:
    - stats: List of lists with tuples (budget, avg_temp, hotspot_avg_temp) per run
    - algorithms: List of algorithm names (same order as stats)

    Outputs:
    - Prints summary table (Mean ± Std for each metric)
    - Plots bar charts, boxplots, and per-run line plots
    - Returns: summary DataFrame
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    summary_data = {'Algorithm': [], 'Mean Budget': [], 'Std Budget': [],
                    'Mean Avg Temp': [], 'Std Avg Temp': [],
                    'Mean Hotspot Temp': [], 'Std Hotspot Temp': []}

    for i, alg in enumerate(algorithms):
        budgets = [x[0] for x in stats[i]]
        avg_temps = [x[1] for x in stats[i]]
        hotspot_temps = [x[2] for x in stats[i]]

        summary_data['Algorithm'].append(alg)
        summary_data['Mean Budget'].append(np.mean(budgets))
        summary_data['Std Budget'].append(np.std(budgets))
        summary_data['Mean Avg Temp'].append(np.mean(avg_temps))
        summary_data['Std Avg Temp'].append(np.std(avg_temps))
        summary_data['Mean Hotspot Temp'].append(np.mean(hotspot_temps))
        summary_data['Std Hotspot Temp'].append(np.std(hotspot_temps))

    summary_df = pd.DataFrame(summary_data)
    print("\n=== Summary Table ===")
    print(summary_df.round(2))

    # === Plot Bar Charts ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].bar(summary_df['Algorithm'], summary_df['Mean Budget'],
               yerr=summary_df['Std Budget'], capsize=5)
    axs[0].set_title('Mean Budget Used')
    axs[0].set_ylabel('Budget')
    axs[0].grid(True, axis='y')

    axs[1].bar(summary_df['Algorithm'], summary_df['Mean Avg Temp'],
               yerr=summary_df['Std Avg Temp'], capsize=5)
    axs[1].set_title('Mean Average Temperature')
    axs[1].set_ylabel('Temperature (°C)')
    axs[1].grid(True, axis='y')

    axs[2].bar(summary_df['Algorithm'], summary_df['Mean Hotspot Temp'],
               yerr=summary_df['Std Hotspot Temp'], capsize=5)
    axs[2].set_title('Mean Hotspot Temperature')
    axs[2].set_ylabel('Temperature (°C)')
    axs[2].grid(True, axis='y')

    plt.tight_layout()
    plt.show()

    # === Plot Boxplots ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    budget_data = [[x[0] for x in stats[i]] for i in range(len(algorithms))]
    axs[0].boxplot(budget_data, labels=algorithms)
    axs[0].set_title('Budget Distribution')
    axs[0].set_ylabel('Budget')
    axs[0].grid(True, axis='y')

    avg_temp_data = [[x[1] for x in stats[i]] for i in range(len(algorithms))]
    axs[1].boxplot(avg_temp_data, labels=algorithms)
    axs[1].set_title('Average Temperature Distribution')
    axs[1].set_ylabel('Temperature (°C)')
    axs[1].grid(True, axis='y')

    hotspot_temp_data = [[x[2] for x in stats[i]] for i in range(len(algorithms))]
    axs[2].boxplot(hotspot_temp_data, labels=algorithms)
    axs[2].set_title('Hotspot Temperature Distribution')
    axs[2].set_ylabel('Temperature (°C)')
    axs[2].grid(True, axis='y')

    plt.tight_layout()
    plt.show()

    # === Plot Line Charts: Run-wise comparison ===
    runs = len(stats[0])
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))

    # Budget per run
    for i, algorithm in enumerate(algorithms):
        budgets = [x[0] for x in stats[i]]
        axs[0].plot(range(1, runs + 1), budgets, label=algorithm)
    axs[0].set_title('Budget Comparison Across Algorithms')
    axs[0].set_xlabel('Run Number')
    axs[0].set_ylabel('Budget Used')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # Avg Temp per run
    for i, algorithm in enumerate(algorithms):
        avg_temps = [x[1] for x in stats[i]]
        axs[1].plot(range(1, runs + 1), avg_temps, label=algorithm)
    axs[1].set_title('Average Temperature Comparison Across Algorithms')
    axs[1].set_xlabel('Run Number')
    axs[1].set_ylabel('Average Temperature')
    axs[1].legend(loc='upper left')
    axs[1].grid(True)

    # Hotspot Temp per run
    for i, algorithm in enumerate(algorithms):
        hotspot_avg = [x[2] for x in stats[i]]
        axs[2].plot(range(1, runs + 1), hotspot_avg, label=algorithm)
    axs[2].set_title('Hotspot Average Temperature Comparison Across Algorithms')
    axs[2].set_xlabel('Run Number')
    axs[2].set_ylabel('Hotspot Avg Temperature')
    axs[2].legend(loc='upper left')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return summary_df



# ===========================
# 6. Optimization - Random insertion
# ===========================

def random_insertion(type_matrix, budget_max, type_to_num, tree_species, cost_factor=1.5):
    H, W = type_matrix.shape
    placed_elements = []
    total_cost = 0
    min_tree_cost = min(tree_species[t]['base_price'] for t in tree_species)
    used_cells = set()
    max_iterations = H * W * 10
    iterations = 0

    while total_cost + min_tree_cost <= budget_max and iterations < max_iterations:
        iterations += 1
        row = random.randint(0, H - 1)
        col = random.randint(0, W - 1)

        if type_matrix[row, col] == type_to_num['building'] or (row, col) in used_cells:
            continue

        species = random.choice(list(tree_species.keys()))
        cell_factor = cost_factor if type_matrix[row, col] == type_to_num['road'] else 1.0
        cost = tree_species[species]['base_price'] * cell_factor

        if total_cost + cost > budget_max:
            continue

        placed_elements.append((row, col, species))
        used_cells.add((row, col))
        total_cost += cost

    return placed_elements


