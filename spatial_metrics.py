import numpy as np
from scipy.spatial import cKDTree

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.stats import entropy as shannon_entropy
from scipy.spatial import distance
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay, ConvexHull
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import pandas as pd
import squidpy as sq
import scanpy as sc
from anndata import AnnData





# Spatial statistics functions




import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc
from anndata import AnnData


def calculate_ripley_l(coords, cell_types):
    """
    Calculate Ripley's L function for spatial point patterns for a specific cell type.
    
    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape (n_cells, 2) containing x,y coordinates of cells
    cell_types : numpy.ndarray
        Array of shape (n_cells,) containing cell type labels
    dist : float, default=200
        Maximum distance up to which to compute the Ripley's statistics
    n_steps : int, default=50
        Number of steps for distances along which to calculate Ripley's statistics
    cancer_type : int, default=1
        Cell type to focus analysis on (only these cells will be analyzed)
        
    Returns
    -------
    dict
        Dictionary with 'r_values' key for radii used, 'k_values' for K function values,
        and 'l_values' for the L function (transformed K function)
    """

    filtered_coords = coords
    # Create AnnData object with only the filtered cells
    adata = AnnData(X=np.zeros((filtered_coords.shape[0], 1)))
    adata.obsm['spatial'] = np.asarray(filtered_coords)
    cell_type_labels = np.array(['stroma', 'cancer'])
    adata.obs['cell_type'] = pd.Categorical(cell_type_labels[cell_types.astype(int)])

    # Calculate Ripley's K
    sq.gr.ripley(
        adata,
        cluster_key="cell_type",
        n_neigh=20,
        max_dist=100,
        n_steps=6,
        mode='L',
        
    )
    
    # Get results
    results = {}
    
    # Extract distance values and K values

    # Store the full F-statistic dataframe
    f_stats_df = adata.uns["cell_type_ripley_L"]["L_stat"]
    
    # Separate stats by cell type for easier analysis
    cancer_stats = f_stats_df[f_stats_df['cell_type'] == 'cancer']
    stroma_stats = f_stats_df[f_stats_df['cell_type'] == 'stroma']



    # Create dictionary entries for each bin's F-statistic values for cancer cells
    results.update({f'cancer_ripley_L_{bin}': stat for bin, stat in zip(cancer_stats["bins"], cancer_stats["stats"])})

    # Create dictionary entries for each bin's F-statistic values for stroma cells
    results.update({f'stroma_ripley_L_{bin}': stat for bin, stat in zip(stroma_stats["bins"], stroma_stats["stats"])})
    
    return results
    











def calculate_bidirectional_min_distance(coords, cell_types):
    """
    Calculate average minimum distance from stroma cells to tumor cells and vice versa.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Array of shape (n, 2) containing x, y coordinates for each cell
    cell_types : numpy.ndarray
        Array of shape (n,) containing cell type labels (0 for stroma, 1 for tumor)
        
    Returns:
    --------
    dict
        Dictionary containing bidirectional distance metrics:
        - 'stroma_to_tumor_mean_dist': Average minimum distance from stroma cells to tumor cells
        - 'tumor_to_stroma_mean_dist': Average minimum distance from tumor cells to stroma cells
        - 'stroma_to_tumor_median_dist': Median minimum distance from stroma cells to tumor cells
        - 'tumor_to_stroma_median_dist': Median minimum distance from tumor cells to stroma cells
    """
    # Get indices for tumor and stroma cells
    tumor_indices = np.where(cell_types == 1)[0]
    stroma_indices = np.where(cell_types == 0)[0]
    
    # Handle edge cases
    if len(tumor_indices) == 0 or len(stroma_indices) == 0:
        return {
            'stroma_to_tumor_mean_dist': 0.0,
            'tumor_to_stroma_mean_dist': 0.0,
        }
    
    # Extract coordinates for tumor and stroma cells
    tumor_coords = coords[tumor_indices]
    stroma_coords = coords[stroma_indices]
    
    # Use KDTree for efficient nearest neighbor search
    from scipy.spatial import cKDTree
    
    # Build KDTrees for both tumor and stroma cells
    tumor_tree = cKDTree(tumor_coords)
    stroma_tree = cKDTree(stroma_coords)
    
    # Find distance from each stroma cell to nearest tumor cell
    stroma_to_tumor_dists, _ = tumor_tree.query(stroma_coords, k=1)
    
    # Find distance from each tumor cell to nearest stroma cell
    tumor_to_stroma_dists, _ = stroma_tree.query(tumor_coords, k=1)
    
    # Calculate statistics
    return {
        'stroma_to_tumor_mean_dist': np.mean(stroma_to_tumor_dists),
        'tumor_to_stroma_mean_dist': np.mean(tumor_to_stroma_dists),
    }

def calculate_newmans_assortativity(coords, cell_types, radius=50):
    if len(coords) < 3:
        return 0.0

    cell_types = np.asarray(cell_types, dtype=int)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(radius, output_type='ndarray')

    if len(pairs) == 0:
        return 0.0

    types_i = cell_types[pairs[:, 0]]
    types_j = cell_types[pairs[:, 1]]

    same_type_edges = np.sum(types_i == types_j)
    total_edges = len(pairs)

    unique_types, type_counts = np.unique(cell_types, return_counts=True)
    type_fractions = type_counts / len(cell_types)
    expected_same_type = np.sum(type_fractions ** 2)
    actual_same_type = same_type_edges / total_edges

    if abs(1 - expected_same_type) < 1e-8:
        return 0.0

    assortativity = (actual_same_type - expected_same_type) / (1 - expected_same_type)
    return float(assortativity) if not np.isnan(assortativity) else 0.0



def calculate_centrality_scores(coords, cell_types, radius=50, filter_type=1):
    
    ##Calculate centrality metrics without using NetworkX.
    ##Uses KDTree for spatial queries and direct calculations for centrality metrics.
    
    if len(coords) < 3:
        return {
            'degree_centrality_ratio': 0.0,
            'betweenness_centrality_ratio': 0.0,
            'closeness_centrality_ratio': 0.0
        }
    
    try:
        # Create KDTree for efficient spatial queries
        tree = cKDTree(coords)
        
        # For each point, find all neighbors within radius
        # query_ball_point returns a list of indices for each point
        neighbors_list = tree.query_ball_point(coords, radius)
        
        # 1. Degree Centrality: just count the number of neighbors (excluding self)
        degree_values = np.array([len(neighbors) - 1 for neighbors in neighbors_list])  # -1 to exclude self
        if len(coords) > 1:  # Normalize by (n-1)
            degree_values = degree_values / (len(coords) - 1)
        
        # 2. Approximation for Betweenness Centrality
        # Instead of computing all shortest paths, use a local approximation:
        # For each point, calculate how many pairs of its neighbors are NOT connected
        # directly to each other (i.e., this point is a potential "bridge")
        betweenness_values = np.zeros(len(coords))
        
        for i in range(len(coords)):
            if len(neighbors_list[i]) <= 2:  # Need at least 2 neighbors to be a bridge
                continue
                
            # Get all neighbors (excluding self)
            neighbors = [n for n in neighbors_list[i] if n != i]
            
            # For each pair of neighbors, check if they're directly connected
            bridge_count = 0
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    # If k is not in j's neighbor list, i is a bridge between them
                    if neighbors[k] not in neighbors_list[neighbors[j]]:
                        bridge_count += 1
            
            # Normalize by total possible pairs
            total_pairs = (len(neighbors) * (len(neighbors) - 1)) / 2
            if total_pairs > 0:
                betweenness_values[i] = bridge_count / total_pairs
        
        # 3. Approximation for Closeness Centrality
        # Use inverse of mean distance to all points within a larger search radius
        larger_radius = radius * 2  # Use a larger radius for closeness calculation
        closeness_values = np.zeros(len(coords))
        
        for i in range(len(coords)):
            # Find all points within the larger radius
            extended_neighbors = tree.query_ball_point(coords[i], larger_radius)
            if len(extended_neighbors) <= 1:  # Skip if only self or empty
                continue
                
            # Calculate distances to all these neighbors
            distances = np.sqrt(np.sum((coords[extended_neighbors] - coords[i])**2, axis=1))
            
            # Closeness is inverse of mean distance (excluding self)
            non_self_distances = distances[extended_neighbors != i]
            if len(non_self_distances) > 0:
                closeness_values[i] = 1.0 / np.mean(non_self_distances)
        
        # Normalize closeness to [0,1] range
        if np.max(closeness_values) > 0:
            closeness_values = closeness_values / np.max(closeness_values)
        
        # Group by cell type and calculate ratios
        cancer_mask = cell_types == 1
        non_cancer_mask = ~cancer_mask
        
        # Calculate mean values for each cell type
        cancer_degree_mean = np.mean(degree_values[cancer_mask]) if np.any(cancer_mask) else 0.0
        non_cancer_degree_mean = np.mean(degree_values[non_cancer_mask]) if np.any(non_cancer_mask) else 0.0
        
        cancer_betweenness_mean = np.mean(betweenness_values[cancer_mask]) if np.any(cancer_mask) else 0.0
        non_cancer_betweenness_mean = np.mean(betweenness_values[non_cancer_mask]) if np.any(non_cancer_mask) else 0.0
        
        cancer_closeness_mean = np.mean(closeness_values[cancer_mask]) if np.any(cancer_mask) else 0.0
        non_cancer_closeness_mean = np.mean(closeness_values[non_cancer_mask]) if np.any(non_cancer_mask) else 0.0
        
        # Calculate ratios (avoiding division by zero)
        degree_ratio = (cancer_degree_mean / non_cancer_degree_mean 
                         if non_cancer_degree_mean > 0 else 0.0)
        betweenness_ratio = (cancer_betweenness_mean / non_cancer_betweenness_mean 
                              if non_cancer_betweenness_mean > 0 else 0.0)
        closeness_ratio = (cancer_closeness_mean / non_cancer_closeness_mean 
                            if non_cancer_closeness_mean > 0 else 0.0)
        
        return {
            'degree_centrality_ratio': degree_ratio,
            'betweenness_centrality_ratio': betweenness_ratio,
            'closeness_centrality_ratio': closeness_ratio
        }
    except Exception as e:
        return {
            'degree_centrality_ratio': 0.0,
            'betweenness_centrality_ratio': 0.0,
            'closeness_centrality_ratio': 0.0
        }
    



def calculate_cluster_cooccurrence_ratio(coords, cell_types, k=5, min_cluster_size=3):
    #Optimized version using DBSCAN-inspired clustering and KDTree for adjacency
    if len(coords) < max(k+1, min_cluster_size*2):
        return {
            'cluster_cooccurrence_ratio': 0.0,
            'mixed_cluster_proportion': 0.0
        }
    
    try:
        # Use NearestNeighbors once for all points
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        
        # Use a more efficient DBSCAN-like clustering approach
        from sklearn.cluster import DBSCAN
        
        # Separate cancer and non-cancer points
        cancer_coords = coords[cell_types == 1]
        non_cancer_coords = coords[cell_types == 0]
        
        # Only proceed if we have enough points of each type
        if len(cancer_coords) < min_cluster_size or len(non_cancer_coords) < min_cluster_size:
            return {
                'cluster_cooccurrence_ratio': 0.0,
                'mixed_cluster_proportion': 0.0
            }
        
        # Use DBSCAN to efficiently identify clusters
        # Estimate eps from k-nearest neighbor distances
        eps = np.mean(distances[:, -1]) * 1.5  # Adjust multiplier as needed
        
        # Run DBSCAN separately on cancer and non-cancer cells
        cancer_indices = np.where(cell_types == 1)[0]
        non_cancer_indices = np.where(cell_types == 0)[0]
        
        cancer_dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size-1)
        non_cancer_dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size-1)
        
        cancer_labels = cancer_dbscan.fit_predict(cancer_coords)
        non_cancer_labels = non_cancer_dbscan.fit_predict(non_cancer_coords)
        
        # Get valid clusters (ignore noise points labeled as -1)
        valid_cancer_clusters = {}
        for i, label in enumerate(cancer_labels):
            if label >= 0:  # Not noise
                if label not in valid_cancer_clusters:
                    valid_cancer_clusters[label] = []
                valid_cancer_clusters[label].append(cancer_indices[i])
        
        valid_non_cancer_clusters = {}
        for i, label in enumerate(non_cancer_labels):
            if label >= 0:  # Not noise
                if label not in valid_non_cancer_clusters:
                    valid_non_cancer_clusters[label] = []
                valid_non_cancer_clusters[label].append(non_cancer_indices[i])
        
        cancer_clusters = list(valid_cancer_clusters.values())
        non_cancer_clusters = list(valid_non_cancer_clusters.values())
        
        # Optimize cluster adjacency check using KDTree
        def clusters_adjacent_optimized(cluster1, cluster2, max_dist=50):
            points1 = coords[cluster1]
            points2 = coords[cluster2]
            
            # Use KDTree to efficiently find if any points are within max_dist
            tree = cKDTree(points1)
            # Find if any point in points2 is within max_dist of any point in points1
            nearest_dists, _ = tree.query(points2, k=1)
            return np.any(nearest_dists <= max_dist)
        
        # Count co-occurring cluster pairs more efficiently
        cooccurring_clusters = 0
        total_clusters = len(cancer_clusters) + len(non_cancer_clusters)
        
        if total_clusters == 0:
            return {
                'cluster_cooccurrence_ratio': 0.0,
                'mixed_cluster_proportion': 0.0
            }
        
        for cancer_cluster in cancer_clusters:
            for non_cancer_cluster in non_cancer_clusters:
                if clusters_adjacent_optimized(cancer_cluster, non_cancer_cluster):
                    cooccurring_clusters += 1
                    break  # Count each cluster only once
        
        cooccurrence_ratio = cooccurring_clusters / total_clusters if total_clusters > 0 else 0.0
        
        # Mixed neighborhood calculation more efficiently
        mixed_count = np.sum([(1 in cell_types[indices[i][1:]]) and 
                              (0 in cell_types[indices[i][1:]]) 
                              for i in range(len(coords))])
        
        mixed_proportion = mixed_count / len(coords) if len(coords) > 0 else 0.0
        
        return {
            'cluster_cooccurrence_ratio': cooccurrence_ratio,
            'mixed_cluster_proportion': mixed_proportion
        }
    except Exception as e:
        return {
            'cluster_cooccurrence_ratio': 0.0,
            'mixed_cluster_proportion': 0.0
        }



def calculate_neighborhood_enrichment_test(coords, cell_types, k=10):
    """
    Perform neighborhood enrichment test to identify significant spatial associations
    between cancer and non-cancer cells.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Array of shape (n, 2) containing x, y coordinates for each cell
    cell_types : numpy.ndarray
        Array of shape (n,) containing cell type labels (0 for non-cancer, 1 for cancer)
    k : int
        Number of nearest neighbors to consider
        
    Returns:
    --------
    dict
        Dictionary containing neighborhood enrichment statistics
    """
    if len(coords) <= k:
        print("warn no coords for enrichment score")
        return {
            'cancer_in_cancer_enrichment': 0.0,
            'non_cancer_in_cancer_enrichment': 0.0,
            
        }
    
    try:
        # Find k nearest neighbors for each cell
        nn = NearestNeighbors(n_neighbors=k+1)  # +1 because cell is its own neighbor
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        
        # Get cell indices by type
        cancer_indices = np.where(cell_types == 1)[0]
        non_cancer_indices = np.where(cell_types == 0)[0]
        
        if len(cancer_indices) == 0 or len(non_cancer_indices) == 0:
            return {
                'cancer_in_cancer_enrichment': 0.0,
                'non_cancer_in_cancer_enrichment': 0.0,
                
            }
        
        # Total counts
        total_cells = len(coords)
        total_cancer = len(cancer_indices)
        total_non_cancer = len(non_cancer_indices)
        
        # Expected probabilities
        expected_cancer_prob = total_cancer / total_cells
        expected_non_cancer_prob = total_non_cancer / total_cells
        
        # Count observed interactions
        cancer_cancer_count = 0  # Cancer cells in cancer neighborhoods
        non_cancer_cancer_count = 0  # Non-cancer cells in cancer neighborhoods
        
        # For each cancer cell, count neighbors by type
        for idx in cancer_indices:
            neighbors = indices[idx][1:]  # Skip the first index (the cell itself)
            cancer_neighbors = np.sum(cell_types[neighbors] == 1)
            non_cancer_neighbors = k - cancer_neighbors
            
            cancer_cancer_count += cancer_neighbors
            non_cancer_cancer_count += non_cancer_neighbors
        
        # Calculate enrichment scores
        # Observed vs expected ratio for cancer cells in cancer neighborhoods
        expected_cancer_cancer = k * len(cancer_indices) * expected_cancer_prob
        cancer_in_cancer_enrichment = (cancer_cancer_count / expected_cancer_cancer if expected_cancer_cancer > 0 else 0.0)
        
        # Observed vs expected ratio for non-cancer cells in cancer neighborhoods
        expected_non_cancer_cancer = k * len(cancer_indices) * expected_non_cancer_prob
        non_cancer_in_cancer_enrichment = (non_cancer_cancer_count / expected_non_cancer_cancer if expected_non_cancer_cancer > 0 else 0.0)
        
        # Simple chi-square test for significance
        observed = np.array([cancer_cancer_count, non_cancer_cancer_count])
        expected = np.array([expected_cancer_cancer, expected_non_cancer_cancer])
        
            
        return {
            'cancer_in_cancer_enrichment': cancer_in_cancer_enrichment,
            'non_cancer_in_cancer_enrichment': non_cancer_in_cancer_enrichment,
        }
    except Exception as e:
        print(e)
        return {
            'cancer_in_cancer_enrichment': 0.0,
            'non_cancer_in_cancer_enrichment': 0.0,
        }

def calculate_objectobject_correlation(coords, cell_types, max_radius=100, num_bins=10):
    """
    Calculate Object-Object Correlation Analysis (similar to Ripley's cross K function)
    to measure spatial correlation between cancer and non-cancer cells.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Array of shape (n, 2) containing x, y coordinates for each cell
    cell_types : numpy.ndarray
        Array of shape (n,) containing cell type labels (0 for non-cancer, 1 for cancer)
    max_radius : float
        Maximum radius to consider for correlation analysis
    num_bins : int
        Number of distance bins for correlation analysis
        
    Returns:
    --------
    dict
        Dictionary containing object-object correlation metrics
    """
    if len(coords) < 4:
        return {
            'cross_k_auc': 0.0,
            'correlation_strength': 0.0
        }
    
    # Get indices by cell type
    cancer_indices = np.where(cell_types == 1)[0]
    non_cancer_indices = np.where(cell_types == 0)[0]
    
    if len(cancer_indices) < 2 or len(non_cancer_indices) < 2:
        return {
            'cross_k_auc': 0.0,
            'correlation_strength': 0.0
        }
    
    try:
        # Get coordinates by cell type
        cancer_coords = coords[cancer_indices]
        non_cancer_coords = coords[non_cancer_indices]
        
        # Calculate area (for normalization)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            area = hull.volume  # For 2D, volume is area
        except:
            # Fallback if convex hull fails
            x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
            y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
            area = x_range * y_range
        
        # Define radius bins
        radii = np.linspace(0, max_radius, num_bins + 1)
        radius_mids = (radii[1:] + radii[:-1]) / 2
        
        # Calculate cross K function (cancer to non-cancer)
        cross_k_values = []
        
        # Calculate distances between all cancer and non-cancer cells
        dist_matrix = distance.cdist(cancer_coords, non_cancer_coords)
        
        # Calculate cross K function for each radius
        for i in range(num_bins):
            r_min, r_max = radii[i], radii[i+1]
            count = np.sum((dist_matrix >= r_min) & (dist_matrix < r_max))
            
            # Normalize by area and populations
            # Formula: K(r) = area * count / (n_cancer * n_non_cancer)
            k_r = area * count / (len(cancer_indices) * len(non_cancer_indices))
            cross_k_values.append(k_r)
        
        cross_k_values = np.array(cross_k_values)
        
        # Calculate expected K values under CSR (Complete Spatial Randomness)
        # For CSR, K(r) = π * r²
        expected_k = np.pi * radius_mids**2
        
        # Calculate difference between observed and expected K
        # Positive values indicate attraction, negative values indicate repulsion
        k_diff = cross_k_values - expected_k
        
        # Calculate AUC as a measure of overall correlation
        # Normalize by dividing by the span of radii
        cross_k_auc = np.trapz(k_diff, dx=max_radius/num_bins) / max_radius
        
        # Calculate correlation strength as maximum deviation from expected
        correlation_strength = np.max(np.abs(k_diff))
        
        return {
            'cross_k_auc': cross_k_auc,
            'correlation_strength': correlation_strength
        }
    except Exception as e:
        return {
            'cross_k_auc': 0.0,
            'correlation_strength': 0.0
        }


def calculate_mixing_score(coords, cell_types, k=10):
    """
    Alternative optimized implementation using scipy's entropy function
    """
    if len(coords) <= k:
        return 0.0
    
    # Make sure cell_types are integers
    cell_types = np.round(cell_types).astype(int)
    
    # Use KD-Tree for efficient nearest neighbor queries
    tree = cKDTree(coords)
    
    # Query k+1 nearest neighbors (including self)
    distances, indices = tree.query(coords, k=k+1)
    
    # Remove self (first column)
    neighbor_indices = indices[:, 1:]
    
    # Calculate entropy for each neighborhood
    entropies = []
    for i in range(len(coords)):
        neighbor_types = cell_types[neighbor_indices[i]]
        type_counts = np.bincount(neighbor_types, minlength=2)
        # Use scipy's entropy function which handles zero probabilities
        entropies.append(shannon_entropy(type_counts, base=2))
    
    return np.mean(entropies)


def calculate_clustering_metric(coords, cell_types, metric_type='silhouette'):
    """
    Calculate clustering metrics for cancer vs non-cancer cells
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Need at least 2 points of each class for clustering metrics
    if len(coords) < 4 or np.sum(cell_types) < 2 or np.sum(cell_types) > len(cell_types) - 2:
        return 0.0
    
    try:
        if metric_type == 'silhouette':
            return silhouette_score(coords, cell_types)
        elif metric_type == 'calinski_harabasz':
            return calinski_harabasz_score(coords, cell_types)
        elif metric_type == 'davies_bouldin':
            return davies_bouldin_score(coords, cell_types)
    except:
        return 0.0
    
    return 0.0




def calculate_edge_interior_ratio(coords, cell_types, cancer_type=1):
    """
    Calculate the ratio of edge cells to interior cells for cancer regions
    """
    # Get cancer cells
    cancer_coords = coords[cell_types == cancer_type]
    
    if len(cancer_coords) < 4:
        return 0.0
    
    try:
        # Create a Delaunay triangulation
        tri = Delaunay(cancer_coords)
        
        # Find the boundary points
        boundary_points = set()
        
        # For each simplex (triangle in 2D)
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                j = (i + 1) % len(simplex)
                edge = frozenset([simplex[i], simplex[j]])
                
                # If this edge appears only once, it's on the boundary
                if edge not in boundary_points:
                    boundary_points.add(edge)
                else:
                    boundary_points.remove(edge)
        
        # Count unique boundary points
        boundary_indices = set()
        for edge in boundary_points:
            boundary_indices.update(edge)
        
        num_boundary = len(boundary_indices)
        num_interior = len(cancer_coords) - num_boundary
        
        if num_interior == 0:
            return float('inf')  # All points are boundary points
        
        return num_boundary / num_interior
    except:
        return 0.0

def calculate_density_ratio(coords, cell_types):
    """
    Calculate the ratio of cancer cell density to non-cancer cell density
    """
    cancer_coords = coords[cell_types == 1]
    non_cancer_coords = coords[cell_types == 0]
    
    if len(cancer_coords) < 2 or len(non_cancer_coords) < 2:
        return 1.0  # Default to neutral ratio
    
    try:
        # Calculate areas using convex hull
        cancer_hull = ConvexHull(cancer_coords)
        non_cancer_hull = ConvexHull(non_cancer_coords)
        
        cancer_area = cancer_hull.volume  # In 2D, volume is area
        non_cancer_area = non_cancer_hull.volume
        
        # Calculate densities
        cancer_density = len(cancer_coords) / cancer_area if cancer_area > 0 else 0
        non_cancer_density = len(non_cancer_coords) / non_cancer_area if non_cancer_area > 0 else 0
        
        # Calculate ratio
        if non_cancer_density == 0:
            return float('inf')
        
        return cancer_density / non_cancer_density
    except:
        return 1.0

def calculate_distance_to_nearest_diff(coords, cell_types):
    """
    Calculate average distance from each cell to the nearest cell of different type
    """
    cancer_indices = np.where(cell_types == 1)[0]
    non_cancer_indices = np.where(cell_types == 0)[0]
    
    if len(cancer_indices) == 0 or len(non_cancer_indices) == 0:
        return 0.0
    
    cancer_coords = coords[cancer_indices]
    non_cancer_coords = coords[non_cancer_indices]
    
    # Calculate distances from cancer to non-cancer
    cancer_to_non = distance.cdist(cancer_coords, non_cancer_coords, 'euclidean')
    # Get minimum distance for each cancer cell
    min_dist_cancer = np.min(cancer_to_non, axis=1)
    
    # Calculate distances from non-cancer to cancer
    non_to_cancer = distance.cdist(non_cancer_coords, cancer_coords, 'euclidean')
    # Get minimum distance for each non-cancer cell
    min_dist_non = np.min(non_to_cancer, axis=1)
    
    # Combine and calculate average
    all_min_dists = np.concatenate([min_dist_cancer, min_dist_non])
    return np.mean(all_min_dists)
