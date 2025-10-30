import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import yaml


def load_config(config_path, dataset, args):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if dataset not in config:
        raise ValueError(f"Dataset {dataset} not found in config file.")

    dataset_config = config[dataset]
    for key, value in dataset_config.items():
        if getattr(args, key) is None:  # Only set if not already passed via CLI
            setattr(args, key, value)
    return args, config


def l2_normalize(arr):
    """Row-wise L2 normalization"""
    arr = torch.tensor(arr, dtype=torch.float32)
    arr = F.normalize(arr, p=2, dim=1)
    return arr.numpy()

def dim_matching(vision, language):
    input_dim1 = vision[0].shape[1]
    input_dim2 = language[0].shape[1]   
    
    linear = nn.Linear(input_dim1, input_dim2)
    with torch.no_grad():
        output = [linear(vision[0])]
    return output



def adaptive_threshold_selection(similarity_matrix, percentile=90):
    """
    Adaptively select threshold based on similarity distribution
    """
    # Remove diagonal (self-similarities)
    sim_no_diag = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]
    threshold = np.percentile(sim_no_diag, percentile)
    return max(threshold, 0.1)  # Ensure minimum connectivity


def build_graph(features, k_neighbors=10, threshold_percentile=85, normalize=True, hybrid=True):
    """
    graph construction with hybrid approach (k-NN + threshold)
    """
    if "torch" in str(type(features)):
        features = features.cpu().numpy()
    
    if normalize:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    similarity_matrix = cosine_similarity(features)
    n_nodes = features.shape[0]
    
    if hybrid:
        # Hybrid approach: k-NN + adaptive threshold
        adjacency_matrix = np.zeros_like(similarity_matrix)
        
        # 1. k-NN connectivity for each node
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, metric='cosine').fit(features)
        distances, indices = nbrs.kneighbors(features)
        
        for i in range(n_nodes):
            # Skip self (first neighbor)
            for j in indices[i][1:]:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Make symmetric
        
        # 2. Add high-similarity edges
        threshold = adaptive_threshold_selection(similarity_matrix, threshold_percentile)
        high_sim_edges = (similarity_matrix >= threshold).astype(int)
        adjacency_matrix = np.logical_or(adjacency_matrix, high_sim_edges).astype(int)
    else:
        # Original threshold-based approach with adaptive threshold
        threshold = adaptive_threshold_selection(similarity_matrix, threshold_percentile)
        adjacency_matrix = (similarity_matrix >= threshold).astype(int)
    
    # Remove self-loops
    np.fill_diagonal(adjacency_matrix, 0)
    
    # Ensure minimum connectivity
    if np.sum(adjacency_matrix) == 0:
        # Fallback: connect each node to its nearest neighbor
        for i in range(n_nodes):
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            nearest = np.argmax(similarities)
            adjacency_matrix[i, nearest] = 1
            adjacency_matrix[nearest, i] = 1
    
    features = [torch.tensor(features, dtype=torch.float32)]
    adjacency_matrix = [torch.tensor(adjacency_matrix, dtype=torch.float32)]
    
    return features, adjacency_matrix

def dim_matching_improved(X1, X2, method='linear'):
    """
    Make X1 to the dimention of X2
    Dimension matching with better initialization
    """
    input_dim1 = X1[0].shape[1]
    input_dim2 = X2[0].shape[1]
    
    if method == 'linear':
        linear = nn.Linear(input_dim1, input_dim2)
        # Xavier initialization for better convergence
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        
        with torch.no_grad():
            output = [linear(X1[0])]
    elif method == 'mlp':
        # Use MLP for non-linear transformation
        hidden_dim = max(input_dim1, input_dim2)
        mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim2)
        )
        
        # Initialize weights
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        with torch.no_grad():
            output = [mlp(X1[0])]
    else:  # PCA-based alignment
        from sklearn.decomposition import PCA
        min_dim = min(input_dim1, input_dim2)
        
        # Apply PCA to both modalities
        pca_vis = PCA(n_components=min_dim)
        pca_lang = PCA(n_components=min_dim)
        
        vis_reduced = pca_vis.fit_transform(X1[0].numpy())
        lang_reduced = pca_lang.fit_transform(X2[0].numpy())
        
        # Project vision to language space using learned mapping
        linear = nn.Linear(min_dim, input_dim2)
        with torch.no_grad():
            vis_tensor = torch.tensor(vis_reduced, dtype=torch.float32)
            output = [linear(vis_tensor)]
    
    return output




def load_vision_language(dataset, vision, language, k_neig=10, portion=1, normalize=True):
    modals_name = ["vision", "language"]
    train_features, train_adj, test_pairs = {}, {}, {}
    
    vis = torch.tensor(np.load(f'../data/vision-language/{dataset}/{vision}_train.npy'))
    if dataset == 'CIFAR-100':
        lan = torch.tensor(np.load(f'../data/vision-language/Language100/prompt_5/{language}.npy'))
    elif dataset == 'Imagenet-100':
        lan = torch.tensor(np.load(f'../data/vision-language/Language100-Imagenet/prompt_5/{language}.npy'))
    else:
        lan = torch.tensor(np.load(f'../data/vision-language/Language/prompt_5/{language}.npy'))
    
    if normalize:
        vis = l2_normalize(vis)
        lan = l2_normalize(lan)

    # train_features['vision'], train_adj['vision'] = build_graph(vis, threshold=0.2)
    # train_features['language'], train_adj['language'] = build_graph(lan, threshold=0.3)
    
    # choose a subset of classes
    if portion != 1:
        n_classes = vis.shape[0]
        n_idx = int(portion*n_classes)
        idx = torch.randperm(n_classes)[0:n_idx]
        vis = vis[idx,:]
        lan = lan[idx,:]
    
    # Build improved graphs
    train_features['vision'], train_adj['vision'] = build_graph(vis, k_neighbors=k_neig, threshold_percentile=80, hybrid=False)
    train_features['language'], train_adj['language'] = build_graph(lan, k_neighbors=k_neig, threshold_percentile=85, hybrid=False)
       
    
    # Test pairs for evaluation (ground truth alignment)
    test_pairs = torch.tensor(np.array([np.arange(0, train_adj['vision'][0].shape[0]), np.arange(0, train_adj['vision'][0].shape[0])]), dtype=torch.long)
    
    return train_features, train_adj, test_pairs, modals_name
    
    
    


def load_vision_language_seed(dataset, vision, language, k_neig=10, normalize=True):
    modals_name = ["vision", "language"]
    train_features, train_adj, test_pairs = {}, {}, {}
    
      
    vis = torch.tensor(np.load(f'../data/vision-language/{dataset}/seed/{vision}_seed_train.npy'))
    if dataset == 'CIFAR-100':
        lan = torch.tensor(np.load(f'../data/vision-language/Language100/prompt_5/{language}.npy'))
    elif dataset == 'Imagenet-100':
        lan = torch.tensor(np.load(f'../data/vision-language/Language100-Imagenet/prompt_5/{language}.npy'))
    else:
        lan = torch.tensor(np.load(f'../data/vision-language/Language/prompt_5/{language}.npy'))
    
    if normalize:
        vis = l2_normalize(vis)
        lan = l2_normalize(lan)

    # train_features['vision'], train_adj['vision'] = build_graph(vis, threshold=0.2)
    # train_features['language'], train_adj['language'] = build_graph(lan, threshold=0.3)
    
    # Build improved graphs
    train_features['vision'] = []
    train_adj['vision'] = []
    for i in range(20):
        features, adj = build_graph(vis[i], k_neighbors=k_neig, threshold_percentile=80, hybrid=False)
        train_features['vision'].append(features[0])
        train_adj['vision'].append(adj[0])
        
    train_features['language'], train_adj['language'] = build_graph(lan, k_neighbors=k_neig, threshold_percentile=85, hybrid=False)
    
        
    # Test pairs for evaluation (ground truth alignment)
    test_pairs = torch.tensor(np.array([np.arange(0, train_adj['vision'][0].shape[0]), np.arange(0, train_adj['vision'][0].shape[0])]), dtype=torch.long)
    
    return train_features, train_adj, test_pairs, modals_name    