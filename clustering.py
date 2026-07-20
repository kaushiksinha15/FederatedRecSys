import numpy as np
from sklearn.cluster import KMeans
import torch

class ClientClusterer:
    """
    Clusters clients based on their behavioral metrics (e.g., interaction frequencies)
    before federated aggregation.
    """
    def __init__(self, num_clusters=2):
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        
    def fit_predict(self, client_metrics_dict):
        """
        Takes a dictionary of {client_id: behavior_dict} and returns clusters.
        """
        if not client_metrics_dict:
            return {}
            
        client_ids = list(client_metrics_dict.keys())
        # Convert dictionary to feature array
        keys = ['preference_diversity', 'behavioral_consistency', 'temporal_stability', 'behavioral_uniqueness', 'cluster_affinity']
        features = np.array([[client_metrics_dict[cid][k] for k in keys] for cid in client_ids])
        
        # If we have fewer clients than requested clusters, reduce n_clusters for the run
        actual_clusters = min(self.num_clusters, len(client_ids))
        if actual_clusters != self.kmeans.n_clusters:
            self.kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            
        labels = self.kmeans.fit_predict(features)
        
        # Return dict mapping client_id -> cluster_id
        return {cid: int(label) for cid, label in zip(client_ids, labels)}

    def compute_cluster_stats(self, client_metrics_dict, cluster_assignments):
        """
        Computes cluster centroids, global mean, cluster variances, outlier scores,
        and individual client privacy risks (distances to centroids).
        """
        if not client_metrics_dict or not cluster_assignments:
            return {}, np.zeros(1), {}, {}, {}
            
        client_ids = list(client_metrics_dict.keys())
        keys = ['preference_diversity', 'behavioral_consistency', 'temporal_stability', 'behavioral_uniqueness', 'cluster_affinity']
        features = {cid: np.array([client_metrics_dict[cid][k] for k in keys]) for cid in client_ids}
        
        # Global mean centroid
        all_vectors = np.array(list(features.values()))
        global_centroid = np.mean(all_vectors, axis=0)
        
        # Group vectors by cluster
        cluster_vectors = {}
        for cid, cluster_id in cluster_assignments.items():
            if cluster_id not in cluster_vectors:
                cluster_vectors[cluster_id] = []
            cluster_vectors[cluster_id].append(features[cid])
            
        centroids = {}
        variances = {}
        outlier_scores = {}
        client_privacy_risks = {}
        
        for cluster_id, vectors in cluster_vectors.items():
            vectors = np.array(vectors)
            centroid = np.mean(vectors, axis=0)
            centroids[cluster_id] = centroid
            
            # Variance V_k (mean squared distance to centroid)
            variance = np.mean(np.sum((vectors - centroid) ** 2, axis=1)) if len(vectors) > 0 else 0.0
            variances[cluster_id] = float(variance)
            
            # Outlier score O_k (squared distance of centroid from global mean)
            outlier_score = np.sum((centroid - global_centroid) ** 2)
            outlier_scores[cluster_id] = float(outlier_score)
            
            # Client privacy risk PR_i (squared distance of client vector to its cluster centroid)
            for cid in client_ids:
                if cluster_assignments[cid] == cluster_id:
                    dist = np.sum((features[cid] - centroid) ** 2)
                    client_privacy_risks[cid] = float(dist)
                    
        return centroids, global_centroid, variances, outlier_scores, client_privacy_risks

class SemanticBehaviorCharacterization:
    """
    Derives behavioral descriptors from the local Transformer's item embeddings.
    """
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
    def extract(self):
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for seqs, targets in self.dataloader:
                seqs = seqs.to(self.device)
                embeddings = self.model.item_embedding(seqs)
                session_repr = embeddings.mean(dim=1) 
                all_embeddings.append(session_repr)
                
                if len(all_embeddings) >= 5: 
                    break
                    
        if len(all_embeddings) == 0:
            semantic_vector = torch.zeros(self.model.embed_dim).cpu().numpy()
        else:
            semantic_vector = torch.cat(all_embeddings, dim=0).mean(dim=0).cpu().numpy()
        
        # Calculate placeholders for behavioral descriptors based on the semantic vector
        # (Using simple descriptive statistics as placeholders)
        preference_diversity = float(np.std(semantic_vector) if len(semantic_vector) > 0 else 0.5)
        behavioral_consistency = float(np.mean(np.abs(semantic_vector)) if len(semantic_vector) > 0 else 0.5)
        temporal_stability = float(np.max(semantic_vector) - np.min(semantic_vector) if len(semantic_vector) > 0 else 0.5)
        behavioral_uniqueness = float(np.linalg.norm(semantic_vector) if len(semantic_vector) > 0 else 0.5)
        cluster_affinity = float(np.median(semantic_vector) if len(semantic_vector) > 0 else 0.5)
        
        # Adding some random Laplace noise for DP conceptually
        noise_scale = 0.05
        preference_diversity += np.random.laplace(0, noise_scale)
        behavioral_consistency += np.random.laplace(0, noise_scale)
        temporal_stability += np.random.laplace(0, noise_scale)
        behavioral_uniqueness += np.random.laplace(0, noise_scale)
        cluster_affinity += np.random.laplace(0, noise_scale)
        
        # Ensure positive
        return {
            "preference_diversity": max(0.01, preference_diversity),
            "behavioral_consistency": max(0.01, behavioral_consistency),
            "temporal_stability": max(0.01, temporal_stability),
            "behavioral_uniqueness": max(0.01, behavioral_uniqueness),
            "cluster_affinity": max(0.01, cluster_affinity)
        }

