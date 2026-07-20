import flwr as fl
from typing import Dict, List, Tuple, Optional
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
import ast

from clustering import ClientClusterer

class AdaptiveClusteredStrategy(fl.server.strategy.FedAvg):
    """
    Custom strategy that extends FedAvg to support Behavioral Clustering
    and Cluster-Specific Privacy Budgeting.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clusterer = ClientClusterer(num_clusters=2)
        self.cluster_models = {}
        self.client_clusters = {}
        self.cluster_budgets = {}
        self.prev_client_clusters = {}
        self.prev_centroids = {}
        self.round_start_models = {}
        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training."""
        # Call the parent configure_fit which handles sampling correctly
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Apply custom config and cluster-specific models
        updated_instructions = []
        self.round_start_models = {}
        for client, fit_ins in client_instructions:
            cid = client.cid if client.cid else str(id(client))
            cluster_id = self.client_clusters.get(cid, 0)
            
            # Send cluster-specific model if available
            if cluster_id in self.cluster_models:
                cluster_params = ndarrays_to_parameters(self.cluster_models[cluster_id])
                self.round_start_models[cid] = self.cluster_models[cluster_id]
            else:
                cluster_params = fit_ins.parameters
                self.round_start_models[cid] = parameters_to_ndarrays(fit_ins.parameters)
                
            custom_config = {
                "noise_multiplier": self.cluster_budgets.get(cluster_id, 1.0)
            }
            updated_fit_ins = fl.common.FitIns(cluster_params, custom_config)
            updated_instructions.append((client, updated_fit_ins))
            
        return updated_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using Behavioral Clustering."""
        if not results:
            return None, {}

        # 1. Extract behavioral metrics sent from clients
        client_metrics = {}
        for client, fit_res in results:
            if "behavior_metric" in fit_res.metrics:
                cid = client.cid if client.cid else str(id(client))
                metric_str = fit_res.metrics["behavior_metric"]
                client_metrics[cid] = ast.literal_eval(metric_str)

        # 2. Cluster the clients based on behaviors
        cluster_assignments = self.clusterer.fit_predict(client_metrics)
        print(f"\n--- Server Round {server_round} ---")
        print(f"Client Clusters: {cluster_assignments}")
        
        # Compute cluster statistics
        centroids, global_centroid, variances, outlier_scores, client_privacy_risks = (
            self.clusterer.compute_cluster_stats(client_metrics, cluster_assignments)
        )
        
        # 3. Compute Jaccard Stability for each cluster
        cluster_stabilities = {}
        if server_round == 1 or not hasattr(self, 'prev_client_clusters') or not self.prev_client_clusters:
            for k in set(cluster_assignments.values()):
                cluster_stabilities[k] = 1.0
        else:
            # Align current clusters with previous clusters based on centroids
            for k, centroid in centroids.items():
                best_prev_k = None
                min_dist = float('inf')
                for prev_k, prev_centroid in self.prev_centroids.items():
                    dist = np.sum((centroid - prev_centroid) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_prev_k = prev_k
                
                # Compute Jaccard
                current_members = {cid for cid, c_id in cluster_assignments.items() if c_id == k}
                prev_members = {cid for cid, c_id in self.prev_client_clusters.items() if c_id == best_prev_k}
                
                union_len = len(current_members.union(prev_members))
                if union_len > 0:
                    jaccard = len(current_members.intersection(prev_members)) / union_len
                else:
                    jaccard = 0.0
                cluster_stabilities[k] = jaccard

        # 4. Semantic Behavior-Aware Privacy Budgets
        for k in set(cluster_assignments.values()):
            N_k = sum(1 for c_id in cluster_assignments.values() if c_id == k)
            S_k = cluster_stabilities.get(k, 1.0)
            
            # Extract cluster centroid behavioral descriptors
            if k in centroids:
                centroid = centroids[k]
                pref_div = centroid[0]
                behav_cons = centroid[1]
                temp_stab = centroid[2]
                behav_uniq = centroid[3]
                cluster_aff = centroid[4]
            else:
                pref_div = behav_cons = temp_stab = behav_uniq = cluster_aff = 0.5
            
            # Semantic Behavior Characterization privacy allocation
            # Higher uniqueness / diversity -> needs more privacy/noise
            # Higher consistency / stability -> needs less noise
            privacy_score = (pref_div + behav_uniq) / (behav_cons + temp_stab + 1e-9)
            
            lambda_val = 1.0 
            sigma_k = lambda_val * privacy_score
            
            # Apply stability scaling: stable -> lower noise, unstable -> higher noise
            sigma_k = sigma_k * (1.5 - 0.5 * S_k)
            
            self.cluster_budgets[k] = float(sigma_k)
            print(f"Cluster {k} -> Size={N_k}, PrivacyScore={privacy_score:.4f}, Stability={S_k:.4f} -> Assigned Noise={sigma_k:.4f}")

        # Update cache for next round
        self.prev_client_clusters = cluster_assignments
        self.prev_centroids = centroids
        self.client_clusters = cluster_assignments

        # 5. Privacy-and-Trust Aware Aggregation
        cluster_clients = {}
        for client, fit_res in results:
            cid = client.cid if client.cid else str(id(client))
            cluster_id = cluster_assignments.get(cid, 0)
            if cluster_id not in cluster_clients:
                cluster_clients[cluster_id] = []
            cluster_clients[cluster_id].append((cid, client, fit_res))
            
        all_client_trust_weights = {}
        all_client_models = {}
        
        for cluster_id, c_clients in cluster_clients.items():
            if not c_clients:
                continue
                
            num_layers = len(parameters_to_ndarrays(c_clients[0][2].parameters))
            
            # Extract client updates (delta_W_i = W_i - W_start)
            client_W_list = []
            delta_W_list = []
            for cid, client, fit_res in c_clients:
                W_i = parameters_to_ndarrays(fit_res.parameters)
                client_W_list.append(W_i)
                
                W_start = self.round_start_models.get(cid)
                if W_start is not None and len(W_start) == len(W_i):
                    delta_W = [w_i - w_s for w_i, w_s in zip(W_i, W_start)]
                else:
                    delta_W = [np.zeros_like(w_i) for w_i in W_i]
                delta_W_list.append(delta_W)
                
            # Compute consensus update (mean delta_W in cluster)
            consensus_update = []
            for layer in range(num_layers):
                layer_updates = [delta_W_list[i][layer] for i in range(len(delta_W_list))]
                consensus_update.append(np.mean(layer_updates, axis=0))
                
            # Calculate trust weight w_i for each client
            w_list = []
            for idx, (cid, client, fit_res) in enumerate(c_clients):
                # Update consistency (UC_i)
                dev_i = sum(np.mean((delta_W_list[idx][layer] - consensus_update[layer])**2) for layer in range(num_layers))
                UC_i = 1.0 / (1.0 + dev_i)
                
                # Data Quality (DQ_i)
                DQ_i = float(fit_res.num_examples)
                
                # Privacy Risk (PR_i)
                PR_i = client_privacy_risks.get(cid, 0.0)
                
                # Trust-aware weight: w_i = max(w_min, DQ_i * UC_i * (1 / (1 + PR_i)))
                w_min = 0.01
                w_i = max(w_min, DQ_i * UC_i * (1.0 / (1.0 + PR_i)))
                w_list.append(w_i)
                
                # Store globally
                all_client_trust_weights[cid] = w_i
                all_client_models[cid] = client_W_list[idx]
                
                print(f"Aggregation weight -> Cluster {cluster_id}, Client {cid}: DQ={DQ_i}, UC={UC_i:.6f}, PR={PR_i:.6f} -> Trust Weight={w_i:.6f}")
                
            w_sum = sum(w_list)
            if w_sum > 0:
                normalized_weights = [w / w_sum for w in w_list]
            else:
                normalized_weights = [1.0 / len(w_list)] * len(w_list)
                
            # Aggregate client models to compute the new cluster model
            W_k_new = []
            for layer in range(num_layers):
                layer_avg = sum(normalized_weights[i] * client_W_list[i][layer] for i in range(len(client_W_list)))
                W_k_new.append(layer_avg)
                
            self.cluster_models[cluster_id] = W_k_new
            
        # 6. Global Model Aggregation using Trust Weights across all clients
        global_w_list = []
        global_W_list = []
        for cid in all_client_trust_weights.keys():
            global_w_list.append(all_client_trust_weights[cid])
            global_W_list.append(all_client_models[cid])
            
        global_w_sum = sum(global_w_list)
        if global_w_sum > 0:
            global_normalized_weights = [w / global_w_sum for w in global_w_list]
        else:
            global_normalized_weights = [1.0 / len(global_w_list)] * len(global_w_list)
            
        num_layers_global = len(global_W_list[0])
        aggregated_ndarrays = []
        for layer in range(num_layers_global):
            layer_avg = sum(global_normalized_weights[i] * global_W_list[i][layer] for i in range(len(global_W_list)))
            aggregated_ndarrays.append(layer_avg)
            
        # Save the global model on the last round
        if server_round == 3:
            np.savez("global_model.npz", *aggregated_ndarrays)
            print("Saved global_model.npz")
            
        return ndarrays_to_parameters(aggregated_ndarrays), {}
        
    def aggregate(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: List[np.ndarray] = [
            np.sum(layer_updates, axis=0) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

if __name__ == "__main__":
    strategy = AdaptiveClusteredStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8082",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
