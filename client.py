import argparse
import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict

from opacus.validators import ModuleValidator
from model import TransformerRecommender
from dataset import get_client_dataloader
from clustering import SemanticBehaviorCharacterization
from privacy import setup_privacy

class RecSysClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerRecommender()
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
        self.model = model.to(self.device)
        self.train_loader = get_client_dataloader(client_id, batch_size=32)
        
        # Calculate local behavior metrics to send to server for clustering
        characterization = SemanticBehaviorCharacterization(self.model, self.train_loader, self.device)
        self.behavior_metric = characterization.extract()
        self.model.train() # Opacus make_private requires model to be in train mode
        
        # Initialize privacy once to avoid adding hooks multiple times
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.private_model, self.optimizer, self.private_loader, self.privacy_engine = setup_privacy(
            self.model, self.optimizer, self.train_loader, noise_multiplier=1.0
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Apply dynamic noise multiplier from server
        if "noise_multiplier" in config:
            self.optimizer.noise_multiplier = float(config["noise_multiplier"])
            print(f"Client {self.client_id} using noise_multiplier: {self.optimizer.noise_multiplier:.2f}")
        
        criterion = nn.CrossEntropyLoss()
        
        self.private_model.train()

        epochs = 1
        for epoch in range(epochs):
            for seqs, targets in self.private_loader:
                seqs, targets = seqs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.private_model(seqs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

        # Get the actual epsilon spent
        epsilon = self.privacy_engine.get_epsilon(1e-5)
        print(f"Client {self.client_id} finished training. Epsilon spent: {epsilon:.2f}")

        # Return updated weights, num examples, and the behavior metric for server clustering
        # Ensure behavior metric is a valid format (e.g. string or list of floats)
        metrics = {
            "behavior_metric": str(self.behavior_metric),
            "epsilon": epsilon
        }
        
        # We need to return parameters of the underlying model, not the wrapped Opacus module
        return self.get_parameters(config), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        # Simplified evaluation
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for seqs, targets in self.train_loader:
                seqs, targets = seqs.to(self.device), targets.to(self.device)
                outputs = self.model(seqs)
                loss += criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracy = correct / total
        return float(loss), len(self.train_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client_id", type=int, required=True, help="Unique ID for this client")
    args = parser.parse_args()
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8082", 
        client=RecSysClient(args.client_id)
    )
