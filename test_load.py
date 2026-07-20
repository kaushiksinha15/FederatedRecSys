import torch
import numpy as np
from collections import OrderedDict
from model import TransformerRecommender
from opacus.validators import ModuleValidator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    print("Loading global_model.npz...")
    npzfile = np.load("global_model.npz")
    parameters = [npzfile[f'arr_{i}'] for i in range(len(npzfile.files))]
    
    print("Initializing model...")
    model = TransformerRecommender().to(device)
    # model = ModuleValidator.fix(model).to(device) # Try both
    
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    
    print("Loading state dict...")
    model.load_state_dict(state_dict, strict=True)
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
