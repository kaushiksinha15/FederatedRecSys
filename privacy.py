from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

def setup_privacy(model, optimizer, dataloader, noise_multiplier=1.0, max_grad_norm=1.0, epochs=1):
    """
    Attaches Opacus PrivacyEngine to the PyTorch training loop.
    Different noise_multipliers can be passed based on the client's cluster.
    """
    privacy_engine = PrivacyEngine()
    
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    
    return model, optimizer, dataloader, privacy_engine
