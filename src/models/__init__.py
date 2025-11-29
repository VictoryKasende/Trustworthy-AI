"""
Module de modèles d'IA
"""

from .cnn_model import FaceClassificationCNN

# Les modules federated_client et federated_server seront ajoutés au notebook 04
# from .federated_client import FederatedClient
# from .federated_server import FederatedServer

__all__ = [
    "FaceClassificationCNN",
    # "FederatedClient", 
    # "FederatedServer"
]