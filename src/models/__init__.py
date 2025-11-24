"""
Module de modèles d'IA
"""

from .cnn_model import FaceClassificationCNN
from .federated_client import FederatedClient
from .federated_server import FederatedServer

__all__ = [
    "FaceClassificationCNN",
    "FederatedClient", 
    "FederatedServer"
]