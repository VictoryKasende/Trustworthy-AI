"""
Trustworthy AI Package
=====================

Un package pour l'IA de confiance avec classification faciale,
apprentissage fédéré et techniques d'explainabilité.
"""

__version__ = "0.1.0"
__author__ = "Équipe Trustworthy AI"

from . import data
from . import models
from . import explainability
from . import privacy
from . import utils

__all__ = [
    "data",
    "models", 
    "explainability",
    "privacy",
    "utils"
]