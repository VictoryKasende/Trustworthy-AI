"""
Chargeurs de données pour l'apprentissage fédéré
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from typing import Tuple, Dict, Generator
import yaml


class FederatedDataLoader:
    """Chargeur de données pour l'apprentissage fédéré"""
    
    def __init__(self, data_path: str, batch_size: int = 32):
        """
        Initialise le chargeur fédéré
        
        Args:
            data_path: Chemin vers les données préprocessées
            batch_size: Taille des batches
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.metadata = self._load_metadata()
        self.num_classes = self.metadata['num_classes']
        
    def _load_metadata(self) -> Dict:
        """Charge les métadonnées"""
        metadata_path = os.path.join(self.data_path, 'metadata.yaml')
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_client_data(self, client_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Récupère les données d'un client spécifique
        
        Args:
            client_id: ID du client (1, 2, ou 3)
            
        Returns:
            Tuple (X_train, y_train) pour ce client
        """
        client_file = os.path.join(self.data_path, f'client_{client_id}.npz')
        
        if not os.path.exists(client_file):
            raise FileNotFoundError(f"Données du client {client_id} introuvables: {client_file}")
        
        data = np.load(client_file)
        X_train = data['X_train']
        y_train = data['y_train']
        
        # Convertir les labels en one-hot encoding
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        
        return X_train, y_train
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Récupère les données de test globales
        
        Returns:
            Tuple (X_test, y_test) 
        """
        test_file = os.path.join(self.data_path, 'test_data.npz')
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Données de test introuvables: {test_file}")
        
        data = np.load(test_file)
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Convertir les labels en one-hot encoding
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        
        return X_test, y_test
    
    def create_client_dataset(self, client_id: int) -> tf.data.Dataset:
        """
        Crée un dataset TensorFlow pour un client
        
        Args:
            client_id: ID du client
            
        Returns:
            Dataset TensorFlow configuré
        """
        X_train, y_train = self.get_client_data(client_id)
        
        # Créer le dataset TensorFlow
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.shuffle(buffer_size=len(X_train))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_test_dataset(self) -> tf.data.Dataset:
        """
        Crée le dataset de test
        
        Returns:
            Dataset TensorFlow de test
        """
        X_test, y_test = self.get_test_data()
        
        dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_client_info(self) -> Dict:
        """
        Retourne les informations sur tous les clients
        
        Returns:
            Dictionnaire avec les infos clients
        """
        info = {
            'num_clients': 3,
            'num_classes': self.num_classes,
            'class_names': self.metadata['classes'],
            'member_names': self.metadata['member_names'],
            'target_size': self.metadata['target_size']
        }
        
        # Ajouter les statistiques de chaque client
        for client_id in range(1, 4):
            try:
                X_train, y_train = self.get_client_data(client_id)
                info[f'client_{client_id}'] = {
                    'num_samples': len(X_train),
                    'member_name': self.metadata['member_names'][client_id - 1],
                    'shape': X_train.shape
                }
            except FileNotFoundError:
                info[f'client_{client_id}'] = {
                    'num_samples': 0,
                    'member_name': f'member{client_id}',
                    'shape': 'No data'
                }
        
        return info


class LocalDataLoader:
    """Chargeur de données pour entraînement local (non-fédéré)"""
    
    def __init__(self, data_path: str, batch_size: int = 32, validation_split: float = 0.2):
        """
        Initialise le chargeur local
        
        Args:
            data_path: Chemin vers les données préprocessées
            batch_size: Taille des batches
            validation_split: Proportion des données de validation
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.federated_loader = FederatedDataLoader(data_path, batch_size)
    
    def get_combined_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine les données de tous les clients pour entraînement centralisé
        
        Returns:
            Tuple (X_train, y_train) combiné
        """
        all_X = []
        all_y = []
        
        # Combiner les données des 3 clients
        for client_id in range(1, 4):
            try:
                X_client, y_client = self.federated_loader.get_client_data(client_id)
                all_X.append(X_client)
                all_y.append(y_client)
            except FileNotFoundError:
                print(f"Attention: Données du client {client_id} introuvables")
                continue
        
        if not all_X:
            raise ValueError("Aucune donnée de client trouvée")
        
        # Concaténer toutes les données
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        
        return X_combined, y_combined
    
    def create_train_val_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Crée les datasets d'entraînement et validation
        
        Returns:
            Tuple (train_dataset, val_dataset)
        """
        X_train, y_train = self.get_combined_training_data()
        
        # Division train/validation
        val_size = int(len(X_train) * self.validation_split)
        train_size = len(X_train) - val_size
        
        # Mélanger les données
        indices = np.random.permutation(len(X_train))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        # Créer les datasets TensorFlow
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_split, y_train_split))
        train_dataset = train_dataset.shuffle(buffer_size=train_size)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset


if __name__ == "__main__":
    # Test des chargeurs de données
    data_path = "../../data/processed"
    
    if os.path.exists(data_path):
        # Test du chargeur fédéré
        fed_loader = FederatedDataLoader(data_path)
        client_info = fed_loader.get_client_info()
        
        print("=== Informations sur les clients ===")
        for key, value in client_info.items():
            print(f"{key}: {value}")
        
        # Test du chargeur local
        local_loader = LocalDataLoader(data_path)
        try:
            train_ds, val_ds = local_loader.create_train_val_datasets()
            print(f"\\nDataset d'entraînement créé avec succès")
            print(f"Dataset de validation créé avec succès")
        except ValueError as e:
            print(f"\\nErreur lors de la création des datasets: {e}")
    
    else:
        print(f"Le dossier {data_path} n'existe pas. Exécutez d'abord preprocessing.py")