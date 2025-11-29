"""
Préprocessing des images pour la classification faciale
"""

import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import yaml
from typing import Tuple, List, Dict, Optional


class ImagePreprocessor:
    """Classe pour le préprocessing des images faciales"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialise le preprocessor
        
        Args:
            target_size: Taille cible pour redimensionner les images
        """
        self.target_size = target_size
        self.label_encoder = LabelEncoder()
        self.member_names = []
        
    def load_member_images(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge les images des 3 membres depuis les dossiers
        
        Args:
            data_path: Chemin vers le dossier data/raw/
            
        Returns:
            Tuple contenant les images et les labels encodés
        """
        images = []
        labels = []
        
        # Parcourir les dossiers des 3 membres
        for member_dir in sorted(os.listdir(data_path)):
            if member_dir.startswith('member'):
                member_path = os.path.join(data_path, member_dir)
                if os.path.isdir(member_path):
                    self.member_names.append(member_dir)
                    
                    # Charger toutes les images du membre
                    for img_file in os.listdir(member_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(member_path, img_file)
                            
                            # Charger et préprocesser l'image
                            image = self._load_and_preprocess_image(img_path)
                            if image is not None:
                                images.append(image)
                                labels.append(member_dir)
        
        # Convertir en arrays numpy
        X = np.array(images)
        y = self.label_encoder.fit_transform(labels)
        
        print(f"Chargé {len(X)} images pour {len(self.member_names)} membres")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return X, y
    
    def _load_and_preprocess_image(self, img_path: str) -> Optional[np.ndarray]:
        """
        Charge et préprocesse une image
        
        Args:
            img_path: Chemin vers l'image
            
        Returns:
            Image préprocessée ou None si erreur
        """
        try:
            # Charger l'image
            image = cv2.imread(img_path)
            if image is None:
                return None
                
            # Convertir BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensionner
            image = cv2.resize(image, self.target_size)
            
            # Normaliser les pixels entre 0 et 1
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Erreur lors du chargement de {img_path}: {e}")
            return None
    
    def split_data_for_federated_learning(self, X: np.ndarray, y: np.ndarray, 
                                        test_size: float = 0.2) -> Dict:
        """
        Divise les données pour l'apprentissage fédéré
        
        Args:
            X: Images
            y: Labels
            test_size: Proportion des données de test
            
        Returns:
            Dictionnaire avec les données pour chaque client et test
        """
        # Split train/test global
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Créer les données fédérées (un client par membre)
        federated_data = {}
        
        for i, member in enumerate(self.member_names):
            # Données de ce membre pour l'entraînement
            member_mask = y_train == i
            client_X = X_train[member_mask]
            client_y = y_train[member_mask]
            
            federated_data[f'client_{i+1}'] = {
                'X_train': client_X,
                'y_train': client_y,
                'member_name': member
            }
        
        federated_data['test'] = {
            'X_test': X_test,
            'y_test': y_test
        }
        
        return federated_data
    
    def save_processed_data(self, federated_data: Dict, output_path: str):
        """
        Sauvegarde les données préprocessées
        
        Args:
            federated_data: Données divisées
            output_path: Chemin de sauvegarde
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Sauvegarder les données de chaque client
        for client_name, data in federated_data.items():
            if client_name != 'test':
                np.savez(
                    os.path.join(output_path, f'{client_name}.npz'),
                    X_train=data['X_train'],
                    y_train=data['y_train']
                )
        
        # Sauvegarder les données de test
        np.savez(
            os.path.join(output_path, 'test_data.npz'),
            X_test=federated_data['test']['X_test'],
            y_test=federated_data['test']['y_test']
        )
        
        # Sauvegarder les métadonnées
        metadata = {
            'target_size': list(self.target_size),  # Convertir tuple en list
            'member_names': self.member_names,
            'classes': self.label_encoder.classes_.tolist(),
            'num_classes': len(self.label_encoder.classes_)
        }
        
        with open(os.path.join(output_path, 'metadata.yaml'), 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)


class DataAugmentation:
    """Classe pour l'augmentation des données"""
    
    def __init__(self):
        """Initialise les transformations d'augmentation"""
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
        ])
    
    def augment_batch(self, images: np.ndarray, factor: int = 3) -> np.ndarray:
        """
        Applique l'augmentation à un batch d'images
        
        Args:
            images: Batch d'images à augmenter
            factor: Facteur d'augmentation
            
        Returns:
            Images augmentées
        """
        augmented_images = []
        
        for img in images:
            # Image originale
            augmented_images.append(img)
            
            # Images augmentées
            for _ in range(factor - 1):
                # Convertir pour albumentations (0-255, uint8)
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Appliquer transformation
                transformed = self.transform(image=img_uint8)
                aug_img = transformed['image'].astype(np.float32) / 255.0
                
                augmented_images.append(aug_img)
        
        return np.array(augmented_images)


if __name__ == "__main__":
    # Script pour tester le preprocessing
    preprocessor = ImagePreprocessor()
    
    # Charger les données
    data_path = "../../data/raw"
    if os.path.exists(data_path):
        X, y = preprocessor.load_member_images(data_path)
        
        # Diviser pour l'apprentissage fédéré
        federated_data = preprocessor.split_data_for_federated_learning(X, y)
        
        # Sauvegarder
        output_path = "../../data/processed"
        preprocessor.save_processed_data(federated_data, output_path)
        
        print("Preprocessing terminé avec succès!")
    else:
        print(f"Le dossier {data_path} n'existe pas. Veuillez ajouter vos photos d'abord.")