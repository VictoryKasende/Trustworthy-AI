"""
Modèle CNN pour la classification faciale
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50
import numpy as np
from typing import Tuple, Optional, Dict


class FaceClassificationCNN:
    """Modèle CNN pour la classification faciale des 3 membres"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 3,
                 model_type: str = "custom"):
        """
        Initialise le modèle CNN
        
        Args:
            input_shape: Forme d'entrée des images (H, W, C)
            num_classes: Nombre de classes (3 membres)
            model_type: Type de modèle ("custom", "mobilenet", "resnet")
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def build_custom_model(self) -> tf.keras.Model:
        """
        Construit un modèle CNN personnalisé
        
        Returns:
            Modèle Keras compilé
        """
        model = models.Sequential([
            # Bloc de convolution 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloc de convolution 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloc de convolution 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloc de convolution 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Couches denses
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Couche de sortie
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_learning_model(self, base_model_name: str = "mobilenet") -> tf.keras.Model:
        """
        Construit un modèle avec transfer learning
        
        Args:
            base_model_name: "mobilenet" ou "resnet"
            
        Returns:
            Modèle Keras avec transfer learning
        """
        # Choisir le modèle de base
        if base_model_name == "mobilenet":
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                alpha=1.0,
                include_top=False,
                weights='imagenet'
            )
        elif base_model_name == "resnet":
            base_model = ResNet50(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Modèle de base non supporté: {base_model_name}")
        
        # Geler les couches du modèle pré-entraîné
        base_model.trainable = False
        
        # Ajouter des couches personnalisées
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_model(self) -> tf.keras.Model:
        """
        Construit le modèle selon le type spécifié
        
        Returns:
            Modèle Keras
        """
        if self.model_type == "custom":
            model = self.build_custom_model()
        elif self.model_type == "mobilenet":
            model = self.build_transfer_learning_model("mobilenet")
        elif self.model_type == "resnet":
            model = self.build_transfer_learning_model("resnet")
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")
        
        # Compiler le modèle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self, model_path: str = "best_model.h5") -> list:
        """
        Configure les callbacks d'entraînement
        
        Args:
            model_path: Chemin pour sauvegarder le meilleur modèle
            
        Returns:
            Liste des callbacks
        """
        callbacks_list = [
            # Sauvegarde du meilleur modèle
            callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Réduction du learning rate
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def train(self, 
              train_dataset: tf.data.Dataset,
              validation_dataset: tf.data.Dataset,
              epochs: int = 100,
              model_save_path: str = "face_classification_model.h5") -> Dict:
        """
        Entraîne le modèle
        
        Args:
            train_dataset: Dataset d'entraînement
            validation_dataset: Dataset de validation
            epochs: Nombre d'époques
            model_save_path: Chemin de sauvegarde du modèle
            
        Returns:
            Historique d'entraînement
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks_list = self.get_callbacks(model_save_path)
        
        # Entraînement
        self.history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history.history
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict:
        """
        Évalue le modèle sur les données de test
        
        Args:
            test_dataset: Dataset de test
            
        Returns:
            Métriques d'évaluation
        """
        if self.model is None:
            raise ValueError("Le modèle doit être construit et entraîné avant évaluation")
        
        # Évaluation
        results = self.model.evaluate(test_dataset, verbose=1)
        
        # Formatage des résultats
        metrics_names = self.model.metrics_names
        evaluation_results = dict(zip(metrics_names, results))
        
        return evaluation_results
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Effectue des prédictions sur de nouvelles images
        
        Args:
            images: Images à classifier
            
        Returns:
            Prédictions (probabilités)
        """
        if self.model is None:
            raise ValueError("Le modèle doit être construit et entraîné avant prédiction")
        
        predictions = self.model.predict(images)
        return predictions
    
    def get_model_summary(self) -> str:
        """
        Retourne un résumé du modèle
        
        Returns:
            Résumé du modèle sous forme de string
        """
        if self.model is None:
            return "Modèle non construit"
        
        # Capturer le résumé dans une string
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        summary = buffer.getvalue()
        sys.stdout = old_stdout
        
        return summary
    
    def save_model(self, filepath: str):
        """
        Sauvegarde le modèle complet
        
        Args:
            filepath: Chemin de sauvegarde
        """
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        self.model.save(filepath)
        print(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath: Chemin du modèle à charger
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Modèle chargé: {filepath}")
    
    def fine_tune(self, 
                  train_dataset: tf.data.Dataset,
                  validation_dataset: tf.data.Dataset,
                  learning_rate: float = 1e-5,
                  epochs: int = 50):
        """
        Fine-tuning pour les modèles avec transfer learning
        
        Args:
            train_dataset: Dataset d'entraînement
            validation_dataset: Dataset de validation
            learning_rate: Taux d'apprentissage pour le fine-tuning
            epochs: Nombre d'époques
        """
        if self.model is None:
            raise ValueError("Le modèle doit être construit avant le fine-tuning")
        
        if self.model_type not in ["mobilenet", "resnet"]:
            print("Le fine-tuning est recommandé pour les modèles avec transfer learning")
            return
        
        # Dégeler quelques couches du modèle de base
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Geler les premières couches, dégeler les dernières
        fine_tune_at = len(base_model.layers) // 2
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompiler avec un learning rate plus faible
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Fine-tuning
        fine_tune_history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            verbose=1
        )
        
        return fine_tune_history.history


if __name__ == "__main__":
    # Test du modèle
    print("=== Test du modèle CNN ===")
    
    # Créer une instance du modèle
    model = FaceClassificationCNN(
        input_shape=(224, 224, 3),
        num_classes=3,
        model_type="custom"
    )
    
    # Construire le modèle
    keras_model = model.build_model()
    
    # Afficher le résumé
    print("\\nRésumé du modèle:")
    print(model.get_model_summary())
    
    # Test avec des données factices
    print("\\nTest avec des données factices...")
    dummy_data = np.random.random((10, 224, 224, 3))
    dummy_labels = tf.keras.utils.to_categorical(
        np.random.randint(0, 3, 10), 
        num_classes=3
    )
    
    # Test de prédiction
    predictions = model.predict(dummy_data)
    print(f"Forme des prédictions: {predictions.shape}")
    print(f"Exemple de prédiction: {predictions[0]}")
    
    print("\\nModèle CNN créé avec succès!")