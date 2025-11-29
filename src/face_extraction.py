"""
Module d'extraction des visages avec DeepFace
Extraction et normalisation des visages pour la reconnaissance faciale
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from deepface import DeepFace

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceExtractor:
    """Extracteur de visages utilisant DeepFace."""
    
    def __init__(
        self,
        detector_backend: str = 'retinaface',
        target_size: Tuple[int, int] = (224, 224),
        align: bool = True,
        anti_spoofing: bool = False
    ):
        """
        Initialiser l'extracteur de visages.
        
        Args:
            detector_backend: Backend de détection ('retinaface', 'mtcnn', 'opencv', etc.)
            target_size: Taille cible des visages extraits
            align: Aligner les visages
            anti_spoofing: Vérifier si le visage est réel (True/False)
        """
        self.detector_backend = detector_backend
        self.target_size = target_size
        self.align = align
        self.anti_spoofing = anti_spoofing
        
        logger.info(f"✅ FaceExtractor initialisé:")
        logger.info(f"   • Detector: {detector_backend}")
        logger.info(f"   • Target size: {target_size}")
        logger.info(f"   • Align: {align}")
        logger.info(f"   • Anti-spoofing: {anti_spoofing}")
    
    def extract_face(self, image_path: str) -> Optional[Dict]:
        """
        Extraire le visage principal d'une image.
        
        Args:
            image_path: Chemin vers l'image
        
        Returns:
            Dict contenant: {
                'face': array du visage (224x224x3),
                'confidence': confiance de la détection,
                'landmarks': points caractéristiques,
                'is_real': si le visage est réel (si anti_spoofing=True),
                'status': 'success' ou 'failed'
            }
        """
        try:
            # Vérifier que l'image existe
            if not os.path.exists(image_path):
                logger.warning(f"⚠️ Image non trouvée: {image_path}")
                return None
            
            # Extraire tous les visages
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                align=self.align,
                anti_spoofing=self.anti_spoofing
            )
            
            if not face_objs:
                logger.warning(f"⚠️ Aucun visage détecté dans: {image_path}")
                return None
            
            # Prendre le plus grand visage (probablement le principal)
            largest_face = max(face_objs, key=lambda x: x['face'].shape[0] * x['face'].shape[1])
            
            # Extraire les informations
            face_image = largest_face['face']
            
            # Vérifier la taille
            if face_image.shape[:2] != self.target_size:
                face_image = cv2.resize(face_image, self.target_size)
            
            # Normaliser entre 0 et 1
            if face_image.max() > 1.0:
                face_image = face_image.astype(np.float32) / 255.0
            
            result = {
                'face': face_image,
                'confidence': largest_face.get('confidence', 0.0),
                'is_real': largest_face.get('is_real', True),
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur d'extraction pour {image_path}: {str(e)}")
            return None
    
    def extract_faces_batch(
        self,
        image_paths: List[str],
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Extraire les visages d'un lot d'images.
        
        Args:
            image_paths: Liste des chemins vers les images
            verbose: Afficher la progression
        
        Returns:
            Dict {image_path: extraction_result}
        """
        results = {}
        successful = 0
        failed = 0
        
        for i, img_path in enumerate(image_paths):
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"   Traitement: {i+1}/{len(image_paths)}")
            
            result = self.extract_face(img_path)
            
            if result and result['status'] == 'success':
                results[img_path] = result
                successful += 1
            else:
                failed += 1
        
        if verbose:
            logger.info(f"✅ Extraction terminée:")
            logger.info(f"   • Succès: {successful}/{len(image_paths)} ({100*successful/len(image_paths):.1f}%)")
            logger.info(f"   • Échecs: {failed}")
        
        return results
    
    def extract_from_directory(
        self,
        directory: str,
        member_name: Optional[str] = None,
        extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extraire les visages de toutes les images d'un répertoire.
        
        Args:
            directory: Chemin du répertoire
            member_name: Nom du membre (pour les logs)
            extensions: Extensions de fichiers acceptées
        
        Returns:
            Tuple (faces_array, image_paths) où faces_array est (N, 224, 224, 3)
        """
        logger.info(f"📂 Extraction des visages du répertoire: {directory}")
        
        # Lister les images
        image_files = []
        for ext in extensions:
            image_files.extend(sorted(Path(directory).glob(f'*{ext}')))
        
        logger.info(f"   • {len(image_files)} images trouvées")
        
        if not image_files:
            logger.warning(f"⚠️ Aucune image trouvée dans {directory}")
            return np.array([]), []
        
        # Extraire les visages
        extracted_faces = []
        valid_paths = []
        
        for i, img_path in enumerate(image_files):
            if (i + 1) % 20 == 0:
                logger.info(f"   Traitement: {i+1}/{len(image_files)}")
            
            result = self.extract_face(str(img_path))
            
            if result and result['status'] == 'success':
                # Filtrer les visages avec faible confiance
                if result['confidence'] > 0.3:
                    extracted_faces.append(result['face'])
                    valid_paths.append(str(img_path))
        
        # Convertir en array
        faces_array = np.array(extracted_faces)
        
        logger.info(f"✅ Extraction réussie:")
        logger.info(f"   • Visages extraits: {len(extracted_faces)}/{len(image_files)}")
        logger.info(f"   • Shape: {faces_array.shape}")
        
        return faces_array, valid_paths
    
    def extract_all_members(
        self,
        data_directory: str,
        member_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Extraire les visages pour tous les membres.
        
        Args:
            data_directory: Répertoire racine contenant les dossiers des membres
            member_names: Liste des noms des membres
        
        Returns:
            Dict {member_name: faces_array}
        """
        logger.info(f"👥 Extraction des visages pour {len(member_names)} membres\n")
        
        all_faces = {}
        
        for member in member_names:
            member_dir = os.path.join(data_directory, member)
            
            if not os.path.isdir(member_dir):
                logger.warning(f"⚠️ Répertoire non trouvé: {member_dir}")
                continue
            
            logger.info(f"🔹 {member.upper()}")
            faces, paths = self.extract_from_directory(member_dir, member_name=member)
            
            if len(faces) > 0:
                all_faces[member] = faces
                logger.info(f"   ✅ {len(faces)} visages extraits\n")
            else:
                logger.warning(f"   ⚠️ Aucun visage extrait pour {member}\n")
        
        return all_faces
    
    def save_extracted_faces(
        self,
        all_faces: Dict[str, np.ndarray],
        output_path: str
    ) -> None:
        """
        Sauvegarder les visages extraits en NPZ.
        
        Args:
            all_faces: Dict {member_name: faces_array}
            output_path: Chemin du fichier NPZ de sortie
        """
        logger.info(f"💾 Sauvegarde des visages extraits: {output_path}")
        
        # Aplatir les données
        X_all = np.concatenate(list(all_faces.values()), axis=0)
        y_all = []
        
        for member_idx, (member, faces) in enumerate(all_faces.items()):
            y_all.extend([member_idx] * len(faces))
        
        y_all = np.array(y_all)
        
        # Sauvegarder
        np.savez(
            output_path,
            X=X_all,
            y=y_all,
            member_names=np.array(list(all_faces.keys()))
        )
        
        logger.info(f"✅ Sauvegarde réussie:")
        logger.info(f"   • X shape: {X_all.shape}")
        logger.info(f"   • y shape: {y_all.shape}")
        logger.info(f"   • Fichier: {output_path}")


def extract_and_save_faces(
    raw_data_dir: str,
    output_dir: str,
    member_names: List[str],
    detector_backend: str = 'retinaface',
    target_size: Tuple[int, int] = (224, 224)
) -> None:
    """
    Fonction complète d'extraction et sauvegarde des visages.
    
    Args:
        raw_data_dir: Répertoire des images brutes
        output_dir: Répertoire de sortie
        member_names: Liste des noms des membres
        detector_backend: Backend de détection à utiliser
        target_size: Taille cible des visages
    """
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialiser l'extracteur
    extractor = FaceExtractor(
        detector_backend=detector_backend,
        target_size=target_size,
        align=True,
        anti_spoofing=False
    )
    
    # Extraire tous les visages
    all_faces = extractor.extract_all_members(raw_data_dir, member_names)
    
    # Sauvegarder
    output_file = os.path.join(output_dir, 'extracted_faces.npz')
    extractor.save_extracted_faces(all_faces, output_file)
    
    logger.info(f"\n✅ EXTRACTION COMPLÈTE RÉUSSIE!")
    logger.info(f"   • Output: {output_file}")
