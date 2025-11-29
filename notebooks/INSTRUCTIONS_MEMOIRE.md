# 🚨 Instructions pour éviter le crash du kernel

## Problème identifié

Le kernel Python meurt après 7-8 minutes d'entraînement à cause d'un **manque de RAM**. Le système tue le processus (OOM Killer) pour protéger le système.

## ✅ Optimisations appliquées

### 1. **Résolution des images réduite**

- **Avant**: 224x224 pixels
- **Après**: 128x128 pixels
- **Économie**: ~70% de mémoire par image

### 2. **Batch size ultra-réduit**

- **Avant**: 32 → 16
- **Après**: 8
- **Économie**: 75% de mémoire par batch vs config initiale

### 3. **Nombre d'epochs réduit**

- **Avant**: 100 → 30
- **Après**: 15
- **Raison**: Test de stabilité, augmentable si ça fonctionne

### 4. **Mixed Precision désactivé**

- **Raison**: Peut causer des pics mémoire imprévisibles
- **Impact**: Entraînement plus stable mais légèrement plus lent

### 5. **Optimisations TensorFlow**

- CPU threads limités à 2
- Prefetch limité à 2 (au lieu de AUTOTUNE)
- Workers=1, pas de multiprocessing
- Garbage collection agressive après chaque epoch

### 6. **Nettoyage mémoire**

- Suppression des variables intermédiaires
- GC forcé plusieurs fois
- Callback personnalisé de nettoyage

## 📋 Étapes pour relancer l'entraînement

1. **Redémarrer le kernel complètement**

   - Kernel → Restart Kernel
   - Cela libère toute la mémoire

2. **Réexécuter les cellules dans l'ordre**

   - Cellule 1: Configuration (nouvelles optimisations)
   - Cellules 2-8: Chargement et préparation données
   - Cellule 9-14: Construction modèle
   - Cellule 15-16: Configuration entraînement
   - **Cellule 17: ENTRAÎNEMENT** (nouvelle version optimisée)

3. **Surveiller la mémoire**
   - Ouvrir un terminal: `htop` ou `watch -n 1 free -h`
   - Observer l'utilisation RAM pendant l'entraînement

## 🎯 Si ça marche

Si l'entraînement se termine sans crash :

1. Vous pouvez augmenter progressivement :

   - Epochs: 15 → 20 → 30
   - Batch size: 8 → 12 → 16
   - Résolution: 128 → 160 → 224

2. Tester une modification à la fois

## ⚠️ Si ça crash encore

Solutions supplémentaires :

1. **Réduire encore le modèle** : Retirer 1 bloc convolutionnel
2. **Batch size à 4** : Entraînement très lent mais stable
3. **Utiliser uniquement 50% des données** : Pour test rapide
4. **Cloud gratuit** : Google Colab (15 GB RAM gratuit)

## 💡 Estimation temps

Avec ces paramètres :

- **Batch size 8** : ~2x plus lent qu'avec 16
- **15 epochs** : ~1.5-2 heures d'entraînement
- **Mais stable** : Pas de crash !

## 🔧 Commandes utiles

```bash
# Voir la mémoire disponible
free -h

# Monitorer en temps réel
htop

# Vider le cache si nécessaire
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

---

**Bonne chance ! 🚀**
