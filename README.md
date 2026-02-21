# IMT ML Classification

Projet minimal de classification d'images de déchets (organique, papier, plastique) avec TensorFlow/Keras.

## Structure attendue du dataset

Le dataset local doit suivre cette structure :

```text
data/
  train/
    organique/
    papier/
    plastique/
```

Comptage observé dans la source actuelle :

- organique: 53 images
- papier: 55 images
- plastique: 62 images

## Lancer le préprocessing + split

```bash
python train.py --data-dir ../data/train
```

Le script charge le dataset, applique un split train/validation (80/20) et affiche les classes détectées.