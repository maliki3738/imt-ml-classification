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

## Lancer l'entraînement

```bash
python train.py --data-dir ../data/train --epochs 8
```

Le script :

- charge le dataset et applique un split train/validation (80/20)
- entraîne un modèle MobileNetV2 (transfer learning)
- sauvegarde le modèle dans `models/model_dechets.keras`
- écrit les métriques d'entraînement/validation dans `logs/metrics.csv`

## Lancer l'app Streamlit (optionnel)

```bash
streamlit run app/app.py
```

Prérequis : un modèle entraîné présent dans `models/model_dechets.keras`.

## Tests rapides

```bash
pytest -q
```

Le test vérifie la dimension de sortie du modèle et la cohérence des probabilités (softmax).