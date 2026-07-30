# Projet de Machine Learning Aérodynamique : Éolienne MEXICO

Ce projet implémente une architecture de réseaux de neurones (MLP) hybrides pour prédire les efforts aérodynamiques sur une éolienne (basé sur l'expérience MEXICO), en utilisant différentes approches spatiales et physiques.

Les codes de cette branche ont pour objectif de permettre la correction du modèle BEM (solveur grossier) par rapport au modèle Free Wake Vortex (solveur fin). 

Les géométries des éoliennes d'un parc sont fixés. De plus, les grilles de discrétisations radiale et azimutale sont également fixés. 

La correction par apprentissage supervisée prendra en covariables tout un ensemble de paramètres environnementaux et tentera d'inférer sur les forces normales et tangentielles qui seront appliquées sur une pale. 

L'objectif sur le long terme est d'accélérer la procédure de calcul de ces efforts en utilisant plutôt la BEM que Free Wake Vortex.
## Table des matières
1. [Pré-requis](#installation)
2. [Exécution](#exécution)
3. [Nomenclature des Modèles](#nomenclature-des-modèles)
4. [Nature des Données](#nature-des-données)
5. [Architecture du Code](#architecture-du-code)

---

## Pré-requis
Placez-vous à la racine du projet et exécutez :
```bash
pip install -r requirements.txt
```

## Exécution

Pour entraîner les 18 modèles (Optuna + évaluation finale) :

```bash
python main.py
```

Pour afficher un tableau résumé comparatif et enregistrer 3 images :

```bash
python summary.py
```

---

## Nomenclature des Modèles

Chaque modèle est nommé selon la convention suivante :  
`{Entrée}_{Résiduelle}_{Intermédiaire}`

### 1. Stratégie d'Entrée (Spatiale)

- **L (Locale)** : prédiction point par point  
  Entrées : rayon local (`r`) et azimut (`theta`)

- **GR (Global Rayons)** :  
  Entrée = azimut (`theta`)  
  Sortie = vecteur complet de toutes les sections de la pale

- **GA (Global Azimuts)** :  
  Entrée = rayon (`r`)  
  Sortie = vecteur complet de la distribution sur un tour de rotor

### 2. Stratégie Résiduelle

- **0 (Standard)** : le réseau prédit directement les grandeurs issues du code SVEN  
- **1 (Résiduel)** : le réseau prédit uniquement la différence (`SVEN - BEM`)  
  → Les données BEM sont fournies en entrée

### 3. Variables Intermédiaires (Cibles)

- **f (Forces)** : prédiction directe des efforts `Fn` (normale) et `Ft` (tangentielle)  
- **v (Vitesses)** : prédiction de la vitesse effective (`V_eff`) et de l'angle d'attaque (`alpha`)  
- **u (Induction)** : prédiction du facteur d'induction (`a`) et de l'angle de flux (`phi`)  

---

## Nature des Données

### 1. Géométrie (`geometry/`)

- `blade_geom.csv` : distribution de la corde (`c`) et du vrillage (`beta`) le long du rayon  
- `airfoils.csv` : base de données des coefficients de portance (`C_L`) et de traînée (`C_D`) en fonction de l’angle d’attaque  

### 2. Données Brutes (`data/raw/`)

- **Forces** : efforts `F_n` et `F_t` en Newton par mètre `[N/m]`  
- **Vitesses** : vitesses effectives 
- **Induction** : facteurs d'induction axiale et angles de flux  

### 3. Modèles Physiques de Référence

- **SVEN** : code de référence (vortex libre), considéré comme la *vérité terrain*  
- **BEM** : modèle simplifié (Blade Element Momentum), utilisé pour les modèles résiduels  
- **CASTOR** : données de comparaison plus précises que SVEN  

---

## Architecture du Code

- `main.py` : orchestrateur de la campagne d'entraînement  
- `summary.py` : génération d’un tableau comparatif des performances 
- `src/data_loader.py` : chargement des données, split Train/Test (70/30) et encodage cyclique de l’azimut (sin/cos)  
- `src/physics.py` : équations aérodynamiques pour convertir les prédictions en forces réelles  
- `src/models.py` : définition de l’architecture MLP (Multi-Layer Perceptron)  
- `src/optimize.py` : optimisation des hyperparamètres via Optuna  
- `src/evaluate.py` : entraînement final, conversion physique et calcul des métriques (RMSE, Wasserstein)  
