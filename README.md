# 🍲 Notation automatique de commentaires de recettes (NLP)

**🇫🇷 Français** · [🇬🇧 English version below](#-english-version)

Classification de commentaires français issus de **Marmiton** en **5 notes (1 à 5)** avec des réseaux récurrents (**GRU** et **LSTM**, PyTorch). Le projet étudie aussi l'effet d'une **augmentation de données** pour compenser le fort déséquilibre des classes.

> Projet de groupe de M1 (CY Cergy Paris Université, 2022–2023). Ce dépôt contient **ma partie** : la classification des notes et l'évaluation.

## Résultats

Évaluation sur un **jeu de test indépendant**, une exécution par modèle (graine 42). Le F1 macro (moyenne du F1 de chaque classe) est l'indicateur principal : l'exactitude est trompeuse car environ 80 % des commentaires ont la note 5.

| Modèle | Exactitude | F1 macro |
|---|---|---|
| Baseline (toujours la note 5) | 0,804 | 0,178 |
| GRU sans augmentation | 0,787 | **0,356** |
| GRU avec augmentation | 0,796 | 0,330 |
| LSTM sans augmentation | 0,772 | 0,318 |
| LSTM avec augmentation | 0,784 | 0,318 |

**Ce qu'on peut en dire**
- Tous les modèles ont un F1 macro nettement supérieur à la baseline (0,32 à 0,36 contre 0,18) : ils apprennent quelque chose sur les classes rares, sans dépasser la classe majoritaire sur l'exactitude brute.
- **GRU et LSTM ont des performances proches.** Le GRU est légèrement devant, mais avec une seule exécution par modèle, l'écart peut être du bruit.
- **L'augmentation de données n'apporte pas d'amélioration mesurable** (aucun effet sur le LSTM, légère baisse sur le GRU).
- La tâche est difficile : 5 notes, texte court et classes très déséquilibrées.

## Données

- Commentaires et notes de recettes collectés sur **Marmiton** (**non fournis dans ce dépôt**).
- Format attendu du fichier JSON : une liste de recettes, chacune avec `_id.$oid` et une liste `comments` contenant `comment` (texte) et `appreciation` (note de 1 à 5).
- Après nettoyage et suppression des doublons : **[~20000] commentaires uniques**.

## Méthode

1. **Nettoyage** : ponctuation, mots vides français (NLTK), minuscules, chiffres, URL, emojis (`demoji`), caractères répétés.
2. **Déduplication** après nettoyage (un seul exemplaire de chaque texte).
3. **Séparation stratifiée train / validation / test (70 / 15 / 15) avant toute augmentation**, avec vérification qu'aucun texte n'est présent dans deux ensembles.
4. **Augmentation (train uniquement, classes rares : notes 1 à 3)** : Word2Vec entraîné sur le train ; environ 30 % des mots d'un commentaire sont remplacés par l'un de leurs 3 plus proches voisins. Une à deux variantes par commentaire (2 pour les notes 1 et 2, 1 pour la note 3).
5. **Modèles** : embeddings appris (256), 1 couche GRU ou LSTM (256 unités), dropout 0,5, Adam (lr = 1e-3), 10 époques, batch de 64, séquences tronquées à 128 mots avec *packing* (le padding n'est pas lu).
6. **Sélection de l'époque** sur le F1 macro de validation, puis **évaluation unique sur le test**.

## Limites

- **Une seule exécution** par configuration : un écart de 2 à 3 points de F1 macro peut venir du hasard. Pour confirmer une différence, il faudrait répéter avec plusieurs graines.
- Les classes rares (notes 1 à 3) restent mal reconnues.
- L'augmentation par substitution de voisins Word2Vec dépend d'un modèle entraîné sur peu de données : les voisins ne sont pas toujours des synonymes.
- Pas de modèle préentraîné (par exemple CamemBERT), qui serait une piste évidente d'amélioration.

## Lancer le projet

```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_DEPOT>
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux
pip install torch gensim nltk demoji scikit-learn pandas numpy jupyter
```

1. Place ton fichier de commentaires dans le dossier et ajuste `DATA_PATH` (cellule 1 du notebook).
2. Ouvre `gru_classification_corrige.ipynb` et exécute les cellules dans l'ordre. Les sections 1 à 7 entraînent les GRU, la section 8 compare au LSTM.
3. L'entraînement fonctionne sur CPU mais est plus rapide avec un GPU.

## Contenu

```
├── gru_classification_corrige.ipynb   # Pipeline complet : nettoyage, séparation, augmentation, GRU, LSTM
└── README.md
```

---

# 🇬🇧 English version

Classification of French **Marmiton** recipe comments into **5 ratings (1 to 5)** with recurrent networks (**GRU** and **LSTM**, PyTorch). The project also studies the effect of **data augmentation** to address strong class imbalance.

> M1 group project (CY Cergy Paris Université, 2022–2023). This repository contains **my part**: rating classification and evaluation.

## Results

Evaluated on an **independent test set**, one run per model (seed 42). Macro-F1 (mean of per-class F1) is the main metric: accuracy is misleading because about 80% of the comments have rating 5.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Baseline (always rating 5) | 0.804 | 0.178 |
| GRU without augmentation | 0.787 | **0.356** |
| GRU with augmentation | 0.796 | 0.330 |
| LSTM without augmentation | 0.772 | 0.318 |
| LSTM with augmentation | 0.784 | 0.318 |

**Takeaways**
- All models reach a macro-F1 well above the baseline (0.32 to 0.36 versus 0.18): they learn something about the rare classes, without beating the majority class on raw accuracy.
- **GRU and LSTM perform similarly.** GRU is slightly ahead, but with a single run per model the gap may be noise.
- **Data augmentation brings no measurable improvement** (no effect on LSTM, slight drop on GRU).
- The task is hard: 5 ratings, short texts and highly imbalanced classes.

## Data

- Recipe comments and ratings collected from **Marmiton** (**not included in this repository**).
- Expected JSON format: a list of recipes, each with `_id.$oid` and a `comments` list containing `comment` (text) and `appreciation` (rating from 1 to 5).
- After cleaning and deduplication: **[~20000] unique comments**.

## Method

1. **Cleaning:** punctuation, French stop words (NLTK), lowercasing, digits, URLs, emojis (`demoji`), repeated characters.
2. **Deduplication** after cleaning (one copy of each text).
3. **Stratified train / validation / test split (70 / 15 / 15) before any augmentation**, with a check that no text appears in two sets.
4. **Augmentation (training set only, rare classes: ratings 1 to 3):** Word2Vec trained on the training set; about 30% of a comment's words are replaced by one of their 3 nearest neighbours. One or two variants per comment (2 for ratings 1 and 2, 1 for rating 3).
5. **Models:** learned embeddings (256), one GRU or LSTM layer (256 units), dropout 0.5, Adam (lr = 1e-3), 10 epochs, batch size 64, sequences truncated to 128 words with packing (padding is not read).
6. **Epoch selection** on validation macro-F1, then a **single evaluation on the test set**.

## Limitations

- **A single run** per configuration: a 2 to 3 point macro-F1 gap may be due to chance. Confirming a difference would require repeating the experiment with several seeds.
- Rare classes (ratings 1 to 3) remain poorly recognised.
- Word2Vec-neighbour substitution relies on a model trained on little data: neighbours are not always synonyms.
- No pretrained model (for example CamemBERT), which would be an obvious improvement.

## Run the project

```bash
git clone <REPO_URL>
cd <REPO_NAME>
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux
pip install torch gensim nltk demoji scikit-learn pandas numpy jupyter
```

1. Put your comments file in the folder and adjust `DATA_PATH` (notebook cell 1).
2. Open `gru_classification_corrige.ipynb` and run the cells in order. Sections 1 to 7 train the GRUs, section 8 compares with the LSTM.
3. Training runs on CPU but is faster on a GPU.

## Contents

```
├── gru_classification_corrige.ipynb   # Full pipeline: cleaning, split, augmentation, GRU, LSTM
└── README.md
```
