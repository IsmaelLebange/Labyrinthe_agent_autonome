# Résolution de labyrinthes : Q-learning et DQN

Projet de groupe consacré à la génération et à la résolution de labyrinthes par apprentissage par renforcement. Il réunit l’environnement de jeu, deux familles d’agents (Q-learning et DQN), leur entraînement et leur évaluation, une interface graphique et une API d’inférence conteneurisée.

**Dépôt GitLab officiel du groupe :** [groupe_52/Labyrinthe_agent_autonome](https://gitlab.com/groupe_52/Labyrinthe_agent_autonome)

Le dépôt GitLab est la référence du travail collectif : l’historique des branches et des commits permet de suivre les contributions de chacun. Ce projet et ses développements de groupe restent dans le groupe GitLab.

## Membres et répartition du travail

La répartition ci-dessous reprend le document de cadrage du projet.

| Membre | Responsabilités prévues |
|---|---|
| **Josué Bilongo** | Données et environnement : générer, préparer et valider les labyrinthes et la structure de l’environnement ; analyser les trajectoires et résultats exploratoires ; préparer les données expérimentales. |
| **Marie-France Engomba** | Modèles ML/DL : implémenter Q-learning et DQN, entraîner les agents, régler leurs hyperparamètres, évaluer les modèles et comparer leurs performances. |
| **Fanny Faila** | MLOps 1 : organiser le dépôt et le travail Git/GitLab, mettre en place le versionnement des données avec DVC, la CI/CD et les contrôles qualité automatisés (Flake8/Black), et automatiser entraînement ou tests si nécessaire. |
| **Ismael Lebange** | MLOps 2 : conteneuriser l’application, exposer l’inférence par API, organiser le chargement des modèles et leur déploiement, puis vérifier leur utilisation depuis l’application. |
| **Rejouie Nkanza** | Documentation et assurance qualité : coordonner le projet, documenter les travaux, suivre les métriques, contribuer aux tests unitaires et d’intégration, préparer le rapport et le support de présentation, et vérifier le respect du protocole. |

## Fonctionnement du projet

L’environnement représente un labyrinthe avec un départ, un objectif, des passages et des murs. À chaque étape, l’agent reçoit un état, choisit une action, puis l’environnement renvoie le nouvel état, une récompense et l’indication de fin d’épisode.

Le dépôt contient deux approches d’apprentissage par renforcement :

- **Q-learning** apprend une valeur d’action par état dans une table ;
- **DQN** apprend ces valeurs avec un réseau de neurones PyTorch.

Les scripts d’entraînement se trouvent dans `training/`, l’environnement dans `environnement.py` et `environments/`, les agents dans `models/`, et les scripts d’évaluation et de visualisation dans `evaluation/` et `visualization/`. Les métriques et courbes produites par les entraînements sont enregistrées sous `results/` lorsqu’elles sont générées.

### État utilisé par les modèles V2

Les modèles V2 attendent **12 valeurs** :

1. les 9 cases de l’observation locale 3×3 autour de l’agent ;
2. 2 valeurs décrivant la direction de l’objectif ;
3. la dernière action effectuée.

L’encodeur correspondant est `environments/qlearning_state_encoder_v2.py`. La taille et l’ordre de l’état doivent correspondre à ceux utilisés lors de l’entraînement du modèle.

## Entraîner les modèles

Depuis la racine du dépôt, dans l’environnement Python du projet :

```bash
python -m training.train_qlearning_improved_v2
python -m training.train_dqn_v2
```

Les scripts plus anciens et leurs variantes restent disponibles dans `training/`. Les fichiers de modèle sont sauvegardés sous `results/` selon le chemin défini dans le script d’entraînement. Les résultats numériques (récompenses, nombre d’étapes, succès, et parfois pertes) servent à analyser et comparer les entraînements ; ce ne sont pas des modèles eux-mêmes.

## Interface graphique

L’interface Tkinter locale historique se lance avec :

```bash
python -m interface.app
```

Elle permet de sélectionner un agent disponible, générer un labyrinthe, lancer la résolution et observer les résultats. Des variantes d’interface connectées à l’API sont également présentes dans `interface/`. Leur configuration de déploiement avec noVNC est distincte de l’interface locale.

## API d’inférence

L’API FastAPI se trouve dans `app/main.py`. Au démarrage, elle cherche les modèles V2 aux emplacements suivants (configurables par variables d’environnement) :

- DQN : `results/dqn_v2/model.pth` (`DQN_MODEL_PATH`) ;
- Q-learning : `results/qlearning_improved_v2/q_table.npy` (`QLEARNING_MODEL_PATH`).

### Lancer l’API en local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Documentation interactive : <http://localhost:8000/docs>.

### Routes disponibles

| Méthode | Route | Utilité |
|---|---|---|
| `GET` | `/health` | État de l’API et des modèles chargés. |
| `GET` | `/models` | Modèles configurés et état de leur chargement. |
| `POST` | `/load_model` | Charger ou recharger un modèle depuis un chemin accessible à l’application. |
| `POST` | `/predict` | Prédire une action à partir d’un état déjà encodé de 12 valeurs. |
| `POST` | `/predict_maze` | Encoder la grille, la position et la dernière action, puis demander une prédiction. |

Dans `/predict` et `/predict_maze`, `model_type` sélectionne `dqn` ou `qlearning`. Les détails des schémas de requête et de réponse sont visibles dans `/docs`.

## Docker et déploiement

L’API peut être exécutée dans le conteneur défini par `Dockerfile` et `docker-compose.yml` :

```bash
docker compose up --build
```

La configuration `Dockerfile.tkinter` et le script `interface/start_novnc.sh` concernent l’interface graphique exposée dans un navigateur via noVNC. C’est un service distinct de l’API ; il utilise l’URL de l’API configurée dans l’environnement (`MAZE_API_URL`). Les deux conteneurs doivent pouvoir accéder aux fichiers nécessaires et à l’API. Le déploiement graphique distant dépend également du bon démarrage de Xvfb, Tkinter, VNC et noVNC.

## Tests

Les tests couvrent l’API, les encodeurs, l’environnement et les modèles. Pour lancer toute la suite :

```bash
pytest
```

## Principaux dossiers

```text
app/          API FastAPI
adapters/     Adaptateurs d’inférence Q-table et DQN
environments/ Génération de labyrinthes et encodage des états
models/       Implémentations des agents Q-learning et DQN
training/     Scripts d’entraînement
evaluation/   Évaluation et comparaison des modèles
results/      Modèles entraînés et métriques générées
interface/    Interfaces Tkinter, dont les variantes connectées à l’API
tests/        Tests automatisés
visualization/ Graphiques et visualisations
```
