# Interface avec pièges

Cette version est isolée de `interface/app.py` et des fichiers d'origine.
Elle produit des labyrinthes à branches, boucles et pièges `T`, de 5×5 à 10×10.
Tomber sur un piège renvoie l'agent au départ et donne une récompense de -25.

## Lancer en local

Depuis la racine du dépôt, démarrer l'API :

```bash
docker compose up -d
```

Puis lancer cette interface dans un autre terminal :

```bash
conda activate ml_env
python -m interface_pieges.app
```

L'interface utilise `http://127.0.0.1:8000` par défaut. Pour une API hébergée,
définir `API_BASE_URL` avant le lancement :

```bash
API_BASE_URL=https://adresse-de-ton-api python -m interface_pieges.app
```

Les modèles sont choisis dans l'interface et chargés par l'API. Elle attend
les modèles avec pièges dans `results/dqn_v2_traps/` et
`results/qlearning_improved_v2_traps/`.

## Déployer l'API sur Render

Tkinter est une interface de bureau : Render héberge l'API, et l'interface
Tkinter reste lancée sur ton ordinateur.

1. Vérifie que les deux fichiers de modèle indiqués ci-dessus sont inclus dans
   le dépôt Git. Le Dockerfile de ce dossier les copie dans l'image.
2. Sur Render, crée un **Web Service** relié au dépôt.
3. Laisse **Root Directory** vide (racine du dépôt), choisis **Docker** comme
   environnement et indique `interface_pieges/Dockerfile.api` comme Dockerfile.
4. Définis le chemin de vérification (`Health Check Path`) à `/health`, puis
   lance le déploiement.
5. Quand le service est actif, ouvre `https://<nom-du-service>.onrender.com/health`
   et vérifie que l'API répond.
6. Sur ton ordinateur, démarre l'interface en lui donnant l'adresse du service :

   ```bash
   API_BASE_URL=https://<nom-du-service>.onrender.com python -m interface_pieges.app
   ```

Le Dockerfile utilise le port fourni par Render et inclut les deux poids. Les
reconstructions de l'image réinstallent les dépendances ; le service web ne
peut pas afficher la fenêtre Tkinter dans le navigateur. Render documente la
configuration Docker, le champ Dockerfile Path et les vérifications HTTP dans
sa [documentation Docker](https://render.com/docs/docker) et sa
[documentation des vérifications de santé](https://render.com/docs/health-checks).
