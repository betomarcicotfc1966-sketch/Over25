# Guide : mettre votre application en ligne

Vous avez 4 fichiers : `app.py`, `model.py`, `requirements.txt`, `GUIDE.md`.
Aucune clé API n'est nécessaire : les données viennent de football-data.co.uk (gratuit).

## Méthode A (recommandée) : en ligne, gratuit, sans rien installer

Résultat : une adresse du type `https://votre-nom.streamlit.app`, utilisable sur Android, PC, partout.
Tout se fait depuis un navigateur, même depuis le téléphone.

### 1. Créer un compte GitHub (5 min)
1. Allez sur **github.com** et touchez **Sign up**.
2. Choisissez un email, un mot de passe et un nom d'utilisateur, puis validez l'email reçu.

### 2. Créer le dépôt et envoyer les fichiers
1. Sur GitHub, touchez **+** puis **New repository**.
2. Nom : `over25`. Laissez **Public**. Touchez **Create repository**.
3. Touchez **uploading an existing file** (lien dans la page).
4. Ajoutez les 4 fichiers (`app.py`, `model.py`, `requirements.txt`, `GUIDE.md`). Sur Android, touchez « choose your files » puis sélectionnez-les dans Téléchargements.
5. Touchez **Commit changes**.

Important : les fichiers doivent être à la racine du dépôt, pas dans un sous-dossier.

### 3. Publier avec Streamlit Community Cloud
1. Allez sur **share.streamlit.io** et connectez-vous avec **Continue with GitHub**.
2. Touchez **Create app**, puis **Deploy a public app from GitHub**.
3. Repository : `votre-nom/over25`. Branch : `main`. Main file path : `app.py`.
4. Touchez **Deploy**. Patientez 2 à 5 minutes (installation automatique).
5. Votre adresse apparaît. Ouvrez-la.

### 4. L'avoir comme une vraie application sur Android
Dans Chrome, ouvrez l'adresse, touchez **⋮** puis **Ajouter à l'écran d'accueil**.

Bon à savoir : après quelques jours sans visite, l'application « dort ». Touchez simplement le bouton de réveil qui s'affiche (30 secondes).

## Méthode B : sur votre ordinateur

1. Installez Python depuis **python.org** (cochez **Add Python to PATH**).
2. Placez les 4 fichiers dans un dossier, ouvrez un terminal dans ce dossier.
3. Tapez : `pip install -r requirements.txt`
4. Tapez : `streamlit run app.py` (le navigateur s'ouvre).
5. Pour l'ouvrir sur votre téléphone, connecté au même Wi-Fi : tapez l'adresse « Network URL » affichée dans le terminal.

## Premier usage

1. Ouvrez l'onglet **Backtest**, choisissez 2 saisons, touchez **Lancer le backtest** (1 à 3 minutes).
2. Lisez le tableau « Qui prédit le mieux ? ». Si le marché à la clôture bat votre mix, n'attendez pas de miracle.
3. Regardez la **CLV** dans la simulation : positive et stable sur plusieurs centaines de paris = vrai signe d'avantage.
4. Seulement ensuite, utilisez l'onglet **Prédictions**.

## Problèmes fréquents

- **« Aucune donnée téléchargée »** : le site football-data.co.uk est momentanément indisponible, réessayez plus tard.
- **Erreur au déploiement** : vérifiez que `requirements.txt` et `app.py` sont à la racine du dépôt.
- **Pas de matchs affichés** : augmentez « Matchs des X prochains jours », ou la trêve internationale vide le calendrier.
- **Colonne « Cote » vide** : le site n'a pas encore publié de cotes pour ce match.

## À savoir

Le modèle ne connaît ni les blessés, ni les compositions, ni les xG. Une « value » affichée est une hypothèse à vérifier, pas une certitude. Ne misez que ce que vous pouvez perdre.
