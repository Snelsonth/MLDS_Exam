Projet MLDS_Exam : Classification et Prédiction avec Docker
Ce projet utilise Docker pour entraîner un modèle de classification avec Scikit-learn et effectuer des prédictions sur des données de validation. Il repose sur deux scripts Python principaux :

train_classifier.py : Entraîne le modèle et sauvegarde le modèle et les données de validation.
predict_classification.py : Charge le modèle et les données de validation pour effectuer des prédictions.
Prérequis
Docker doit être installé sur votre machine.
Docker Compose est recommandé pour simplifier l’exécution.
Étapes
Cloner le projet Clonez le repository Git ou téléchargez les fichiers et placez-les dans un répertoire local :

bash
Copier le code
git clone https://github.com/<your_github_username>/MLDS_Exam.git
cd MLDS_Exam
Construire les images Docker

Construisez l’image pour l’entraînement :
bash
Copier le code
docker build -t train_model -f Dockerfile.train .
Construisez l’image pour la prédiction :
bash
Copier le code
docker build -t predict_model -f Dockerfile.predict .
Exécuter les conteneurs

Pour entraîner le modèle, exécutez :
bash
Copier le code
docker run -v [renseignez votre chemin]/output:/MLDS_Exam/output train_model
Pour effectuer des prédictions, exécutez :
bash
Copier le code
docker run -v [renseignez votre chemin]/output:/MLDS_Exam/output predict_model
Les fichiers générés seront sauvegardés dans le répertoire local output.

Utiliser Docker Compose (optionnel) Si vous préférez une exécution automatisée, utilisez Docker Compose :

Pour construire et exécuter les conteneurs :
bash
Copier le code
docker-compose up --build
Pour exécuter une étape spécifique, par exemple l’entraînement :
bash
Copier le code
docker-compose run train_model
Résultats

classifier_model.pkl : Modèle entraîné.
validation_data.pkl : Données de validation utilisées pour la prédiction.
Dépendances
Les bibliothèques Python nécessaires sont spécifiées dans requirements.txt :

scikit-learn
joblib
