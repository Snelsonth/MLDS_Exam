import os
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def generate_data():
    X, y = make_classification(
        n_samples=14000,          # Nombre d'exemples
        n_features=20,           # Nombre total de caractéristiques
        n_informative=10,        # Caractéristiques informatives
        n_redundant=5,           # Caractéristiques redondantes
        n_classes=3,             # Nombre de classes
        n_clusters_per_class=2,  # Nombre de clusters par classe
        class_sep=1.5,           # Séparation entre les classes
        random_state=42          # Reproductibilité
    )
    return X, y


# Séparer les données
def split_data(X, y):
    # Séparation initiale en 70% pour l'entraînement et 30% pour le test + validation
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Séparation des 30% restants en 20% pour les tests et 10% pour la validation
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42
    )  # 1/3 ≈ 10% pour la validation

    # Sauvegarder les données de validation pour un traitement futur
    save_validation_data(X_val, y_val)

    return X_train, X_test, y_train, y_test


# Générer les données
def generate_data():
    X, y = make_classification(
        n_samples=14000,          # Nombre d'exemples
        n_features=20,           # Nombre total de caractéristiques
        n_informative=10,        # Caractéristiques informatives
        n_redundant=5,           # Caractéristiques redondantes
        n_classes=3,             # Nombre de classes
        n_clusters_per_class=2,  # Nombre de clusters par classe
        class_sep=1.5,           # Séparation entre les classes
        random_state=42          # Reproductibilité
    )
    return X, y


# Séparer les données
def split_data(X, y):
    # Séparation initiale en 70% pour l'entraînement et 30% pour le test + validation
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Séparation des 30% restants en 20% pour les tests et 10% pour la validation
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42
    )  # 1/3 ≈ 10% pour la validation

    # Sauvegarder les données de validation pour un traitement futur
    save_validation_data(X_val, y_val)

    return X_train, X_test, y_train, y_test


# Sauvegarder les données de validation
def save_validation_data(X_val, y_val, filename="validation_data.pkl"):
    os.makedirs("output", exist_ok=True)  # Crée un dossier de sortie si nécessaire
    filepath = os.path.join("output", filename)
    joblib.dump((X_val, y_val), filepath)


# Entraînement du modèle
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


# Sauvegarder le modèle
def save_model(model, filename="classifier_model.pkl"):
    os.makedirs("output", exist_ok=True)  # Crée un dossier de sortie si nécessaire
    filepath = os.path.join("output", filename)
    joblib.dump(model, filepath)


# Fonction principale
def main():
    # Générer les données
    X, y = generate_data()

    # Séparer les données en entraînement et test
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Taille du jeu d'entraînement : {len(X_train)} exemples")
    print(f"Taille du jeu de test : {len(X_test)} exemples")

    # Entraîner le modèle
    model = train_model(X_train, y_train)

    # Évaluer les performances sur le jeu de test
    y_pred = model.predict(X_test)
    print("Rapport de classification (données de test) :")
    print(classification_report(y_test, y_pred))

    # Sauvegarder le modèle
    save_model(model)


# Exécution du script
if __name__ == "__main__":
    main()