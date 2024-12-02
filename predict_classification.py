# Fichier pour prédire et valider le modèle

# Importations nécessaires
import joblib
from sklearn.metrics import classification_report


model = joblib.load("output/classifier_model.pkl")
X_val, y_val = joblib.load("output/validation_data.pkl")


y_pred_val = model.predict(X_val)

# Évaluer les performances du modèle
# Calcul et affichage du rapport de classification pour les données de validation
print("Rapport de classification (validation) :")
print(classification_report(y_val, y_pred_val))
