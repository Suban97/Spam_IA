# ==========================
# Détecteur de spam complet
# ==========================

# 1️⃣ Import des bibliothèques
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib  # pour sauvegarder le modèle

# 2️⃣ Jeu de données initial
emails = [
    "Gagnez de l'argent facilement, cliquez ici",
    "Votre facture est disponible en ligne",
    "Vous avez gagné un iPhone gratuit",
    "Rendez-vous demain à 15h pour la réunion",
    "Offre exclusive, cliquez maintenant"
]

labels = ["spam", "ham", "spam", "ham", "spam"]  # spam ou ham

# 3️⃣ Transformation du texte en chiffres
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# 4️⃣ Création et entraînement du modèle
model = MultinomialNB()
model.fit(X, labels)

# 5️⃣ Tester le modèle avec un nouvel email
new_email = ["Cliquez vite pour recevoir votre cadeau"]
X_new = vectorizer.transform(new_email)
prediction = model.predict(X_new)
print("Prédiction avant réentraînement :", prediction)  # ['spam'] ou ['ham']

# 6️⃣ Ajouter le nouvel email à l'entraînement
emails.append(new_email[0])
labels.append("spam")  # on lui dit que c'est bien spam

# 7️⃣ Réentraîner le modèle pour apprendre le nouveau mot
X = vectorizer.fit_transform(emails)  # le vectorizer apprend les nouveaux mots
model.fit(X, labels)

# 8️⃣ Tester à nouveau
prediction_after = model.predict(vectorizer.transform(new_email))
print("Prédiction après réentraînement :", prediction_after)  # ['spam']

# 9️⃣ Sauvegarder le modèle et le vectorizer pour usage futur
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# ==========================
# Le script est prêt !
# Tu peux maintenant charger le modèle et le vectorizer ailleurs :
# model = joblib.load("spam_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")
# ==========================