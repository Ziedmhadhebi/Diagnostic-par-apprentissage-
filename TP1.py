# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Chargement du jeu de données Iris
iris = load_iris()
X = iris.data
y = iris.target

# Étape 1 : Découpage des données
# 60% pour l'entraînement, 40% pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Découpage des 80% d'entraînement : 70% pour l'entraînement final, 30% pour la validation
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=5)

# Étape 2 : Comparaison des performances pour k=3 et k=4
# Modèle avec k=3
model_k3 = KNeighborsClassifier(n_neighbors=3)
model_k3.fit(X_train_final, y_train_final)
val_score_k3 = accuracy_score(y_val, model_k3.predict(X_val))

# Modèle avec k=4
model_k4 = KNeighborsClassifier(n_neighbors=4)
model_k4.fit(X_train_final, y_train_final)
val_score_k4 = accuracy_score(y_val, model_k4.predict(X_val))

# Affichage des scores pour la validation
print(f"Score validation avec k=3 : {val_score_k3}")
print(f"Score validation avec k=4 : {val_score_k4}")
    
# Étape 3 : Évaluation finale sur le jeu de test
# Entraînement du modèle final avec le meilleur k
final_model = KNeighborsClassifier(n_neighbors=3)  # Choisir k=3 ou k=4 selon les résultats précédents
final_model.fit(X_train_final, y_train_final)
final_score = accuracy_score(y_test, final_model.predict(X_test))

print(f"Score final sur le jeu de test : {final_score}")

# Étape 4 : Validation croisée pour k=3 et k=4
cv_score_k3 = cross_val_score(KNeighborsClassifier(n_neighbors=3), X_train_final, y_train_final, cv=5, scoring='accuracy').mean()
cv_score_k4 = cross_val_score(KNeighborsClassifier(n_neighbors=4), X_train_final, y_train_final, cv=5, scoring='accuracy').mean()

print(f"Score moyen en validation croisée pour k=3 : {cv_score_k3}")
print(f"Score moyen en validation croisée pour k=4 : {cv_score_k4}")

# Étape 5 : Validation croisée pour plusieurs valeurs de k
val_scores = []
for k in range(1, 21):  # Tester k de 1 à 20
    score = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_final, y_train_final, cv=5, scoring='accuracy').mean()
    val_scores.append(score)

# Tracer les scores
plt.plot(range(1, 21), val_scores)
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Score moyen (validation croisée)')
plt.title('Validation croisée pour différents k')
plt.show()

# Meilleur k
best_k = range(1, 21)[val_scores.index(max(val_scores))]
print(f"Meilleur k : {best_k}, avec un score moyen de {max(val_scores):.2f}")

# Étape 6 : Courbes d'apprentissage avec validation_curve
k_range = np.arange(1, 21)
train_scores, val_scores_curve = validation_curve(
    KNeighborsClassifier(),
    X_train_final,
    y_train_final,
    param_name='n_neighbors',
    param_range=k_range,
    cv=5
)

# Moyennes des scores
train_scores_mean = train_scores.mean(axis=1)
val_scores_mean = val_scores_curve.mean(axis=1)

# Tracer les courbes
plt.plot(k_range, train_scores_mean, label='Entraînement')
plt.plot(k_range, val_scores_mean, label='Validation')
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Score')
plt.title('Courbes d’apprentissage pour différents k')
plt.legend()
plt.show()


#exo4 
from sklearn.model_selection import GridSearchCV


param_grid ={'n_neighbors':np.arange(1,50),'metric': ['euclidian','manhattan']}

Grid=GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)


X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=5)


Grid.fit(X_train,Y_train)
# Affichage des meilleurs paramètres
print("Meilleurs paramètres :", Grid.best_params_)

# Affichage du meilleur score de validation croisée
print("Meilleur score de validation :", Grid.best_score_)


# Sauvegarde du meilleur modèle
best_model = Grid.best_estimator_

print("Meilleur modèle :", best_model)


# Évaluation des performances sur le jeu de test
test_score = best_model.score(X_test, y_test)
print("Score sur le jeu de test :", test_score)

from sklearn.metrics import confusion_matrix

# Calcul de la matrice de confusion
conf_matrix = confusion_matrix(y_test, best_model.predict(X_test))

print("Matrice de confusion :\n", conf_matrix)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve



# Génération des courbes d'apprentissage
N, train_scores, val_scores = learning_curve(
    best_model,
    X_train,  # Données d'entraînement
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 lots de tailles croissantes, de 10% à 100% du jeu d'entraînement
    cv=5  # Cross-validation à 5 folds
)

# Moyennes des scores
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

# Visualisation des courbes
plt.figure(figsize=(10, 6))

# Courbe pour le jeu d'entraînement
plt.plot(N, train_mean, label="Entraînement", color="blue", marker="o")

# Courbe pour le jeu de validation
plt.plot(N, val_mean, label="Validation", color="orange", marker="o")

# Légendes et titres
plt.title("Courbes d'apprentissage")
plt.xlabel("Taille des données d'entraînement")
plt.ylabel("Score (Accuracy)")
plt.legend()
plt.grid()
plt.show()

# Interprétation
print("Interprétation des courbes :")
if val_mean[-1] < 1.0 and train_mean[-1] < 1.0:
    print("- Les performances semblent plafonner, suggérant que le modèle a atteint ses limites d'apprentissage.")
    print("- Si l'écart entre entraînement et validation est faible, il n'y a pas de sur-apprentissage.")
    print("- Si l'écart entre entraînement et validation est grand, il peut y avoir un sous-apprentissage ou un sur-ajustement.")
else:
    print("- Les performances du modèle pourraient encore s'améliorer en collectant davantage de données.")

