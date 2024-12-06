import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Génération du jeu de données
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)

# Affichage du jeu de données
plt.scatter(X[:, 0], X[:, 1])
plt.title("Jeu de données généré")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


#2)

from sklearn.cluster import KMeans

# Définition du modèle KMeans
model = KMeans(n_clusters=3)

#3) 

# Entraînement du modèle
model.fit(X) 


#4) 

# Prédiction des clusters
predictions = model.predict(X)

# Affichage des prédictions
plt.scatter(X[:, 0], X[:, 1], c=predictions)
plt.title("Prédictions du modèle KMeans")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

#5)

# Affichage des centroïdes
centroids = model.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=predictions, label='Données')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroïdes')
plt.title("Centroïdes du modèle KMeans")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()



#7) 
# Calcul du coût
inertia = model.inertia_
print("Coût (inertia):", inertia)


#8)

# Évaluation du modèle
score = model.score(X)
print("Score du modèle:", score)

#9)

# Initialisation d'une liste vide pour stocker les valeurs d'inertia
inertia = []

# Définition de la plage de valeurs pour le nombre de clusters à tester
K_range = range(1, 20)

# Boucle sur chaque valeur de k dans la plage définie
for k in K_range:
    # Création et entraînement du modèle KMeans avec k clusters
    model = KMeans(n_clusters=k).fit(X)

    # Ajout de l'inertia du modèle à la liste
    inertia.append(model.inertia_)

# Tracé de la courbe de l'inertia en fonction du nombre de clusters
plt.plot(K_range, inertia)

# Ajout d'une étiquette pour l'axe des abscisses
plt.xlabel('Nombre de clusters')

# Ajout d'une étiquette pour l'axe des ordonnées
plt.ylabel('Coût du modèle (Inertia)')

# Ajout d'un titre au graphique
plt.title('Méthode du coude')

# Affichage du graphique
plt.show()


