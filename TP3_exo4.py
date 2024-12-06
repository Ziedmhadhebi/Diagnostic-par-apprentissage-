from sklearn.datasets import load_digits


# Téléchargement du jeu de données
digits = load_digits()

# Récupération des images et des targets
images = digits.images
X = digits.data
y = digits.target

# Affichage des dimensions du jeu de données
print("Dimensions du jeu de données X:", X.shape)
print("Dimensions des cibles y:", y.shape)


#B 
from sklearn.decomposition import PCA



# Sélection du modèle PCA avec 2 composantes
model = PCA(n_components=2)

# Entraînement du modèle et réduction des données
X_reduced = model.fit_transform(X)
 
# Affichage des dimensions des données réduites
print("Dimensions des données réduites X_reduced:", X_reduced.shape)


import matplotlib.pyplot as plt

# Affichage des composantes principales
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title("Composantes principales des données réduites")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.show()

# Ajout de couleurs au graphique en fonction des cibles
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.title("Composantes principales des données réduites avec couleurs")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.show()


# Ajout d'une barre des couleurs
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.colorbar(label='Cible')
plt.title("Composantes principales des données réduites avec barre des couleurs")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.show()

# Affichage des composantes du modèle PCA
print("Forme des composantes du modèle PCA:", model.components_.shape)



#PARTIE C : 

# Sélection du modèle PCA avec le même nombre de dimensions que X
model = PCA(n_components=X.shape[1])

# Entraînement du modèle et réduction des données
X_reduced = model.fit_transform(X)

# Affichage du pourcentage de variance expliquée par chaque composante
explained_variance_ratio = model.explained_variance_ratio_
print("Pourcentage de variance expliquée par chaque composante:", explained_variance_ratio)


import numpy as np

# Calcul de la somme cumulée de variance
cumulative_variance = np.cumsum(explained_variance_ratio)
print("Somme cumulée de variance:", cumulative_variance)

# Tracé du pourcentage de variance cumulée
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.title("Pourcentage de variance cumulée")
plt.xlabel("Nombre de composantes")
plt.ylabel("Pourcentage de variance cumulée")
plt.grid()
plt.show()


# Détermination du nombre de composantes pour atteindre 99% de variance
n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
print("Nombre de composantes pour atteindre 99% de variance:", n_components_99)

# Sélection du modèle PCA avec le nombre de composantes pour atteindre 99% de variance
model = PCA(n_components=n_components_99)

# Entraînement du modèle et réduction des données
X_reduced = model.fit_transform(X)

# Décompression des données
X_recovered = model.inverse_transform(X_reduced)

# Affichage d'une image décompressée
plt.imshow(X_recovered[0].reshape((8, 8)), cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Image décompressée")
plt.show()


from sklearn.preprocessing import MinMaxScaler

# Mise à échelle des données
scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(X)

# Sélection du modèle PCA avec 40% de variance préservée
model = PCA(n_components=0.40)

# Entraînement du modèle et réduction des données
model.fit(data_rescaled)
X_reduced = model.transform(data_rescaled)

# Vérification du nombre de composantes utilisées
print("Nombre de composantes utilisées pour 40% de variance:", model.n_components_)
X_recovered2 = model.inverse_transform(X_reduced)
plt.imshow(X_recovered2[0].reshape((8, 8)), cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Image décompressée avec  40 % de variance ")
plt.show()











