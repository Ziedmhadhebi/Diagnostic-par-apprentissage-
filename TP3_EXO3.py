#1)
from sklearn.datasets import load_digits


#2) 
# Téléchargement du jeu de données
digits = load_digits()

# Récupération des images et des targets
images = digits.images

#3) 
X = digits.data
y = digits.target

#4) 

# Taille de X
print("Taille de X:", X.shape)

# Taille de chaque image
print("Taille de chaque image:", images.shape[1:])

import matplotlib.pyplot as plt

# Visualisation de quelques images
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, image, label in zip(axes, images[0:5], y[0:5]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('%i' % label)
plt.show()


from sklearn.ensemble import IsolationForest

# Sélection du modèle IsolationForest
model = IsolationForest(random_state=0, contamination=0.02)

#6) 
# Entraînement du modèle
model.fit(X)

#7) 
# Évaluation du modèle
predictions = model.predict(X)

# Interprétation des résultats
outliers = predictions == -1
print("Nombre d'anomalies détectées:", outliers.sum())

import matplotlib.pyplot as plt
#9/ 

# Sélectionnez la première image détectée comme anomalie
first_outlier_image = images[outliers][0]

# Affichage de la première image détectée comme anomalie
plt.imshow(first_outlier_image, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Première image détectée comme anomalie")
plt.show()


#10/ 
from sklearn.ensemble import IsolationForest

# Modification de l'hyperparamètre contamination
model = IsolationForest(random_state=0, contamination=0.05)

# Entraînement du modèle
model.fit(X)

# Évaluation du modèle
predictions = model.predict(X)

# Interprétation des résultats
outliers = predictions == -1
print("Nombre d'anomalies détectées avec contamination=0.05:", outliers.sum())


