import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Génération du jeu de données
X, y = make_blobs(n_samples=50, centers=1, cluster_std=0.1, random_state=0)

# Ajout d'un point anomalie
X[-1, :] = np.array([2, 2.55])

# Affichage du jeu de données
plt.scatter(X[:, 0], X[:, 1])
plt.title("Jeu de données avec anomalie")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


#2) 
from sklearn.ensemble import IsolationForest

# Sélection du modèle IsolationForest
model = IsolationForest(contamination=0.01)

# Entraînement du modèle
model.fit(X)

# Prédiction des anomalies
predictions = model.predict(X)

# Affichage des résultats
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm')
plt.title("Détection d'anomalies avec Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label='Prédiction (1: normal, -1: anomalie)')
plt.show()

# Modification de l'hyperparamètre contamination
model = IsolationForest(contamination=0.05)

# Entraînement du modèle
model.fit(X)

# Prédiction des anomalies
predictions = model.predict(X)

# Affichage des résultats
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm')
plt.title("Détection d'anomalies avec Isolation Forest (contamination=0.05)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label='Prédiction (1: normal, -1: anomalie)')
plt.show()









