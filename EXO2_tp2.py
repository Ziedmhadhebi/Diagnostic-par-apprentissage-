import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

titanic = sns.load_dataset('titanic')
print(titanic.shape)
print(titanic.head())

titanic = titanic[['survived', 'pclass', 'sex', 'age']] # Sélection des colonnes

titanic.dropna(axis=0, inplace=True) # Suppression des données manquantes

titanic['sex'].replace(['male', 'female'], [0, 1], inplace=True) #  Numérisation du jeu de données
print(titanic.head())




Y = titanic['survived']
X = titanic.drop('survived', axis=1)


model = KNeighborsClassifier()

model.fit(X, Y)

score = model.score(X, Y)
print(f"Score du modèle : {score}")
 

def survie(model, pclass=3, sex=0, age=26):
    x = np.array([pclass, sex, age]).reshape(1, 3)
    prediction = model.predict(x)
    probabilities = model.predict_proba(x)
    print(f"Prédiction de survie : {prediction[0]}")
    print(f"Probabilités : {probabilities}")
    return prediction, probabilities


survie(model, pclass=3, sex=0, age=26)

#8 
best_score = 0 
best_n_neighbors= 0 

for n_neighbors in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, Y)
    score = model.score(X, Y)
    print(f'Score avec {n_neighbors} voisins : {score}')
    if score > best_score:
        best_score = score
        best_n_neighbors = n_neighbors

print(f'Meilleur nombre de voisins : {best_n_neighbors} avec un score de {best_score}')






    