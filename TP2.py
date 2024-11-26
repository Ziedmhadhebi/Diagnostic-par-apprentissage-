from sklearn.linear_model import LinearRegression
import numpy as np 
from sklearn.svm import SVR



import matplotlib.pyplot as plt



np.random.seed(0)
m=100
X = np.linspace(0,10,m).reshape(m,1)
Y = X + np.random.random_sample((m,1))

 


plt.scatter(X, Y)
plt.title("Nuage de points ")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


model = LinearRegression()  
model.fit(X, Y)
score = model.score(X, Y)
print(f"Score pour la régression linéaire : {score:.2f}")

Y_pred = model.predict(X)
plt.scatter(X, Y, color='blue', label='Données réelles')
plt.plot(X, Y_pred, color='red', label="Modèle prédit")
plt.title("Régression linéaire")
plt.xlabel("X")
plt.ylabel("Y") 
plt.legend()
plt.show()


X2 = np.linspace(0,10,m).reshape(m,1) 
Y2 = X2**2 + np.random.randn(m,1) 
plt.scatter(X2,Y2)
plt.title("Deuxieme Nuage de points ")
plt.xlabel("X2")
plt.ylabel("Y2")
plt.show()

model = LinearRegression()
model.fit(X2, Y2)
score1 = model.score(X2, Y2)
print(f"Score pour le deuxieme nuage : {score:.2f}")

for m in [100, 1000, 10000]:
    X1 = np.linspace(0, 10, m).reshape(m, 1)
    Y1 = X1 ** 2 + np.random.random((m, 1))
    model.fit(X1, Y1)
    plt.scatter(X1, Y1)
    plt.plot(X1, model.predict(X1), color='red')
    plt.xlabel('X1')
    plt.ylabel('Y2')
    plt.title(f'Régression linéaire avec m = {m}')
    plt.show()

#8/ 

model_SVR = SVR(C=100)
model_SVR.fit(X2, Y2.ravel())  # .ravel() pour convertir Y en vecteur 1D
score_SVR = model_SVR.score(X2, Y2.ravel())
print(f" pour SVR avec C=100 : {score_SVR:}")

for m in [100, 1000, 10000]:
    X2 = np.linspace(0, 10, m).reshape(m, 1)
    Y2 = X2 ** 2 + np.random.random((m, 1))
    model_SVR = SVR(C=100)
    model_SVR.fit(X2, Y2.ravel()) 
    plt.scatter(X2, Y2)
    plt.plot(X2, model_SVR.predict(X2), color='red')
    plt.xlabel('X2')
    plt.ylabel('Y2')
    plt.title(f'SVR avec m = {m}')
    plt.show()




 


