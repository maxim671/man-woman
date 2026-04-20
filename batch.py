import numpy as np
import pandas as pd

df = pd.read_csv('iris.csv')
weight_raw = np.array(df['Вес']).reshape(-1, 1)
height_raw = np.array(df['Рост']).reshape(-1, 1)
y_true = np.array(df['Y_True']).reshape(-1, 1)


weight = (weight_raw - weight_raw.mean()) / weight_raw.std()
height = (height_raw - height_raw.mean()) / height_raw.std()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(X, W1, W2, B1, B2, Y_true, learning_rate = 0.01):
    for i in range(5000):

        H = X @ W1 + B1
        Z = sigmoid(H)
        S = Z @ W2 + B2
        Y = sigmoid(S)
        L = np.linalg.norm(Y_true - Y) / X.shape[0]


        dl_dy_pred = -1 * (Y_true - Y)
        dy_pred_ds = Y * (1 - Y)
        ds_dw2 = Z.T

        delta2 = dl_dy_pred * dy_pred_ds
        dl_dw2 = ds_dw2 @ delta2


        delta1 = (delta2 @ W2.T) * Z * (1 - Z)
        dl_dw1 = X.T @ delta1

        dl_db2 = np.sum(delta2, axis=0, keepdims=True)
        dl_db1 = np.sum(delta1, axis=0, keepdims=True)

        W1 -= learning_rate * dl_dw1
        W2 -= learning_rate * dl_dw2
        B1 -= learning_rate * dl_db1
        B2 -= learning_rate * dl_db2

    return W1, W2

X = np.hstack((weight, height))
W1 = np.random.rand(2,2)
W2 = np.random.rand(2,1)
B1 = np.random.rand(1,2)
B2 = np.random.rand(1,1)


W1, W2 = train(X, W1, W2, B1, B2, y_true)



def predict(weight, height):
    weight = (weight - weight_raw.mean()) / weight_raw.std()
    height = (height - height_raw.mean()) / height_raw.std()
    X = np.array([[weight, height]])   # форма (1,2)
    if sigmoid(sigmoid(X @ W1 + B1) @ W2 + B2).item() > 0.5:
        return "Женщина"
    return "Мужчина"

print(predict(87, 175))





