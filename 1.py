import numpy as np
import pandas as pd

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def train(self, weight, height, y_true):
        TrainingConst = 0.01
        for epoch in range(5000):
            y_predicted = []
            for i in range(len(weight)):
                # считается для одного примера
                h1 = sigmoid(weight[i] * self.w1 + height[i] * self.w3 + self.b1)
                h2 = sigmoid(weight[i] * self.w2 + height[i] * self.w4 + self.b2)
                y_pred = sigmoid(h1 * self.w5 + h2 * self.w6 + self.b3)
                y_predicted.append(y_pred.item())
                dl_dypred = 2 * (y_pred - y_true[i])
                dypred_dh1 = (y_pred * (1 - y_pred)) * (self.w5)
                dypred_dh2 = (y_pred * (1 - y_pred)) * (self.w6)
                dh1_dw1 = (h1 * (1 - h1)) * (weight[i])
                dh2_dw2 = (h2 * (1 - h2)) * (weight[i])
                dh1_dw3 = (h1 * (1 - h1)) * (height[i])
                dh2_dw4 = (h2 * (1 - h2)) * (height[i])
                dypred_dw5 = y_pred * (1 - y_pred) * (h1)
                dypred_dw6 = y_pred * (1 - y_pred) * (h2)
                dypred_db3 = y_pred * (1 - y_pred)
                dh1_db1 = h1 * (1 - h1)
                dh2_db2 = h2 * (1 - h2)

                dl_dw1 = dl_dypred * dypred_dh1 * dh1_dw1
                dl_dw2 = dl_dypred * dypred_dh2 * dh2_dw2
                dl_dw3 = dl_dypred * dypred_dh1 * dh1_dw3
                dl_dw4 = dl_dypred * dypred_dh2 * dh2_dw4
                dl_dw5 = dl_dypred * dypred_dw5
                dl_dw6 = dl_dypred * dypred_dw6
                dl_db1 = dl_dypred * dypred_dh1 * dh1_db1
                dl_db2 = dl_dypred * dypred_dh2 * dh2_db2
                dl_db3 = dl_dypred * dypred_db3

                self.w1 -= TrainingConst * dl_dw1
                self.w2 -= TrainingConst * dl_dw2
                self.w3 -= TrainingConst * dl_dw3
                self.w4 -= TrainingConst * dl_dw4
                self.w5 -= TrainingConst * dl_dw5
                self.w6 -= TrainingConst * dl_dw6
                self.b1 -= TrainingConst * dl_db1
                self.b2 -= TrainingConst * dl_db2
                self.b3 -= TrainingConst * dl_db3

    def predict(self, weight_prob, height_prob):
        x1 = (weight_prob - weight_neobr.mean()) / weight_neobr.std()
        x2 = (height_prob - height_neobr.mean()) / height_neobr.std()
        h1 = sigmoid(x1* self.w1 + x2 * self.w3 + self.b1)
        h2 = sigmoid(x1 * self.w2 + x2 * self.w4 + self.b2)
        y_pred = sigmoid(h1 * self.w5 + h2 * self.w6 + self.b3)
        return y_pred

df = pd.read_csv('iris.csv')
weight_neobr = np.array(df['Вес'])
height_neobr = np.array(df['Рост'])
y_true = np.array(df['Y_True'])

weight = (weight_neobr - weight_neobr.mean()) / weight_neobr.std()
height = (height_neobr - height_neobr.mean()) / height_neobr.std()


neuron = NeuralNetwork()
neuron.train(weight, height, y_true)

print(neuron.predict(0,175))

