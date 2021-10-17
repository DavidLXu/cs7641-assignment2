import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, cross_val_score, validation_curve
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


df_conc = pd.read_csv("concrete.csv")

# original concrete data
inputs_o = df_conc.drop('strength',axis='columns')
targets_o = df_conc['strength']
# making a binary classification
# >=35 -> 1
# <35 -> 0
inputs = inputs_o.values
targets= [0 if strength < 35 else 1  for strength in targets_o.values]

x_train,x_test,y_train,y_test = train_test_split(inputs,targets,random_state=1,test_size=0.2)


x_train = scale(x_train)
x_test = scale(x_test)

"""GD tune plot"""
curves_gd = []

for rate in [0.005,0.001,0.00055,0.0001]:
    t = time.time()
    nn_model_gd = mlrose.NeuralNetwork(hidden_nodes=[15,5], activation='relu',
                                   algorithm='gradient_descent', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=rate,
                                   early_stopping=True, clip_max=5, max_attempts=1000,
                                   random_state=1,curve=True) # 0.95995, 0.8398
    nn_model_gd.fit(x_train, y_train)

    y_train_pred = nn_model_gd.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print("========================================================")
    print("Train accuracy: {}".format(y_train_accuracy))

    y_test_pred = nn_model_gd.predict(x_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test accuracy: {}".format(y_test_accuracy))
    print("Time needed: {}".format(time.time()-t))
    curves_gd.append(nn_model_gd.fitness_curve)
    print("Loss:",nn_model_gd.fitness_curve[-1])
    print("Iteration:",len(nn_model_gd.fitness_curve))

plt.figure("Loss of GD with different learning rates")
for i in range(4):
    plt.plot(-curves_gd[i])
    print("Loss:",curves_gd[i][-1],"iterations:",len(curves_gd[i]))
plt.legend([0.005,0.001,0.00055,0.0001])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("Loss of GD with different learning rates")
plt.xlim(-10,1000)
plt.ylim(0,2.5)
plt.show()