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

"""RHC tune plot"""
nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes=[15,5], activation='relu',
                                    algorithm='random_hill_climb', max_iters=4000,
                                    bias=True, is_classifier=True, learning_rate=0.9,
                                    early_stopping=True, clip_max=5, max_attempts=100,
                                    curve=True) # 0.8883 0.8203
curves_rhc = []
model_rhc = []
for i in range(5):
    t = time.time()
    nn_model_rhc.fit(x_train, y_train)

    y_train_pred = nn_model_rhc.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print("========================================================")
    print("Train accuracy: {}".format(y_train_accuracy))

    y_test_pred = nn_model_rhc.predict(x_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test accuracy: {}".format(y_test_accuracy))
    print("Time needed: {}".format(time.time()-t))
    curves_rhc.append(nn_model_rhc.fitness_curve)
    model_rhc.append(nn_model_rhc)
    print("Loss:",nn_model_rhc.fitness_curve[-1,0])
    print("Iteration:",nn_model_rhc.fitness_curve[-1,1]-nn_model_rhc.fitness_curve[0,1])

for i in range(5):
    plt.plot(curves_rhc[i][:,0])
    print("Loss:",curves_rhc[i][-1,0],"iterations:",len(curves_rhc[i][:,0]))
plt.legend([1,2,3,4,5])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("Loss of RHC with different restarts")
plt.xlim(0,2000)
plt.show()