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

"""SA tune plot"""

curves_sa = []
for decay in [mlrose.GeomDecay(init_temp=1.0, decay=0.0001, min_temp=0.001), # 0.8859, 0.8155
              mlrose.ArithDecay(init_temp=1.0, decay=0.01, min_temp=0.001), #  0.8980, 0.8252
              mlrose.ExpDecay(init_temp=1.5, exp_const=0.05, min_temp=0.001)]: # 0.90, 0.82
    nn_model_sa = mlrose.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                   algorithm='simulated_annealing', schedule = decay,
                                   max_iters=2000,
                                   bias=True, is_classifier=True, learning_rate=0.6,
                                   early_stopping=True, clip_max=5, max_attempts=100,
                                   random_state=1,curve=True)
    t = time.time()
    nn_model_sa.fit(x_train, y_train)

    y_train_pred = nn_model_sa.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print("========================================================")
    print("Train accuracy: {}".format(y_train_accuracy))

    y_test_pred = nn_model_sa.predict(x_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test accuracy: {}".format(y_test_accuracy))
    print("Time needed: {}".format(time.time()-t))
    curves_sa.append(nn_model_sa.fitness_curve)
    print("Loss:",nn_model_sa.fitness_curve[-1,0])
    print("Iteration:",nn_model_sa.fitness_curve[-1,1]-nn_model_sa.fitness_curve[0,1])


for i in range(3):
    plt.plot(curves_sa[i][:,0])
    print("Loss:",curves_sa[i][-1,0],"iterations:",len(curves_sa[i][:,0]))
plt.legend(["GeomDecay (init_temp=1, decay=0.001)",
            "ArithDecay (init_temp=1, decay=0.01)",
            "ExpDecay (init_temp=1.5, exp_const=0.05)"])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("Loss of SA with different cooling schedules")
plt.xlim(0,2000)
plt.show()