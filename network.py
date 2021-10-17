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


nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes=[15,5], activation='relu',
                                    algorithm='random_hill_climb', max_iters=2000,
                                    bias=True, is_classifier=True, learning_rate=0.9,
                                    early_stopping=True, clip_max=5, max_attempts=100,
                                    random_state=1,curve=True) # 0.8883 0.8203

nn_model_sa = mlrose.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                   algorithm='simulated_annealing', max_iters=2000,
                                   bias=True, is_classifier=True, learning_rate=0.6,
                                   early_stopping=True, clip_max=5, max_attempts=100,
                                   random_state=1,curve=True,schedule=mlrose.ExpDecay(init_temp=1.5, exp_const=0.05, min_temp=0.001))

nn_model_ga = mlrose.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                   algorithm='genetic_alg', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=1,
                                   early_stopping=True, clip_max=5, max_attempts=50, 
                                   random_state=1,curve=True,pop_size=300,mutation_prob=0.1) #  0.8191, 0.7864

nn_model_gd = mlrose.NeuralNetwork(hidden_nodes=[15,5], activation='relu',
                                   algorithm='gradient_descent', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=0.00055,
                                   early_stopping=True, clip_max=5, max_attempts=1000,
                                   random_state=1,curve=True) # 0.95995, 0.8398

nn_model_rhc.fit(x_train, y_train)
nn_model_sa.fit(x_train, y_train)
nn_model_ga.fit(x_train, y_train)
nn_model_gd.fit(x_train, y_train)

plt.plot(nn_model_rhc.fitness_curve[:,0])
plt.plot(nn_model_sa.fitness_curve[:,0])
plt.plot(nn_model_ga.fitness_curve[:,0])
plt.plot(-nn_model_gd.fitness_curve)
plt.legend(["RHC","SA","GA","GD"])
plt.xlim(0,600)
plt.ylim(0,5)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("Comparison of Loss Curves of Four Algorithms")
plt.show()