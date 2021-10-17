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



"""GA tune plot"""

curves_ga_prob = []
for prob in [0.1,0.2,0.3]:
    nn_model_ga = mlrose.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                   algorithm='genetic_alg', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=1,
                                   early_stopping=True, clip_max=5, max_attempts=50, #50, 
                                   random_state=1,curve=True,
                                   pop_size=300,mutation_prob=prob) #  0.8191, 0.7864
    t = time.time()
    nn_model_ga.fit(x_train, y_train)

    y_train_pred = nn_model_ga.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print("========================================================")
    print("Train accuracy: {}".format(y_train_accuracy))

    y_test_pred = nn_model_ga.predict(x_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test accuracy: {}".format(y_test_accuracy))
    print("Time needed: {}".format(time.time()-t))
    curves_ga_prob.append(nn_model_ga.fitness_curve)
    print("Loss:",nn_model_ga.fitness_curve[-1,0])
    print("Iteration:",nn_model_ga.fitness_curve[-1,1]-nn_model_ga.fitness_curve[0,1])

curves_ga_pop = []
for pop in [100,200,300]:
    nn_model_ga = mlrose.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                   algorithm='genetic_alg', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=1,
                                   early_stopping=True, clip_max=5, max_attempts=50,
                                   random_state=1,curve=True,
                                   pop_size=pop,mutation_prob=0.1) #  0.8191, 0.7864
    t = time.time()
    nn_model_ga.fit(x_train, y_train)

    y_train_pred = nn_model_ga.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print("========================================================")
    print("Train accuracy: {}".format(y_train_accuracy))

    y_test_pred = nn_model_ga.predict(x_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test accuracy: {}".format(y_test_accuracy))
    print("Time needed: {}".format(time.time()-t))
    curves_ga_pop.append(nn_model_ga.fitness_curve)
    print("Loss:",nn_model_ga.fitness_curve[-1,0])
    print("Iteration:",nn_model_ga.fitness_curve[-1,1]-nn_model_ga.fitness_curve[0,1])


plt.figure("Loss of GA with different mutation rates")
for i in range(3):
    plt.plot(curves_ga_prob[i][:,0])
    print("Loss:",curves_ga_prob[i][-1,0],"iterations:",len(curves_ga_prob[i][:,0]))
plt.legend(["0.1","0.2","0.3"])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("Loss of GA with different mutation rates")
#plt.xlim(0,2000)

plt.figure("Loss of GA with different population")
for i in range(3):
    plt.plot(curves_ga_pop[i][:,0])
    print("Loss:",curves_ga_pop[i][-1,0],"iterations:",len(curves_ga_pop[i][:,0]))
plt.legend(["100","200","300"])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("Loss of GA with different population")
#plt.xlim(0,2000)

plt.show()

"""
results:
# varying mutation rates

= 0.1 (this one)=======================================================
Train accuracy: 0.8288834951456311
Test accuracy: 0.8106796116504854
Time needed: 314.79906725883484

= 0.2 =======================================================
Train accuracy: 0.7754854368932039
Test accuracy: 0.7815533980582524
Time needed: 192.51200151443481
= 0.3 =======================================================
Train accuracy: 0.8009708737864077
Test accuracy: 0.7281553398058253
Time needed: 121.52951645851135

# varying population
= 100 =======================================================
Train accuracy: 0.8155339805825242
Test accuracy: 0.8009708737864077
Time needed: 62.34781289100647
= 200 =======================================================
Train accuracy: 0.8191747572815534
Test accuracy: 0.7864077669902912
Time needed: 80.07752108573914
= 300 (this one)=======================================================
Train accuracy: 0.8288834951456311
Test accuracy: 0.8106796116504854
Time needed: 200.01593685150146

# varying mutation rates
Loss: 2.648126954180265 iterations: 170 (this one)
Loss: 3.281434460600253 iterations: 109
Loss: 3.8177082467147563 iterations: 90

# varying population
Loss: 3.175500039622265 iterations: 168
Loss: 3.159353745392628 iterations: 116
Loss: 2.648126954180265 iterations: 170 (this one)
"""