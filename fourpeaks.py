import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

fitness = mlrose.FourPeaks(t_pct=0.1)
problem = mlrose.DiscreteOpt(length=56, fitness_fn=fitness, maximize=True, max_val=2)

print("Running RHC...")
rhc = mlrose.RHCRunner(problem=problem,
                       experiment_name="FourPeaks_RHC",
                       output_directory="./",
                       seed=None,
                       iteration_list=2 ** np.arange(24),
                       max_attempts=200,
                       restart_list=[5])
rhc_run_stats, rhc_run_curves = rhc.run()

print("Running SA...")
sa = mlrose.SARunner(problem=problem,
                     experiment_name="FourPeaks_SA",
                     output_directory="./",
                     seed=None,
                     iteration_list=2 ** np.arange(13),
                     max_attempts=500,
                     temperature_list=[0.5,1,5,10],
                     decay_list=[mlrose.ExpDecay])
sa_run_stats, sa_run_curves = sa.run()

print("Running GA...")
ga = mlrose.GARunner(problem=problem,
                     experiment_name="FourPeaks_GA",
                     output_directory="./",
                     seed=None,
                     iteration_list=2 ** np.arange(13),
                     max_attempts=1000,
                     population_sizes=[20,50,100,200],
                     mutation_rates=[0.1,0.2,0.3])
ga_run_stats, ga_run_curves = ga.run()

print("Running MIMIC...")
mimic = mlrose.MIMICRunner(problem=problem,
                           experiment_name="FourPeaks_MIMIC",
                           output_directory="./",
                           seed=None,
                           iteration_list=2 ** np.arange(13),
                           population_sizes=[50,100,200,500],
                           max_attempts=50,
                           keep_percent_list=[0.2,0.3,0.5,0.7],
                           use_fast_mimic=True)
mimic_run_stats, mimic_run_curves = mimic.run()

print("Running Different Problem Sizes...")
# plot of problem sizes
fits_sa = []
fits_ga = []
fits_rhc = []
fits_mimic = []
sizes = []

for length in range(4,65,4):
  print(length)
  sizes.append(length)

  problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
  init_state = np.random.randint(2,size=(length))#([0, 1, 2, 3, 4, 5, 6, 7])
  # SA
  t1 = time.time()
  best_state,best_fitness_sa,curve_sa = mlrose.simulated_annealing(problem, 
                                                        schedule = mlrose.ExpDecay(), 
                                                        max_attempts = 500, 
                                                        max_iters = 500, 
                                                        init_state = init_state,
                                                        random_state = 1,curve = True)
  fits_sa.append(best_fitness_sa)
  # GA
  t2 = time.time()
  best_state,best_fitness_ga,curve_ga = mlrose.genetic_alg(problem, 
                                                 pop_size=20, 
                                                 mutation_prob=0.2, 
                                                 max_attempts = 500, 
                                                 max_iters = 500, 
                                                 random_state = 1,curve = True)
  fits_ga.append(best_fitness_ga)
  # RHC
  t3 = time.time()
  best_state,best_fitness_rhc,curve_rhc = mlrose.random_hill_climb(problem, 
                                                 max_attempts = 500, 
                                                 max_iters = 500, 
                                                 restarts = 2,
                                                 random_state = 1,curve = True)
  fits_rhc.append(best_fitness_rhc)
  # MIMIC
  t4 = time.time()
  best_state,best_fitness_mimic,curve_mimic = mlrose.mimic(problem, pop_size=500, keep_pct=0.3, max_attempts=10,
          max_iters=500, curve=True, random_state=1)
  fits_mimic.append(best_fitness_mimic)
  
  t5 = time.time()
  print("====================")
  print("Problem sizes:",length)
  print("best_fitness_rhc:",best_fitness_rhc,"iteration:",curve_rhc[:,1][-1]-curve_rhc[:,1][0],"time:",t4-t3)
  print("best_fitness_sa:",best_fitness_sa,"iteration:",curve_sa[:,1][-1]-curve_sa[:,1][0],"time:",t2-t1)
  print("best_fitness_ga:",best_fitness_ga,"iteration:",curve_ga[:,1][-1]-curve_ga[:,1][0],"time:",t3-t2)
  print("best_fitness_mimic:",best_fitness_mimic,"iteration:",curve_mimic[:,1][-1]-curve_mimic[:,1][0],"time:",t5-t4)

# plt.plot(sizes,fits_sa)
# plt.plot(sizes,fits_ga)
# plt.plot(sizes,fits_rhc)
# plt.plot(sizes,fits_mimic)

# plt.scatter(sizes,fits_sa)
# plt.scatter(sizes,fits_ga)
# plt.scatter(sizes,fits_rhc)
# plt.scatter(sizes,fits_mimic)

# plt.legend(["SA","GA","RHC","MIMIC"])
# plt.ylabel("fitness")
# plt.xlabel("problem size")
# plt.title("FourPeaks: Fitness of problem sizes with different algorithms")
# plt.show()

