import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# iteration vs fitness
def plot_SA_iter_fits(problem_name):
  
  folder_name = problem_name + "_SA"
  file_path = "./"+ folder_name + "/sa__" + folder_name + "__curves_df.csv"
  curves = pd.read_csv(file_path)

  # seperate curves by different temperature
  temperatures = set(curves["Temperature"])

  # plot iteration vs fitness
  iter_fits = []
  for temp in temperatures:
    iter_fits.append(curves[curves["Temperature"]==temp].reset_index())

  plt.figure(problem_name+": Fitness of SA with different temperatures")
  plt.title(problem_name+": Fitness of SA with different temperatures")
  plt.xlabel("iteration")
  plt.ylabel("fitness")
  for iter_fit in iter_fits:
    plt.plot(iter_fit["Iteration"],iter_fit["Fitness"])
  plt.legend(temperatures)

def plot_GA_iter_fits(problem_name,fix_mutation=0.3,fix_population=100):
  folder_name = problem_name + "_GA"
  file_path = "./"+ folder_name + "/ga__" + folder_name + "__curves_df.csv"
  curves = pd.read_csv(file_path)

  ### Fixed Mutation rates, different popultation
  curves_population = curves[curves["Mutation Rate"]==fix_mutation]
  population_sizes = set(curves["Population Size"])
  # seperate curves by different population sizes
  iter_fits = []
  for size in population_sizes:
    iter_fits.append(curves_population[curves_population["Population Size"]==size].reset_index())
  # plot iteration vs fitness
  plt.figure(problem_name+": Fitness of GA with different population")
  plt.title(problem_name+": Fitness of GA with different population")
  plt.xlabel("iteration")
  plt.ylabel("fitness")
  for iter_fit in iter_fits:
    plt.plot(iter_fit["Iteration"],iter_fit["Fitness"])
  plt.legend(population_sizes)

  ### Fixed popultation, Different Mutation rates, 
  curves_mutation = curves[curves["Population Size"]==fix_population]
  mutation_rates = set(curves["Mutation Rate"])
  iter_fits = []
  # seperate curves by different mutation_rates
  for rate in mutation_rates:
    iter_fits.append(curves_mutation[curves_mutation["Mutation Rate"]==rate].reset_index())
  # plot iteration vs fitness
  plt.figure(problem_name+": Fitness of GA with different mutation rates")
  plt.title(problem_name+": Fitness of GA with different mutation rates")
  plt.xlabel("iteration")
  plt.ylabel("fitness")
  for iter_fit in iter_fits:
    plt.plot(iter_fit["Iteration"],iter_fit["Fitness"])
  plt.legend(mutation_rates)

def plot_RHC_iter_fits(problem_name):
  folder_name = problem_name + "_RHC"
  file_path = "./"+ folder_name + "/rhc__" + folder_name + "__curves_df.csv"
  curves = pd.read_csv(file_path)

  restarts = set(curves["current_restart"])
  iter_fits = []
  for restart in restarts:
    iter_fits.append(curves[curves["current_restart"]==restart].reset_index())
  # plot iteration vs fitness
  plt.figure(problem_name+": Fitness of RHC with different restarts")
  plt.title(problem_name+": Fitness of RHC with different restarts")
  plt.xlabel("iteration")
  plt.ylabel("fitness")
  for iter_fit in iter_fits:
    plt.plot(iter_fit["Iteration"],iter_fit["Fitness"])
  plt.legend(restarts)


def plot_MIMIC_iter_fits(problem_name,fix_percent=0.5,fix_population=150):
  folder_name = problem_name + "_MIMIC"
  file_path = "./"+ folder_name + "/mimic__" + folder_name + "__curves_df.csv"
  curves = pd.read_csv(file_path)

  ### Fixed Mutation rates, different popultation
  curves_population = curves[curves["Keep Percent"]==fix_percent]
  population_sizes = set(curves["Population Size"])
  # seperate curves by different population sizes
  iter_fits = []
  for size in population_sizes:
    iter_fits.append(curves_population[curves_population["Population Size"]==size].reset_index())
  # plot iteration vs fitness
  plt.figure(problem_name+": Fitness of MIMIC with different population")
  plt.title(problem_name+": Fitness of MIMIC with different population")
  plt.xlabel("iteration")
  plt.ylabel("fitness")
  for iter_fit in iter_fits:
    plt.plot(iter_fit["Iteration"],iter_fit["Fitness"])
  plt.legend(population_sizes)

  ### Fixed popultation, Different Mutation rates, 
  curves_percentile = curves[curves["Population Size"]==fix_population]
  keep_percents = set(curves["Keep Percent"])
  iter_fits = []
  # seperate curves by different mutation_rates
  for percent in keep_percents:
    iter_fits.append(curves_percentile[curves_percentile["Keep Percent"]==percent].reset_index())
  # plot iteration vs fitness
  plt.figure(problem_name+": Fitness of MIMIC with different kept percentile")
  plt.title(problem_name+": Fitness of MIMIC with different kept percentile")
  plt.xlabel("iteration")
  plt.ylabel("fitness")
  for iter_fit in iter_fits:
    plt.plot(iter_fit["Iteration"],iter_fit["Fitness"])
  plt.legend(keep_percents)


def plot_ALL_iter_fits(problem_name, 
                       best_sa_temperature, 
                       best_ga_population, 
                       best_ga_mutation,
                       best_rhc_restart,
                       best_mimic_population,
                       best_mimic_percentage,
                       upper_bound = 2000):
  folder_name = problem_name + "_SA"
  file_path = "./"+ folder_name + "/sa__" + folder_name + "__curves_df.csv"
  curves_SA = pd.read_csv(file_path)

  folder_name = problem_name + "_GA"
  file_path = "./"+ folder_name + "/ga__" + folder_name + "__curves_df.csv"
  curves_GA = pd.read_csv(file_path)

  folder_name = problem_name + "_RHC"
  file_path = "./"+ folder_name + "/rhc__" + folder_name + "__curves_df.csv"
  curves_RHC = pd.read_csv(file_path)

  folder_name = problem_name + "_MIMIC"
  file_path = "./"+ folder_name + "/mimic__" + folder_name + "__curves_df.csv"
  curves_MIMIC = pd.read_csv(file_path)



  curve_SA = curves_SA[curves_SA["Temperature"]==best_sa_temperature].reset_index()
  curve_GA_ = curves_GA[curves_GA["Mutation Rate"]==best_ga_mutation].reset_index()
  curve_GA = curve_GA_[curve_GA_["Population Size"]==best_ga_population].reset_index()
  curve_RHC = curves_RHC[curves_RHC["current_restart"]==best_rhc_restart].reset_index()
  curve_MIMIC_ = curves_MIMIC[curves_MIMIC["Population Size"]==best_mimic_population].reset_index()
  curve_MIMIC = curve_MIMIC_[curve_MIMIC_["Keep Percent"]==best_mimic_percentage].reset_index()

  # plot all iteration vs fitness in one figure
  plt.figure(problem_name+": Comparison of four algorithms")
  plt.title(problem_name+": Comparison of four algorithms")
  plt.xlabel("iteration")
  plt.ylabel("fitness")

  plt.plot(curve_SA["Iteration"],curve_SA["Fitness"])
  plt.plot(curve_GA["Iteration"],curve_GA["Fitness"])
  plt.plot(curve_RHC["Iteration"],curve_RHC["Fitness"])
  plt.plot(curve_MIMIC["Iteration"],curve_MIMIC["Fitness"])
  plt.xlim(0,upper_bound) 
  plt.legend(["SA","GA","RHC","MIMIC"])



if __name__ == "__main__":
  problem_name = "FourPeaks"
  plot_SA_iter_fits(problem_name)
  plot_GA_iter_fits(problem_name,fix_mutation=0.3,fix_population=200)
  plot_RHC_iter_fits(problem_name)
  plot_MIMIC_iter_fits(problem_name,fix_population=200,fix_percent=0.2)
  plot_ALL_iter_fits(problem_name,
                   best_sa_temperature=10,
                   best_ga_population=200,
                   best_ga_mutation=0.3,
                   best_rhc_restart = 2,
                   best_mimic_population=200,
                   best_mimic_percentage=0.2,
                   upper_bound = 1200)

  problem_name = "Queen"
  plot_SA_iter_fits(problem_name)
  plot_GA_iter_fits(problem_name,fix_mutation=0.1,fix_population=150)
  plot_RHC_iter_fits(problem_name)
  plot_MIMIC_iter_fits(problem_name,fix_population=100,fix_percent=0.3)
  plot_ALL_iter_fits(problem_name,
                   best_sa_temperature=0.5,
                   best_ga_population=150,
                   best_ga_mutation=0.1,
                   best_rhc_restart = 2,
                   best_mimic_population=100,
                   best_mimic_percentage=0.3,
                   upper_bound = 500)


  problem_name = "FlipFlop"
  plot_SA_iter_fits(problem_name)
  plot_GA_iter_fits(problem_name,fix_mutation=0.3,fix_population=200)
  plot_RHC_iter_fits(problem_name)
  plot_MIMIC_iter_fits(problem_name,fix_population=100,fix_percent=0.3)
  plot_ALL_iter_fits(problem_name,
                   best_sa_temperature=5,
                   best_ga_population=200,
                   best_ga_mutation=0.3,
                   best_rhc_restart = 0,
                   best_mimic_population=500,
                   best_mimic_percentage=0.3,
                   upper_bound = 1200)

  plt.show()
