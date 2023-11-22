import imaplib
import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm
import time
from deap import base, creator, tools, algorithms
import random
import cProfile
import io
import pstats
import memory_profiler
from utils import objective
import pandas as pd

# Systematic study ??

def ga_search(NGEN):
    alpha1 = 0.1
    Average = 0
    Std = 3
    Mut_p = 0.2
    POP_SIZE = 100
    CXPB = 0.7
    MUTPB = 0.2
    Cr = 8 / 100

    # aim to minimize its values
    # How?
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    creator.create("Individual", list, fitness=creator.FitnessMin)

    # The toolbox is a container for various functions used in the genetic algorithm
    toolbox = base.Toolbox()

    # valid_values = [10**i for i in range(-20, 20)]
    # toolbox.register("attr_float", random.choice, valid_values) # discrete

    valid_values = [i for i in range(-10, 10)]
    toolbox.register("attr_float", random.choice, valid_values)


    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=7)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # create a list of individuals, essentially generating the initial population

    # Genetic Operators
    toolbox.register("mate", tools.cxBlend, alpha= alpha1)

    toolbox.register("mutate", tools.mutGaussian, mu=Average, sigma=Std, indpb=Mut_p)

    # Evaluation Function
    toolbox.register("evaluate", objective)

    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=POP_SIZE)

    fits = list(map(toolbox.evaluate, pop))

    for fit, ind in zip(fits, pop):
        ind.fitness.values = fit
    # print("line 62   ", len(pop))
    for gen in tqdm(range(NGEN)):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        # fits = list(map(toolbox.evaluate, offspring))
        # print("line 66   ", len(offspring))
        for ind in offspring:
            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))

    top10 = tools.selBest(pop, k=10)
    return top10

def ga(num_trials = 1, num_gen = 1):
    df_list = []
    for i in tqdm(range(num_trials)):
        top = ga_search(num_gen)
        df_top = pd.DataFrame(top, columns = ['c1', 'c2', 'c3', 'c4', 'k31', 'k41', 'k51'])
        df_top['num_trial'] = i
        df_list.append(df_top)
    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    final_df.to_csv("ga.csv", index=False)


# profiler = cProfile.Profile()

# profiler.enable()

ga(num_trials = 30, num_gen = 60)

# profiler.disable()
# s = io.StringIO()
# ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
# ps.print_stats() 

# print(s.getvalue())




