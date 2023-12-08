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
from synthetic_data import objective_syn, objective_syn_183
from separate_fitting import obj_separate, gen, power
import pandas as pd

# Systematic study ??

def ga_search(NGEN, objective_fn):
    alpha1 = 0.1
    Average = 0
    Std = 3
    Mut_p = 0.3
    POP_SIZE = 100
    CXPB = 0.7
    MUTPB = 0.3
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
    toolbox.register("evaluate", objective_fn)

    # if not synthetic: 
    #     toolbox.register("evaluate", objective)
    # else:
    #     if syn183:
    #         toolbox.register("evaluate", objective_syn_183)
    #     else:
    #         toolbox.register("evaluate", objective_syn)

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

def ga(num_trials = 1, num_gen = 1, name = '', objective_fn = objective):
    df_list = []
    for i in tqdm(range(num_trials)):
        top = ga_search(num_gen, objective_fn)
        df_top = pd.DataFrame(top, columns = ['c1', 'c2', 'c3', 'c4', 'k31', 'k41', 'k51'])
        df_top['num_trial'] = i
        df_list.append(df_top)
    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    # if syn183:
    #     final_df.to_csv("ga_syn183_1.csv", index=False)
    # else:
    #     final_df.to_csv("ga_syn_1.csv", index=False)
    final_df.to_csv(f"{name}.csv")

# profiler = cProfile.Profile()

# profiler.enable()

# ga(num_trials = 10, num_gen = 100, synthetic=True)

# ga(num_trials = 10, num_gen = 100, synthetic=True, syn183=True)

# profiler.disable()
# s = io.StringIO()
# ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
# ps.print_stats() 

# print(s.getvalue())



gen(10, [2.25551330653112,
 -1.3056691656198742,
 0.4984769347191149,
 2.31942225639846,
 -0.631828119307082,
 2.7201431777574223,
 -0.005981397391665376])

ga(num_trials = 10, num_gen = 100, name = 'conc_10', objective_fn = obj_separate(10))

gen(12, [2.2476613699587853,
 -1.5335185237890552,
 0.37214497901439947,
 1.3738754904537522,
 -8.951898730459568,
 1.3048106923399783,
 -1.084474034127443])

ga(num_trials = 10, num_gen = 100, name = 'conc_12', objective_fn = obj_separate(12))

gen(15, [2.1020291697247773,
 -1.5520505539736358,
 0.17454089093484454,
 2.7790240674847073,
 -3.6039012238749386,
 2.705152690070094,
 -2.5482077679848203])

ga(num_trials = 10, num_gen = 100, name = 'conc_15', objective_fn = obj_separate(15))