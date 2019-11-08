import cma
import gym
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
import time
from deap import base
from deap import creator
from deap import tools
import numpy
import numpy as np
import array
import random

IND_SIZE = 5
MIN_VALUE = -30
MAX_VALUE = 30
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3

def eval_nn(env, genotype, render=False):
    nn=SimpleNeuralControllerNumpy(4,1,2,5)
    nn.set_parameters(genotype)
    observation = env.reset()
    x = 0
    y = 0
    for t in range(500):
        if render:
            env.render()
            time.sleep(0.01)
        action=nn.predict(observation)
        if action>0:
            action=1
        else:
            action=0
        observation, reward, done, info = env.step(action)
        x += abs(observation[0])
        y += abs(observation[2]) 
        if done:
            print("Episode finished after %d timesteps"%(t+1))
            break
    return x / t, y / t

# Génération d'un individu avec une distribution uniforme dans les bornes indiquées
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind


### A completer pour optimiser les parametres du reseau de neurones avec CMA-ES ###
def es(env,mu=100, lambda_=200, cxpb=0.6, mutpb=0.3, ngen=1000, display=False, verbose=False):
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
    creator.create("Strategy", array.array, typecode="d")
    #toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
        IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
    toolbox.register("select", tools.selNSGA2)
    random.seed()

    population = toolbox.population(n=mu)
    
    paretofront = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    ## A completer: évaluation des individus et mise à jour de leur fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = [eval_nn(env,ind,render=display) for ind in invalid_ind]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    #======================================

    if paretofront is not None:
        paretofront.update(population)

    #print("Pareto Front: "+str(paretofront))

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):


        ## A completer, génération des 'offspring' et sélection de la nouvelle population
        # Select the next generation individuals
        offspring = toolbox.select(population)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))  

        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random()<cxpb:
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values


        #mutation
        for mutant in offspring:
            if np.random.random() < mutpb:
                tools.mutGaussian(mutant, mu=0.0, sigma=1, indpb=0.1)
                del mutant.fitness.values
      
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
        # Update the hall of fame with the generated individuals
        if paretofront is not None:
            paretofront.update(offspring)

        


        if display:
            plot_pop_pareto_front(population, paretofront, "Gen: %d"%(gen))

        
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        population = offspring


    return population, logbook, paretofront

