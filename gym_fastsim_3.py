import cma
import gym, gym_fastsim
from deap import *
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import array
import random
import operator
import math
import numpy
import time
from plot import *

from scoop import futures

from novelty_search import *

but_atteint = False

def simulation(env,genotype,display=False):
    global but_atteint
    global size_nn
    nn=SimpleNeuralControllerNumpy(5,2,2,10)
    if genotype != None:
        nn.set_parameters(genotype)
    observation = env.reset()
    if(display):
        env.enable_display()
    then = time.time()
    but = 0
    for i in range(800):
        env.render()
        action=nn.predict(observation)
        action = [i * env.maxVel for i in action]
        observation,reward,done,info=env.step(action)
        #print("Step %d Obs=%s  reward=%f  dist. to objective=%f  robot position=%s  End of ep=%s" % (i, str(observation), reward, info["dist_obj"], str(info["robot_pos"]), str(done)))
        if(display):
            time.sleep(0.01)
        if done:
            but_atteint = True
            but += 1
            break

    now = time.time()
    #print("%d timesteps took %f seconds" % (i, now - then))
    xg,yg = env.goalPos
    x,y,theta = env.get_robot_pos()    # x,y,theta    ?? pourquoi theta??? to do
    return but,math.sqrt((x-xg)**2+(y-yg)**2),[x,y]  


## Il vous est recommandé de gérer les différentes variantes avec cette variable. Les 3 valeurs possibles seront:
## "FIT+NS": expérience multiobjectif avec la fitness et la nouveauté (NSGA-2)
## "NS": nouveauté seule
## "FIT": fitness seule
## pour les variantes avec un seul objectif, vous pourrez, au choix, utiliser CMA-ES ou NSGA-2 avec un seul objectif,
## il vous est cependant recommandé d'utiliser NSGA-2 car cela limitera la différence entre les variantes et cela 
##vvous fera gagner du temps pour la suite.
#####################################################################
##########################################################################

def launch_nsga2(env,variant,size_pop=100,pb_crossover=0.6, pb_mutation=0.3, nb_generation=1000, display=False, verbose=False):
    global but_atteint
    but_generation = None
    # votre code contiendra donc des tests comme suit pour gérer la différence entre ces variantes:
    if (variant=="FIT+NS"):
        creator.create("FitnessMax",base.Fitness,weights=(1.0,-1.0,1.0))
    elif (variant=="FIT"):
        creator.create("FitnessMax",base.Fitness,weights=(1.0,-1.0))
    elif (variant=="NS"):
        creator.create("FitnessMax",base.Fitness,weights=(1.0,))
    else:
        print("Variante inconnue: "+variant)
    
    IND_SIZE = 192
    random.seed()

    #create class
    creator.create("FitnessMax",base.Fitness,weights=(1.0,-1.0,1.0))
    creator.create("Individual",list,fitness=creator.FitnessMax,but=float,fit=float,bd=list,novelty=float)
    # toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list,  toolbox.individual)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate",tools.cxBlend,alpha=0.3)
    #halloffame
    paretofront = tools.ParetoFront()    
    #statistics
    statistics = tools.Statistics(lambda ind: ind.fitness.values)
    statistics.register("avg", numpy.mean)
    statistics.register("std", numpy.std)
    statistics.register("min", numpy.min)
    statistics.register("max", numpy.max)
        #log
    logbook = tools.Logbook()
    logbook.header = ["gen","nevals"]+ statistics.fields

    # pour plot heatmap
    position_record = []

    # generer la population initiale
    pop = toolbox.population(size_pop)

    # simulation
    for ind in pop:
        ind.but,ind.fit,ind.bd = simulation(env,ind,display=display)
        position_record.append(ind.bd)

    # MAJ archive
    if (variant=="FIT+NS") or(variant=="NS"):
        arc = updateNovelty(pop,pop,None)    

    # MAJ fitness
    for ind in pop:
        ind.fitness.values = list([])
        if variant == "FIT+NS" or variant == "FIT":
            ind.fitness.values = list(ind.fitness.values)+list([ind.but,ind.fit])
        if variant == "FIT+NS" or variant == "NS":
            ind.fitness.values = list(ind.fitness.values)+list([ind.novelty])

    # Update the hall of fame with the generated individuals
    paretofront.update(pop)
    record = statistics.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    if verbose:
        print(logbook.stream)

    means = []
    mins = []

    # main boucle
    for gen in range(1, nb_generation+1):

        #print("generation ",gen)
        

        if but_atteint and but_generation==None:
            but_generation = gen

        #if gen%50 == 0:
            #print("generation ",gen)

        # Select the next generation individuals
        offspring = toolbox.select(pop, size_pop)
        
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))  
        #print(np.mean(np.array([ind.fitness.values[1] for ind in offspring])))
        #print(np.min(np.array([ind.fitness.values[1] for ind in offspring])))

        if (variant=="NS"):
            means.append(np.mean(np.array([ind.fitness.values[0] for ind in offspring])))
            mins.append(np.min(np.array([ind.fitness.values[0] for ind in offspring])))

        else:
            means.append(np.mean(np.array([ind.fitness.values[1] for ind in offspring])))
            mins.append(np.min(np.array([ind.fitness.values[1] for ind in offspring])))


        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random()<pb_crossover:
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values

        #mutation
        for mutant in offspring:
            if np.random.random() < pb_mutation:
                tools.mutGaussian(mutant, mu=0.0, sigma=1, indpb=0.1)
                del mutant.fitness.values

        # simulation
        invalid_inds = [ind for ind in offspring if ind.fitness.valid == False]
        for ind in invalid_inds:
            ind.but,ind.fit,ind.bd = simulation(env,ind,display=display)
            position_record.append(ind.bd)
        # MAJ archive
        if (variant=="FIT+NS") or(variant=="NS"):
            arc = updateNovelty(offspring,offspring,arc,k=15)  #Update the novelty criterion (including archive update) 
        # MAJ fitness
        for ind in offspring:
            ind.fitness.values = list([])
            if variant == "FIT+NS" or variant == "FIT":
                ind.fitness.values = list(ind.fitness.values)+list([ind.but,ind.fit])
            if variant == "FIT+NS" or variant == "NS":
                ind.fitness.values = list(ind.fitness.values)+list([ind.novelty])

        # Select the next generation population
        pop[:] = offspring + pop

        # Update the hall of fame with the generated individuals
        paretofront.update(pop)
        record = statistics.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)
        if verbose:
            print(logbook.stream)

        if but_atteint:
            break
            
    return pop,logbook, paretofront,position_record,but_atteint,but_generation, means, mins


