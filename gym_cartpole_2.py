import cma
import gym
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
import time
from deap import base
from deap import creator
from deap import tools
import numpy 
from plot import *
import random

def eval_nn(env, genotype, render=False):
    energie = 500
    nn=SimpleNeuralControllerNumpy(4,1,2,10)
    nn.set_parameters(genotype)
    observation = env.reset()
    x = 0
    y = 0
    for t in range(energie):
        if render:
            env.render()
            time.sleep(0.05)
        action=nn.predict(observation)
        if action>0:
            action=1
        else:
            action=0
        observation, reward, done, info = env.step(action) 
        x += abs(observation[0])
        y += abs(observation[2])
        if done:
            break
    x = x/t
    x += (energie - t)*2.4  #penalisation pour les tours restant
    y = y/t
    y += (energie - t)*41.8  #penalisation pour les tours restant
    return x,y

def es(env,size_pop=50,lambda_=100,pb_crossover=0.6, pb_mutation=0.5,nb_generation=100, display=False, verbose=False):

    IND_SIZE = 171
    random.seed()
    
    #create class
    creator.create("FitnessMax",base.Fitness,weights=(-1.0,-1.0))
    creator.create("Individual",list,fitness=creator.FitnessMax)
    # toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list,  toolbox.individual)
    toolbox.register("select", tools.selNSGA2,k=lambda_)
    toolbox.register("mate",tools.cxBlend,alpha=0.1)
    #statistics
    statistics = tools.Statistics(lambda ind: ind.fitness.values)
    statistics.register("avg", numpy.mean)
    statistics.register("std", numpy.std)
    statistics.register("min", numpy.min)
    statistics.register("max", numpy.max)
        #log
    logbook = tools.Logbook()
    logbook.header = ["gen","nevals"]+ statistics.fields
    #pareto
    paretofront = tools.ParetoFront()

    # generer la population initiale
    pop = toolbox.population(size_pop)

    # evaluation
    for ind in pop:
        ind.fitness.values = eval_nn(env,ind)
    
    # display
    if display:
        plot_pop_pareto_front(pop, [], "Gen: %d"%(0))

    # Update the hall of fame with the generated individuals
    if paretofront is not None:
        paretofront.update(pop)

    record = statistics.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    if verbose:
        print(logbook.stream)

    
    for gen in range(1, nb_generation+1):
        print("generation ",gen)
        # Select the next generation individuals
        offspring = toolbox.select(pop)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))  

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

        # evaluation
        invalid_inds = [ind for ind in offspring if ind.fitness.valid == False]
        for ind in invalid_inds:
            ind.fitness.values = eval_nn(env,ind)
 
        # remplacement
        pop[:] = offspring + pop

            # Update the pareto with the generated individuals
        if paretofront is not None:
            paretofront.update(pop)

        if display:
            plot_pop_pareto_front(pop, paretofront, "Gen: %d"%(gen))

        record = statistics.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)
        if verbose:
            print(logbook.stream)
            
    return pop,logbook, paretofront
