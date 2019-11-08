import cma
import gym, gym_fastsim
from deap import *
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
from scipy.spatial import KDTree

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

def simulation(env,genotype,display=True):
    global but_atteint
    global size_nn
    nn=SimpleNeuralControllerNumpy(5,2,2,10)
    if genotype != None:
        nn.set_parameters(genotype)
    observation = env.reset()
    if(display):
        env.enable_display()
    then = time.time()
    reward_final = 0
    for i in range(20):
        env.render()
        action=nn.predict(observation)
        action = [i * env.maxVel for i in action]
        observation,reward,done,info=env.step(action)
        #print("Step %d Obs=%s  reward=%f  dist. to objective=%f  robot position=%s  End of ep=%s" % (i, str(observation), reward, info["dist_obj"], str(info["robot_pos"]), str(done)))
        if(display):
            time.sleep(0.01)
        if done:
            but_atteint = True
            reward_final = 1
            break


    now = time.time()

    #print("%d timesteps took %f seconds" % (i, now - then))

    x,y,theta = env.get_robot_pos()    # x,y,theta    ?? pourquoi theta??? to do
    xg,yg = env.goalPos
    return reward_final,math.sqrt((x-xg)**2+(y-yg)**2) 
## Il vous est recommandé de gérer les différentes variantes avec cette variable. Les 3 valeurs possibles seront:
## "FIT+NS": expérience multiobjectif avec la fitness et la nouveauté (NSGA-2)
## "NS": nouveauté seule
## "FIT": fitness seule
## pour les variantes avec un seul objectif, vous pourrez, au choix, utiliser CMA-ES ou NSGA-2 avec un seul objectif,
## il vous est cependant recommandé d'utiliser NSGA-2 car cela limitera la différence entre les variantes et cela 
##vvous fera gagner du temps pour la suite.
#####################################################################
variant="FIT"
#####################################################################

# votre code contiendra donc des tests comme suit pour gérer la différence entre ces variantes:
if (variant=="FIT+NS"):
    pass ## A remplacer par les instructions appropriées
elif (variant=="FIT"):
    pass ## A remplacer par les instructions appropriées
elif (variant=="NS"):
    pass ## A remplacer par les instructions appropriées
else:
    print("Variante inconnue: "+variant)
##########################################################################

def launch_nsga2(env,size_pop=50,pb_crossover=0.6, pb_mutation=0.3, nb_generation=100, display=False, verbose=False):

    IND_SIZE = 192
    random.seed()

    #create class
    creator.create("FitnessMax",base.Fitness,weights=(1.0,-1.0))
    creator.create("Individual",list,fitness=creator.FitnessMax,but_atteint=float,bd=list,fit=float,novelty=float)
    # toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list,  toolbox.individual)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate",tools.cxBlend,alpha=0.1)
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

    pop[0].fitness.values = []
    print("****: ",pop[0].fitness.values)
    pop[0].fitness.values = list(pop[0].fitness.values)+list([1,2,3])
    print("****: ",pop[0].fitness.values)
    # simulation
    for ind in pop:
#        ind.bd = simulation(env,ind,display=display)
        ind.but_atteint,ind.fit = simulation(env,ind,display=display)
        position_record.append(ind.bd)

    # MAJ archive
#    arc = updateNovelty(pop,pop,None)    


    # MAJ fitness
    for ind in pop:
        ind.fitness.values = (ind.but_atteint,ind.fit)

    # Update the hall of fame with the generated individuals
    paretofront.update(pop)
    record = statistics.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    if verbose:
        print(logbook.stream)

    # main boucle
    for gen in range(1, nb_generation+1):
        print("generation ",gen)

        # Select the next generation individuals
        offspring = toolbox.select(pop, size_pop)
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

        # simulation
        invalid_inds = [ind for ind in offspring if ind.fitness.valid == False]
        for ind in invalid_inds:
#            ind.bd = simulation(env,ind,display=display)
            ind.but_atteint,ind.fit = simulation(env,ind,display=display)
            position_record.append(ind.bd)
        # MAJ archive
#        arc = updateNovelty(offspring,offspring,arc,k=15)  #Update the novelty criterion (including archive update) 
        # MAJ fitness
        for ind in offspring:
#            ind.fitness.values = (ind.novelty,)
            ind.fitness.values = (ind.but_atteint,ind.fit)

        # Select the next generation population
        pop[:] = offspring

        # Update the hall of fame with the generated individuals
        paretofront.update(pop)
        record = statistics.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)
        if verbose:
            print(logbook.stream)

        if but_atteint:
            break
            
    return pop,logbook, paretofront,position_record







st = time.time()




display= False
env = gym.make('FastsimSimpleNavigation-v0')

but_atteint = False
#simulation(env,None,True)
_,_,paretofront,position_record = launch_nsga2(env,nb_generation=10, size_pop=100,pb_crossover=0.1,pb_mutation=0.9,display=display,verbose=True)

print("test1")
print("test1")
print("test1")
print("test1")
print("test1")
print("pareto*****:  ",paretofront)
plot_pareto_front(paretofront, "Final pareto front")
env.close()

"""

#=================== Traitement du resultat ==========================================================
name = 'log/position_record_07_nov_18_00'
import pickle
# open a file, where you ant to store the data
file = open(name, 'wb')     # le 07 nov  X:Y
# dump information to that file
pickle.dump(position_record, file)
# close the file
file.close()

# plot
heatmap = np.zeros((120,120))
for i in range(10):
    for position in position_record:
        x = int(position[0]) // 5
        y = int(position[1]) // 5
        heatmap[y][x] += 1
plt.imshow(heatmap)
print(but_atteint)
print(time.time()-st)
plt.savefig(name)
plt.show()
"""