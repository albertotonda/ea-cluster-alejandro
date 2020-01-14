
from joblib import Parallel, delayed
import multiprocessing
import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys
import random
# this is an incredibly useful function
from pandas import read_csv
import pandas as pd 

from runClassifiers import *

def testFeaturesParallel(X,y,tempIds):
    dataRed=np.zeros((len(X),len(tempIds)))

    for i in range(0,len(tempIds)):
        for j in range (0,len(X[0])):
            if (j== tempIds[i]):
                for k in range(0,len(X)):
                    dataRed[k,i]=X[k,j]
    cost=runFeatureReduce(dataRed,y)
    #print(tempIds)
    return cost

def getBestTournament(population,cost,tSize):
    #Tornamment
    pop=len(population)
    #tSize=25
    popTournament=[0 for i in range (pop)]
    for i in range(pop):
        popTournament[i]=i

    tourVector=random.sample(popTournament,tSize)
    #print(tourVector)

    subPopulation=[]
    subCost=np.zeros(tSize)
    for i in range(0,tSize):
        subPopulation.append(population[tourVector[i]])
        subCost[i]=cost[tourVector[i]]

    #formattedCost = [ '%.2f' % elem for elem in subCost ]
    #print(formattedCost)
    list1, list2 = zip(*sorted(zip(subCost, subPopulation)))
    subCost=list1
    subPopulation=list2
    #formattedCost = [ '%.2f' % elem for elem in subCost ]
    #print(formattedCost)
    return subPopulation[tSize-1],subCost[tSize-1]

def mateChromosome(father,mother,size,nVar):

    print("Crossover, father (len " + str(len(father)) + ") =", father)
    print("Crossover, mother (len " + str(len(mother)) + ") =", mother)

    cut=random.randint(0,size)
    son1=[]
    son2=[]
    #print(cut)
    for i in range(size):
        if (i<cut):
            son1.append(father[i])
            son2.append(mother[i])
        else:
            son1.append(mother[i])
            son2.append(father[i])
    
    # after creating the two 'sons', we proceed to check if they are valid
    for i in range(size):
        # 'set' removes all duplicate elements
        new_son1 = list(set(son1))
        # if the 'set' and the original list are of different size, it means there were duplicate features
        if len(new_son1) < len(son1) :
            # create list of all features not included in son1, and sample it, so it's guaranteed there are no duplicates
            features_not_in_son1 = [i for i in range(0, nVar) if i not in new_son1]
            features_to_be_added = random.sample(features_not_in_son1, len(son1) - len(new_son1))
            new_son1.extend(features_to_be_added)
            son1 = new_son1
            
    for i in range(size):
        # 'set' removes all duplicate elements
        new_son2 = list(set(son2))
        # if the 'set' and the original list are of different size, it means there were duplicate features
        if len(new_son2) < len(son2) :
            # create list of all features not included in son1, and sample it, so it's guaranteed there are no duplicates
            features_not_in_son2 = [i for i in range(0, nVar) if i not in new_son2]
            features_to_be_added = random.sample(features_not_in_son2, len(son2) - len(new_son2))
            new_son2.extend(features_to_be_added)
            son2 = new_son2

    # and now, we can perform mutations
    features_not_in_son1 = [i for i in range(0, nVar) if i not in son1]
    indexes_to_be_changed = []
    for i in range(size):
        value=random.random()
        if (value<0.20):
            indexes_to_be_changed.append(i)
    features_to_be_added = random.sample(features_not_in_son1, len(indexes_to_be_changed))
    for i in range(0, len(indexes_to_be_changed)) :
        son1[indexes_to_be_changed[i]] = features_to_be_added[i]

    features_not_in_son2 = [i for i in range(0, nVar) if i not in son2]
    indexes_to_be_changed = []
    for i in range(size):
        value=random.random()
        if (value<0.20):
            indexes_to_be_changed.append(i)
    features_to_be_added = random.sample(features_not_in_son2, len(indexes_to_be_changed))
    for i in range(0, len(indexes_to_be_changed)) :
        son2[indexes_to_be_changed[i]] = features_to_be_added[i]

    print("child1 (len " + str(len(son1)) + ") =", son1)
    print("child2 (len " + str(len(son2)) + ") =", son2)

    return son1,son2

def runTest():
        
    # set random seed
    random.seed(42) # TODO remove/comment this line for the real experiments
        
    #load Data
    X, y, features = loadDataset()

    #fixed output features
    size=64
    #total population size
    pop=8
    #number of original features
    nVar=len(features)
    #how many solutions are kept each generation
    keep=4
    #total iterations
    iter=3000
    #tournament size
    tournament=2
    #total threads
    #threads=5
    threads=1 # TODO set correct number of threads
    
    f=open("results.txt","w+")
    f.close()

    #create Original Population********************************************************
    base=[i for i in range (nVar)]
    #for i in range(nVar):
    #   base[i]=i
        

    population=[]
    cost=np.zeros(pop)
    for i in range(0,pop):
        chromosome=random.sample(base,size)
        population.append(chromosome)
        print("Individual #%d: %s" % (i, str(chromosome)))# TODO comment this line

    results = Parallel(n_jobs=threads, verbose=5, backend="multiprocessing")(delayed(testFeaturesParallel)(X,y,population[i]) for i in range(0,pop))
    for i in range(0,pop):
        cost[i]=results[i]
    #********************************************************
    #for i in range(0,pop):
    #   cost[i]=testFeaturesParallel(X,y,population[i])
        #print(i, end = ' ')
    
    #Sort Solutions********************************************************
    list1, list2 = zip(*sorted(zip(cost, population)))
    cost=list1
    population=list2
    #********************************************************
    print(np.max(cost))
    #Main For ********************************************************
    for i in range(0,iter):
        
        newPopulation=[]
        newCost=np.zeros(pop)
        while len(newPopulation) < keep :
            fatherTournament,fatherCost=getBestTournament(population,cost,tournament)
            motherTournament,motherCost=getBestTournament(population,cost,tournament)
            son1,son2=mateChromosome(fatherTournament,motherTournament,size,nVar)
            newPopulation.append(son1)
            newPopulation.append(son2)
        
        results = Parallel(n_jobs=threads,backend="multiprocessing")(delayed(testFeaturesParallel)(X,y,newPopulation[j]) for j in range(0,len(newPopulation)))
        for j in range (0,len(newPopulation)):
            newCost[j]=results[j]
        
        count=len(newPopulation)
        
        for j in range(keep):
            newPopulation.append(population[count+j])
            
        #for j in range(keep):
        #   newCost[count+j]=testFeaturesParallel(X,y,population[count+j])
        
        results = Parallel(n_jobs=threads, backend="multiprocessing")(delayed(testFeaturesParallel)(X,y,population[count+j]) for j in range(0,keep))
        for j in range(0,keep):
            newCost[count+j]=results[j]
        
        population=newPopulation
        cost=newCost
        
        list1, list2 = zip(*sorted(zip(cost, population)))
        cost=list1
        population=list2
        print("New population has now size %d" % len(population))
        print(cost[pop-1])
        print(features[population[pop-1]])
        print("%d\t%.4f\t%.4f" %(i,np.max(cost),np.mean(cost)))
        
        
        f1=open("./results.txt","a+")
        for item in features[population[pop-1]]:
            f1.write("%s\t" % item)
        f1.write(str(cost[pop-1])+"\n")
        f1.close()
        
    for i in range(10):
        resultCost=testFeaturesParallel(X,y,population[pop-1])
        print(resultCost)
        print(features[population[pop-1]])


if __name__ == "__main__" :
    sys.exit( runTest() )
