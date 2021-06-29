import random 
import torch
import numpy as np
from src.constants import *  
 
class Particle(object): 
    def __init__(self, dataset, apps, graph, f, bounds, data_type, hosts, chromosome=[]): 
        self.dataset = dataset
        self.f = f
        self.apps, self.graph = apps, graph
        self.hosts = hosts
        self.data_type = data_type
        self.chromosome = torch.tensor(chromosome if chromosome != [] else random.choice(dataset)[0], dtype=torch.float)
        self.fitness = self.cal_fitness() 
        self.bounds = bounds
  
    def optimize(self): 
        par2 = random.choice(self.dataset)[0]
        child_chromosome = [] 
        for gp1, gp2 in zip(self.chromosome, par2):       
            prob = random.random() 
            if prob < 0.85: 
                child_chromosome.append(gp1) 
            else: 
                child_chromosome.append(gp2) 
        alloc = []
        for i in child_chromosome:
            oneHot = [0] * self.hosts; alist = i.tolist()[-self.hosts:]
            oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
        child_chromosome = np.concatenate((self.chromosome[:,0:-self.hosts], np.array(alloc)), axis=1)
        if self.cal_fitness() > self.f(torch.tensor(child_chromosome, dtype=torch.float), self.apps, self.graph):
            return self
        return Particle(self.dataset, self.apps, self.graph, self.f, self.bounds, self.data_type, self.hosts, child_chromosome) 
  
    def cal_fitness(self): 
        res = self.f(self.chromosome, self.apps, self.graph)
        return res
  
def psoW(dataset, apps, graph, f, bounds, data_type, hosts): 
    generation = 1
    population = [] 
    best_fitness = []
  
    for _ in range(POPULATION_SIZE): 
        population.append(Particle(dataset, apps, graph, f, bounds, data_type, hosts)) 
  
    while True: 
        population = sorted(population, key = lambda x:x.fitness) 
  
        new_generation = [] 
        new_generation.extend(population[:int(0.1*POPULATION_SIZE)]) 
  
        s = int(0.9*POPULATION_SIZE) 
        for _ in range(s): 
            parent1 = random.choice(population[:50]) 
            parent2 = random.choice(population[:50]) 
            child1, child2 = parent1.optimize(), parent2.optimize()
            new_generation += [child1, child2]
  
        population = new_generation
        population = sorted(population, key = lambda x:x.fitness)  
  
        best_fitness.append(population[0].fitness)
        if len(best_fitness) > 10 and best_fitness[-1] >= best_fitness[-2]: break
        generation += 1
  
    return population[0].chromosome, generation, population[0].fitness

