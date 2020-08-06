import random 
import torch
import numpy as np
from src.constants import *  
 
class Individual(object): 
    def __init__(self, dataset, f, bounds, data_type, chromosome=[]): 
        self.dataset = dataset
        self.f = f
        self.data_type = data_type
        self.chromosome = torch.tensor(chromosome if chromosome != [] else random.choice(dataset)[0], dtype=torch.float)
        self.fitness = self.cal_fitness() 
        self.bounds = bounds
  
    def mate(self, par2): 
        child_chromosome = [] 
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):       
            prob = random.random() 
            if prob < 0.45: 
                child_chromosome.append(gp1) 
            elif prob < 0.90: 
                child_chromosome.append(gp2) 
            else: 
                child_chromosome.append(random.choice(self.dataset)[0][len(child_chromosome)]) 
        if "a" in self.data_type :
            for i in range(4):
                child_chromosome[i] = max(self.bounds[i][0], min(child_chromosome[i], self.bounds[i][1]))
        else:
            alloc = []
            for i in child_chromosome:
                oneHot = [0] * 50; alist = i.tolist()[1:]
                oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
            child_chromosome = np.concatenate((self.chromosome[:,0].reshape(-1,1), np.array(alloc)), axis=1)
        return Individual(self.dataset, self.f, self.bounds, self.data_type, child_chromosome) 
  
    def cal_fitness(self): 
        res = self.f(self.chromosome)
        return CoeffCd * res[1] - CoeffCl * res[0] if "a" in self.data_type else res
  
def ga(dataset, f, bounds, data_type): 
    generation = 1
    population = [] 
    best_fitness = []
  
    for _ in range(POPULATION_SIZE): 
        population.append(Individual(dataset, f, bounds, data_type)) 
  
    while True: 
        population = sorted(population, key = lambda x:x.fitness) 
  
        new_generation = [] 
        new_generation.extend(population[:int(0.1*POPULATION_SIZE)]) 
  
        s = int(0.9*POPULATION_SIZE) 
        for _ in range(s): 
            parent1 = random.choice(population[:50]) 
            parent2 = random.choice(population[:50]) 
            child = parent1.mate(parent2) 
            new_generation.append(child) 
  
        population = new_generation 
  
        best_fitness.append(population[0].fitness)
        if len(best_fitness) > (100 if "a" in data_type else 10) and best_fitness[-1] >= best_fitness[-2]: break
        generation += 1
  
      
    print("Generation: {}\nResult: {}\nFitness: {}".format(generation, 
          "".join(str(population[0].chromosome)), 
          population[0].fitness)) 
