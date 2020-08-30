import numpy as np
from net import Network
from sim import *

POP_SIZE = 50
N_BESTS = int(POP_SIZE*0.1)
EPOCHS = 50
ROUNDS = 100
HIDDEN_SIZE = [5]
RUNS_PER_GENOME = 10

def get_bests(scores, n):
    best_genome_indexes = np.argsort(scores)[::-1]

    return best_genome_indexes[:n]

def log_change(bests, pop_size, proportion_log=0.5):
    vectors = []
    log_space = int((pop_size - len(bests))*proportion_log)
    random_uniform = pop_size - log_space

    for i, j in zip(np.logspace(-4, -1, log_space), range(len(bests))):
        vectors.append(bests[j] + (np.random.uniform(-1, 1, len(bests[j]))*i)*bests[j])

    for _ in range(random_uniform):
        vectors.append(np.random.uniform(-1, 1, len(bests[0])))

    for i in range(len(bests)):
        vectors.append(bests[i])

    return np.array(vectors)

def main():
    # generate the population
    population = []
    for _ in range(POP_SIZE):
        population.append(Network(np.concatenate(([2], HIDDEN_SIZE, [2])))) 
        # 2 input = distance between attacker and goal, angle between attacker and goal;
        # 2 output = distance of GK, angle of GK
    # repeat N times
    for epoch in range(EPOCHS):
        print("Epoch: ", epoch)
        # iterate over genomes and call simulation
        scores = []
        for g in population:
            means = []
            # store score for each genome
            for _ in range(RUNS_PER_GENOME):
                means.append(simulation(g, rounds=ROUNDS))
            
            scores.append(np.mean(means))
        
        # select the n best genomes
        #idx_bests = np.argmax(scores)
        idx_bests = get_bests(scores, n=N_BESTS) # n of best genomes
        print("Best score: ", scores[idx_bests[0]])
        bests = []
        for i in idx_bests:
            bests.append(population[i].export())
        # call log_change on best genomes
        new_weights = log_change(bests, POP_SIZE, proportion_log=0.5)

        # generate new population based on log_change
        for i, nw in enumerate(new_weights):
            population[i]._import(nw)
    
    # save to file
    for i, p in enumerate(population):
        if i == idx_bests[0]:
            p.export().tofile(f"saves/best_net.txt", sep=" ")
        else:
            p.export().tofile(f"saves/net_{i}.txt", sep=" ")


   

if __name__ == '__main__':
    main()