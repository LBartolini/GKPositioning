import numpy as np
from net import Network
from sim import *

POP_SIZE = 50
EPOCHS = 500
HIDDEN_SIZE = [1, 3, 1]

def get_bests(scores, n):
    best_genome_indexes = np.argsort(scores)[::-1]

    return best_genome_indexes[:n]


def log_change(best_five):
    vectors = []

    for i, j in zip(np.logspace(-5, -1, 4), range(len(best_five))):
        vectors.append(best_five[j] + (np.random.uniform(-1, 1, len(best_five[j]))*i)*best_five[j])

    for _ in range(25):
        vectors.append(np.random.uniform(-1, 1, len(best_five[0])))

    for i in range(len(best_five)):
        vectors.append(best_five[i])

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
            # store score for each genome
            scores.append(simulation(g, rounds=500))
        
        # select the n best genomes
        #idx_bests = np.argmax(scores)
        idx_bests = get_bests(scores, n=5) # n of best genomes
        print("Best score: ", scores[idx_bests[0]])
        bests = []
        for i in idx_bests:
            bests.append(population[i].export())
        # call log_change on best genomes
        new_weights = log_change(bests)

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