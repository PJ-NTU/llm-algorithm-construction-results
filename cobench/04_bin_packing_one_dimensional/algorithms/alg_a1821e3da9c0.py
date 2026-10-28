# Library slot 10/10  id=alg_a1821e3da9c0
# score=144.3  feasible_rate=1.0
# score_vec=[49.0, 215.0, 137.0, 156.0, 148.0, 199.0, 126.0, 96.0, 173.0, 144.0]

import time
import random

def decode(perm, weights, capacity, num_bins_max):
    """Decode a permutation of items into a feasible bin packing.
    Returns: (bin_of_item, used_bins, num_bins_used)
    """
    n = len(weights)
    bin_load = []  # total weight per bin
    bin_of_item = [-1] * n
    for idx in perm:
        w = weights[idx]
        placed = False
        # try existing bins in order
        for b, cur_load in enumerate(bin_load):
            if cur_load + w <= capacity:
                bin_load[b] += w
                bin_of_item[idx] = b
                placed = True
                break
        if not placed:
            # open new bin
            b = len(bin_load)
            bin_load.append(w)
            bin_of_item[idx] = b
    num_bins_used = len(bin_load)
    # convert to output format
    used_bins = [0] * num_bins_max
    for b in range(num_bins_used):
        used_bins[b] = 1
    return bin_of_item, used_bins, num_bins_used

def evaluate(perm, weights, capacity, num_bins_max):
    """Return (bin_of_item, used_bins, num_bins_used) for permutation."""
    return decode(perm, weights, capacity, num_bins_max)

def random_permutation(n):
    """Generate a random permutation of 0..n-1."""
    perm = list(range(n))
    random.shuffle(perm)
    return perm

def initial_population(pop_size, n, weights, capacity, num_bins_max):
    """Generate initial population, optionally including a greedy FFD heuristic."""
    pop = []
    # First-Fit Decreasing heuristic
    items_sorted = sorted(range(n), key=lambda i: weights[i], reverse=True)
    _, _, ffd_bins = decode(items_sorted, weights, capacity, num_bins_max)
    pop.append(items_sorted[:])  # copy
    # fill the rest with random permutations
    while len(pop) < pop_size:
        pop.append(random_permutation(n))
    return pop

def order_crossover(p1, p2):
    """Order crossover (OX). Returns two children."""
    n = len(p1)
    a = random.randint(0, n-1)
    b = random.randint(a+1, n)  # exclusive end
    if b >= n:
        b = n
    # child1: keep segment from p1[a:b], fill rest from p2 in order
    def make_child(parent_source, parent_fill, start, end):
        child = [None]*n
        used = set()
        for i in range(start, end):
            child[i] = parent_source[i]
            used.add(parent_source[i])
        # fill remaining positions from parent_fill
        fidx = 0
        for i in range(n):
            if child[i] is None:
                # find next item from parent_fill not already in child
                while parent_fill[fidx] in used:
                    fidx += 1
                child[i] = parent_fill[fidx]
                used.add(parent_fill[fidx])
                fidx += 1
        return child
    child1 = make_child(p1, p2, a, b)
    child2 = make_child(p2, p1, a, b)
    return child1, child2

def swap_mutate(perm, prob):
    """Swap two random elements with given probability."""
    if random.random() < prob:
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def tournament_selection(population, fitnesses, tournament_size):
    """Select one parent via tournament."""
    best = None
    best_fit = None
    for _ in range(tournament_size):
        idx = random.randrange(len(population))
        if best is None or fitnesses[idx] < best_fit:
            best = population[idx]
            best_fit = fitnesses[idx]
    return best[:]  # return a copy

def solve(instance, time_limit_s):
    # --- ADAPTED INPUT PARSING (target schema) ---
    n = instance["n"]                 # number of items (= max possible bins)
    capacity = instance["capacity"]
    weights = instance["weights"]     # list of length n

    num_items = n
    num_bins_max = n                  # maximum bins we may open

    # Parameter settings
    pop_size = min(100, max(20, num_items // 2))
    tour_size = 3
    mutation_prob = 0.1
    elitism_count = 1

    # Initialize population
    pop = initial_population(pop_size, num_items, weights, capacity, num_bins_max)
    # Evaluate initial population
    fitnesses = []
    solutions = []
    for perm in pop:
        _, used, nb = decode(perm, weights, capacity, num_bins_max)
        fitnesses.append(nb)
        solutions.append((perm[:], used, nb))
    # best so far
    best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
    best_perm, best_used, best_nb = solutions[best_idx]

    t0 = time.time()
    margin = 0.005  # small safety margin

    # Main GA loop
    generation = 0
    while time.time() - t0 < time_limit_s - margin:
        generation += 1
        new_pop = []
        new_fitnesses = []
        new_solutions = []

        # Elitism: keep best solution(s)
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        for k in range(elitism_count):
            idx = sorted_indices[k]
            new_pop.append(pop[idx][:])
            new_fitnesses.append(fitnesses[idx])
            new_solutions.append((pop[idx][:], solutions[idx][1], fitnesses[idx]))

        # Generate rest of population via crossover and mutation
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(pop, fitnesses, tour_size)
            parent2 = tournament_selection(pop, fitnesses, tour_size)
            child1, child2 = order_crossover(parent1, parent2)
            swap_mutate(child1, mutation_prob)
            swap_mutate(child2, mutation_prob)
            # Evaluate and keep if feasible (always feasible with our decode)
            for child in (child1, child2):
                if len(new_pop) >= pop_size:
                    break
                _, used_child, nb_child = decode(child, weights, capacity, num_bins_max)
                new_pop.append(child)
                new_fitnesses.append(nb_child)
                new_solutions.append((child, used_child, nb_child))

        # Replace old population
        pop = new_pop
        fitnesses = new_fitnesses
        solutions = new_solutions

        # Update best overall
        cur_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[cur_best_idx] < best_nb:
            best_perm, best_used, best_nb = solutions[cur_best_idx]

    # --- ADAPTED OUTPUT BUILDING (target schema) ---
    # Decode best_perm to get bin_of_item
    bin_of_item, _, _ = decode(best_perm, weights, capacity, num_bins_max)

    # Ensure bin indices are 0..num_bins_used-1 (already true)
    # Normalize if necessary (optional, but decode already produces contiguous indices)
    # Build the solution dict exactly as required
    return {
        "bin_activation": best_used,        # list of 0/1 of length n
        "item_assignments": bin_of_item     # list of ints of length n
    }