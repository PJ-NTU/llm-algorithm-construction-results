# Library slot 2/10  id=alg_b84e9eacc71f
# score=142.4  feasible_rate=1.0
# score_vec=[49.0, 211.0, 134.0, 156.0, 145.0, 196.0, 125.0, 95.0, 170.0, 143.0]

import time
import random
import copy

def solve(instance, time_limit_s):
    t0 = time.time()
    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    
    # Pre-sort items for some initial heuristics
    indexed_weights = sorted([(w, i) for i, w in enumerate(weights)], reverse=True)
    
    def get_fitness(bins):
        """
        Fitness function for Bin Packing GGA.
        Rewards bins that are nearly full. 
        Formula: sum((load/capacity)^k) / number_of_bins
        We use k=2 to strongly favor full bins.
        """
        if not bins:
            return 0
        f = sum((sum(weights[i] for i in b) / capacity) ** 2 for b in bins)
        return f / len(bins)

    def first_fit_decreasing():
        bins = []
        bin_loads = []
        item_to_bin = [0] * n
        for w, i in indexed_weights:
            placed = False
            for b_idx in range(len(bins)):
                if bin_loads[b_idx] + w <= capacity:
                    bins[b_idx].append(i)
                    bin_loads[b_idx] += w
                    item_to_bin[i] = b_idx
                    placed = True
                    break
            if not placed:
                item_to_bin[i] = len(bins)
                bins.append([i])
                bin_loads.append(w)
        return bins

    def first_fit_random(items_to_pack):
        # items_to_pack is a list of item indices
        random.shuffle(items_to_pack)
        bins = []
        bin_loads = []
        for i in items_to_pack:
            w = weights[i]
            placed = False
            for b_idx in range(len(bins)):
                if bin_loads[b_idx] + w <= capacity:
                    bins[b_idx].append(i)
                    bin_loads[b_idx] += w
                    placed = True
                    break
            if not placed:
                bins.append([i])
                bin_loads.append(w)
        return bins

    def crossover(p1, p2):
        # Select two crossover points in the bin list of p1
        if len(p1) < 2 or len(p2) < 2:
            return copy.deepcopy(p1)
        
        cp1, cp2 = sorted(random.sample(range(len(p1)), 2))
        child_bins = copy.deepcopy(p1[cp1:cp2+1])
        
        # Identify items already packed in child
        packed_items = {item for b in child_bins for item in b}
        
        # Add bins from p2 if they don't contain already packed items
        for b in p2:
            if not any(item in packed_items for item in b):
                child_bins.append(list(b))
                packed_items.update(b)
        
        # Re-pack missing items
        missing_items = [i for i in range(n) if i not in packed_items]
        if missing_items:
            # Sort missing items descending to fill better
            missing_items.sort(key=lambda x: weights[x], reverse=True)
            for i in missing_items:
                w = weights[i]
                placed = False
                for b in child_bins:
                    if sum(weights[idx] for idx in b) + w <= capacity:
                        b.append(i)
                        placed = True
                        break
                if not placed:
                    child_bins.append([i])
        return child_bins

    def mutate(sol):
        if not sol: return sol
        # Eliminate a few bins and re-pack their items
        num_to_eliminate = max(1, int(len(sol) * 0.1))
        indices = random.sample(range(len(sol)), num_to_eliminate)
        indices.sort(reverse=True)
        
        repack_items = []
        for idx in indices:
            repack_items.extend(sol.pop(idx))
            
        repack_items.sort(key=lambda x: weights[x], reverse=True)
        for i in repack_items:
            w = weights[i]
            placed = False
            for b in sol:
                if sum(weights[idx] for idx in b) + w <= capacity:
                    b.append(i)
                    placed = True
                    break
            if not placed:
                sol.append([i])
        return sol

    # Initial Population
    pop_size = 20
    population = [first_fit_decreasing()]
    for _ in range(pop_size - 1):
        population.append(first_fit_random(list(range(n))))
    
    best_sol = min(population, key=len)
    
    # Evolution Loop
    margin = 0.1
    while time.time() - t0 < time_limit_s - margin:
        # Sort population by fitness
        population.sort(key=get_fitness, reverse=True)
        
        new_population = population[:2] # Elitism
        
        while len(new_population) < pop_size and time.time() - t0 < time_limit_s - margin:
            # Selection (Tournament)
            p1 = min(random.sample(population, 3), key=len)
            p2 = min(random.sample(population, 3), key=len)
            
            child = crossover(p1, p2)
            
            if random.random() < 0.2:
                child = mutate(child)
            
            new_population.append(child)
            
            if len(child) < len(best_sol):
                best_sol = child
        
        population = new_population

    # Format result
    item_assignments = [0] * n
    bin_activation = [0] * n
    for b_idx, b_items in enumerate(best_sol):
        bin_activation[b_idx] = 1
        for item_idx in b_items:
            item_assignments[item_idx] = b_idx
            
    return {
        "bin_activation": bin_activation,
        "item_assignments": item_assignments
    }