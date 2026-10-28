"""Format adapter for schwerdfeger2016 (identical parallel machines, normalised sum of squared workload deviations).

Official instance: {"num_machines", "num_jobs", "processing_times", "total_processing_time",
                    "average_machine_completion_time", "processing_time_range", "processing_time_distribution",
                    "generation_iteration", ...}.
Task format:       the same fields under the same names (jobs and machines keep their 0-based indices).

Official solution: {"objective_value", "assignment": {"<machine>": [job, ...]}, "machine_completion_times":
{"<machine>": C_i}} with one key per machine 0..num_machines-1, which is also the task's solution shape. Completion
times and the objective (1/mu) * sqrt(sum_i (C_i - mu)^2), mu = sum(p) / m, are recomputed from the assignment.
"""
import math

FIELDS = ("num_machines", "num_jobs", "processing_times", "total_processing_time", "average_machine_completion_time",
          "processing_time_range", "processing_time_distribution", "generation_iteration")


def to_ours(official_instance: dict) -> dict:
    return {k: official_instance.get(k) for k in FIELDS}


def to_official(official_instance: dict, our_solution: dict) -> dict:
    p = official_instance["processing_times"]
    given = our_solution["assignment"]
    assignment = {}
    for i in range(official_instance["num_machines"]):
        jobs = given.get(str(i), given.get(i, []))
        assignment[str(i)] = [int(j) for j in jobs]
    completion = {k: float(sum(p[j] for j in jobs if 0 <= j < len(p))) for k, jobs in assignment.items()}
    solution = {"assignment": assignment, "machine_completion_times": completion}
    solution["objective_value"] = official_objective(official_instance, solution)
    return solution


def official_objective(official_instance: dict, official_solution: dict) -> float:
    p = official_instance["processing_times"]
    mu = sum(p) / official_instance["num_machines"]
    loads = [sum(p[j] for j in jobs if 0 <= j < len(p)) for jobs in official_solution["assignment"].values()]
    return math.sqrt(sum((c - mu) ** 2 for c in loads)) / mu
