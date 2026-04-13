import joblib
import random
import math
import pandas as pd
import numpy as np
import warnings
import gc
import csv
import os
import time
from joblib import Parallel, delayed  # <-- Added for parallel processing
from pymoo.indicators.hv import HV
import matplotlib

matplotlib.use('Agg')  # Prevents headless display crashes

warnings.filterwarnings('ignore')


# ==========================================
# AI BRAIN & CONTROLLER (Shared)
# ==========================================
class AIBrain:
    def __init__(self):
        self.model = None
        self.loaded = False
        try:
            self.model = joblib.load("honest_ai_model.pkl")
            self.loaded = True
            print("✅ AI Brain Loaded.")
        except:
            print("⚠️ Brain not found. Using Heuristic Fallback.")
            self.loaded = False

    def predict_rate(self, n_var, n_obj, nb_gen, n_constr, pop_size):
        default_rate = 1.0 / n_var
        if not self.loaded:
            if n_constr > 0: return default_rate * 2.0
            return default_rate

        k_values = [1.5, 2, 2.5, 3, 4, 5]
        test_rates = [k / n_var for k in k_values if k / n_var <= 0.5]
        if not test_rates: return default_rate

        rows = [{
            'Population Size': pop_size, 'Archive Size': pop_size,
            'Constraints Number': n_constr, 'Objectives Number': n_obj,
            'Decision Variables Number': n_var, 'Crossover Rate': 0.0,
            'Number of Generations': nb_gen, 'Mutation Rate': rate,
            'Complexity_Index': n_obj * n_var,
            'Constraint_Density': n_constr / (n_var + 1e-5),
            'Mutation_Strength': rate * n_var
        } for rate in test_rates]

        try:
            predicted_scores = self.model.predict(pd.DataFrame(rows))
            return test_rates[np.argmax(predicted_scores)]
        except:
            return default_rate


AI = AIBrain()


class TriPhaseController:
    def __init__(self, initial_pm, n_vars):
        self.pm = initial_pm
        self.min_pm = max(0.05, 1.0 / n_vars)
        self.max_pm = 0.30
        self.stagnation_counter = 0
        self.stagnation_limit = 10
        self.in_sos_mode = False
        self.sos_duration = 0
        self.max_sos_duration = 3

    def update(self, success_rate):
        if self.in_sos_mode:
            self.sos_duration += 1
            if self.sos_duration >= self.max_sos_duration:
                self.in_sos_mode = False
                self.sos_duration = 0
                self.pm = max(self.min_pm * 2, 0.15)
            return self.pm

        if success_rate > 0.15:
            self.pm *= 1.05
            self.stagnation_counter = 0
        elif success_rate < 0.02:
            self.pm *= 0.92
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        self.pm = max(self.min_pm, min(0.25, self.pm))

        if self.stagnation_counter >= self.stagnation_limit:
            self.in_sos_mode = True
            self.pm = self.max_pm
            self.stagnation_counter = 0
        return self.pm


# ==========================================
# CMOEA/D-DMA 
# ==========================================
class Individual:
    def __init__(self, m, p_name='m-cdtlz'):
        self.p_name = p_name
        self.m = m
        self.X, self.g, self.f = [], [], []
        self.feasible = False
        self.eta_m = 20.0
        self.n = m * 10
        self.bounds = [(0, 1)] * self.n

    def d_variables(self):
        self.X = [random.uniform(b[0], b[1]) for b in self.bounds]

    def fitness(self, m):
        self.f = []
        for i in range(1, m + 1):
            s = 0.0
            k = int((i - 1) * (self.n / m))
            r = int(i * (self.n / m))
            for l in range(k, r):
                s += self.X[l] ** 0.5
            self.f.append((1.0 / (self.n / m)) * s)

    def constraints_g(self, m):
        self.g = []
        for j in range(len(self.f)):
            d = sum(self.f[l] ** 2 for l in range(len(self.f)) if l != j)
            g1 = (self.f[j] ** 2) + (4 * d) - 1
            self.g.append(g1)

    def feasibility(self):
        s = sum(abs(item) for item in self.g if item < 0)
        self.feasible = (s <= 1e-6)


class Population:
    def __init__(self, n, t, q, alpha, nb_g, Pc, Pm, p_name="m-cdtlz"):
        self.p_name = p_name
        self.N, self.T, self.m, self.Alph = n, t, q, alpha
        self.nbGen = nb_g
        self.Pc, self.Pm = Pc, Pm
        self.final_pm = Pm
        self.individuals = []
        self.weights, self.neighbours, self.z = [], [], []
        self.A = [[] for _ in range(n)]

    def generateWeightVectors(self):
        if self.m == 2:
            self.weights = [[i / (self.N - 1), 1.0 - i / (self.N - 1)] for i in range(self.N)]
        else:
            self.weights = np.random.dirichlet(np.ones(self.m), self.N).tolist()

    def generate_neighbors(self):
        for i in range(self.N):
            dists = [(math.sqrt(sum([(a - b) ** 2 for a, b in zip(self.weights[i], w)])), j) for j, w in
                     enumerate(self.weights)]
            dists.sort()
            self.neighbours.append([d[1] for d in dists[:self.T]])

    def init_Z(self):
        self.z = [float('inf')] * self.m

    def Tchebychef(self, x, ind):
        return max([self.weights[ind][i] * abs(x.f[i] - self.z[i]) for i in range(self.m)])

    def dominance(self, x, y):
        return all(x.f[i] <= y.f[i] for i in range(self.m)) and any(x.f[i] < y.f[i] for i in range(self.m))

    def makeInitialPopulation(self):
        for _ in range(self.N):
            idiv = Individual(self.m, self.p_name)
            idiv.d_variables()
            idiv.fitness(self.m)
            idiv.constraints_g(self.m)
            idiv.feasibility()
            self.individuals.append(idiv)

    def best_point_z(self, y):
        for i in range(self.m):
            if y.feasible and self.z[i] > y.f[i]:
                self.z[i] = y.f[i]

    def exchange(self, y, k):
        self.individuals[k] = y

    def Update_Solution_Archive(self, y, ind, alpha):
        improved = False
        for k in self.neighbours[ind]:
            r = self.Tchebychef(y, k)
            r1 = self.Tchebychef(self.individuals[k], k)
            if self.individuals[k].feasible and y.feasible and (r < r1):
                self.exchange(y, k)
                improved = True
            elif not self.individuals[k].feasible and y.feasible:
                self.exchange(y, k)
                improved = True
        return improved

    def gentic_operators(self, x, y, Pm, Pc):
        def poly_mut(ind_in, rate):
            new_ind = Individual(self.m, self.p_name)
            new_ind.X = list(ind_in.X)
            eta = 20.0
            for i in range(len(new_ind.X)):
                if random.random() < rate:
                    u = random.random()
                    beta = (2 * u) ** (1 / (eta + 1)) - 1 if u <= 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    new_ind.X[i] = max(0, min(1, new_ind.X[i] + beta))
            new_ind.fitness(self.m)
            new_ind.constraints_g(self.m)
            new_ind.feasibility()
            return new_ind

        child = poly_mut(x, Pm)
        return child

    def CMOEA_D_DMA(self, strategy_type="Fixed-Rate"):
        self.generateWeightVectors()
        self.generate_neighbors()
        self.makeInitialPopulation()
        self.init_Z()
        for ind in self.individuals: self.best_point_z(ind)

        controller = TriPhaseController(self.Pm, self.individuals[0].n)

        for j in range(self.nbGen):
            success_count = 0
            for i in range(self.N):
                pa = self.individuals[i]
                pb = self.individuals[random.choice(self.neighbours[i])]

                y = self.gentic_operators(pa, pb, self.Pm, self.Pc)
                self.best_point_z(y)
                if self.Update_Solution_Archive(y, i, self.Alph):
                    success_count += 1

            if strategy_type == "AI-Adaptive":
                self.Pm = controller.update(success_count / self.N)

        self.final_pm = self.Pm
        return self.individuals


# ==========================================
# CNSGA-II 
# ==========================================
class MCDTLZProblem:
    def __init__(self, num_obj):
        self.n_objs = num_obj
        self.n_vars = num_obj * 10
        self.n_constrs = num_obj
        self.vars_lb = np.zeros(self.n_vars)
        self.vars_ub = np.ones(self.n_vars)

    def evaluate_population(self, pop_X):
        n_pop = pop_X.shape[0]
        split_indices = np.linspace(0, self.n_vars, self.n_objs + 1, dtype=int)

        pop_f = np.zeros((n_pop, self.n_objs))
        for i in range(self.n_objs):
            start, end = split_indices[i], split_indices[i + 1]
            pop_f[:, i] = np.sum(pop_X[:, start:end] ** 0.5, axis=1) / (end - start)

        pop_g = np.zeros((n_pop, self.n_constrs))
        for j in range(self.n_constrs):
            mask = np.arange(self.n_objs) != j
            pop_g[:, j] = pop_f[:, j] ** 2 + 4 * np.sum(pop_f[:, mask] ** 2, axis=1) - 1

        pop_omega = np.sum(np.maximum(0, pop_g), axis=1)
        return pop_f, pop_g, pop_omega


class NSGA2_Individual:
    def __init__(self, num_vars, num_obj):
        self.X = np.random.uniform(0, 1, num_vars)
        self.f, self.g = np.zeros(num_obj), np.zeros(num_obj)
        self.omega, self.crowding_distance, self.domination_count = 0, 0, 0
        self.feasible, self.is_offspring = False, False
        self.rank = None
        self.dominated_solutions = []


class CNSGA2:
    def __init__(self, pop_size, max_gen, problem, pm_rate, strategy_type="Fixed-Rate"):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.problem = problem
        self.mutation_prob = pm_rate
        self.strategy_type = strategy_type
        self.final_pm = pm_rate
        if self.strategy_type == "AI-Adaptive":
            self.controller = TriPhaseController(initial_pm=pm_rate, n_vars=problem.n_vars)

    def run(self):
        pop = [NSGA2_Individual(self.problem.n_vars, self.problem.n_objs) for _ in range(self.pop_size)]
        pop_X = np.array([ind.X for ind in pop])
        pop_f, pop_g, pop_omega = self.problem.evaluate_population(pop_X)
        for i, ind in enumerate(pop):
            ind.f, ind.g, ind.omega = pop_f[i], pop_g[i], pop_omega[i]
            ind.feasible = np.all(ind.g <= 0)

        for gen in range(self.max_gen):
            fronts = self.fast_non_dominated_sort(pop)
            for front in fronts: self.crowding_distance_assignment(front)

            offspring = []
            while len(offspring) < self.pop_size:
                p1, p2 = random.sample(pop, 2)
                child = NSGA2_Individual(self.problem.n_vars, self.problem.n_objs)
                child.X = p1.X.copy()
                child = self.polynomial_mutation(child)
                child.is_offspring = True
                offspring.append(child)

            off_X = np.array([ind.X for ind in offspring])
            off_f, off_g, off_omega = self.problem.evaluate_population(off_X)
            for i, ind in enumerate(offspring):
                ind.f, ind.g, ind.omega = off_f[i], off_g[i], off_omega[i]
                ind.feasible = np.all(ind.g <= 0)

            combined = pop + offspring
            fronts = self.fast_non_dominated_sort(combined)
            pop = self.environmental_selection(fronts)

            if self.strategy_type == "AI-Adaptive":
                success_rate = sum(1 for ind in pop if ind.is_offspring) / self.pop_size
                self.mutation_prob = self.controller.update(success_rate)

            for ind in pop: ind.is_offspring = False

        self.final_pm = self.mutation_prob
        return [ind for ind in pop if ind.rank == 0]

    def polynomial_mutation(self, ind):
        mutant = NSGA2_Individual(self.problem.n_vars, self.problem.n_objs)
        mutant.X = ind.X.copy()
        eta_m = 20
        u = np.random.rand(self.problem.n_vars)
        delta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
        mask = np.random.rand(self.problem.n_vars) < self.mutation_prob
        mutant.X += np.where(mask, delta * (self.problem.vars_ub - self.problem.vars_lb), 0)
        mutant.X = np.clip(mutant.X, self.problem.vars_lb, self.problem.vars_ub)
        return mutant

    def fast_non_dominated_sort(self, pop):
        fronts = [[]]
        for i, ind in enumerate(pop):
            ind.domination_count = 0
            ind.dominated_solutions = []
            for other in pop:
                if self.dominates(ind, other):
                    ind.dominated_solutions.append(other)
                elif self.dominates(other, ind):
                    ind.domination_count += 1
            if ind.domination_count == 0:
                ind.rank = 0
                fronts[0].append(ind)
        i = 0
        while fronts[i]:
            next_front = []
            for ind in fronts[i]:
                for dom in ind.dominated_solutions:
                    dom.domination_count -= 1
                    if dom.domination_count == 0:
                        dom.rank = i + 1
                        next_front.append(dom)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def dominates(self, a, b):
        if a.feasible and not b.feasible: return True
        if not a.feasible and b.feasible: return False
        if not a.feasible and not b.feasible: return a.omega < b.omega
        return np.all(a.f <= b.f) and np.any(a.f < b.f)

    def crowding_distance_assignment(self, front):
        if not front: return
        num_obj = len(front[0].f)
        front_f = np.array([ind.f for ind in front])
        for m in range(num_obj):
            sorted_idx = np.argsort(front_f[:, m])
            front[sorted_idx[0]].crowding_distance = float('inf')
            front[sorted_idx[-1]].crowding_distance = float('inf')
            norm = front_f[sorted_idx[-1], m] - front_f[sorted_idx[0], m]
            if norm == 0: norm = 1
            for i in range(1, len(sorted_idx) - 1):
                front[sorted_idx[i]].crowding_distance += (front_f[sorted_idx[i + 1], m] - front_f[
                    sorted_idx[i - 1], m]) / norm

    def environmental_selection(self, fronts):
        new_pop, remaining = [], self.pop_size
        for front in fronts:
            if len(front) <= remaining:
                new_pop.extend(front)
                remaining -= len(front)
            else:
                front.sort(key=lambda x: -x.crowding_distance)
                new_pop.extend(front[:remaining])
                break
        return new_pop


# ==========================================
# PARALLEL EXECUTION HELPER
# ==========================================
def execute_single_run(alg, m, pop_size, nb_generation, strategy, pm_rate):
    start_time = time.time()
    if "CMOEA/D-DMA" in alg:
        pop = Population(pop_size, 10, m, 3, nb_generation, 0.0, pm_rate, "m-cdtlz")
        res = pop.CMOEA_D_DMA(strategy_type=strategy)
        f_vals = np.array([ind.f for ind in res if ind.feasible])
    elif "CNSGA2" in alg:
        prob = MCDTLZProblem(m)
        nsga2 = CNSGA2(pop_size, nb_generation, prob, pm_rate, strategy_type=strategy)
        res = nsga2.run()
        f_vals = np.array([ind.f for ind in res if ind.feasible])

    return f_vals, time.time() - start_time


# ==========================================
# MAIN EXECUTION (OBJECTIVE SCALING)
# ==========================================
if __name__ == '__main__':
    # 📉 drastically lowered parameters for speed
    m_values = [3, 5, 7, 8, 10]
    pop_size = 100  # Slashed from 200 to prevent NSGA-II sorting bottleneck
    nb_generation = 500  # Lowered from 1000. 300 is usually enough to establish a trend.
    runs_per_setting = 15  # Dropped from 30. 15 is perfectly fine for research constraint handling.

    csv_filename = "objective_scaling_results.csv"

    print("🚀 Starting M-Objective Scaling Experiment (MOEA/D vs NSGA-II)")
    print(f"⚙️  Settings: Pop={pop_size}, Gens={nb_generation}, Runs={runs_per_setting}\n")

    algorithms = ["CMOEA/D-DMA conventional", "CMOEA/D-DMA AI", "CNSGA2 conventional", "CNSGA2 AI"]

    for m in m_values:
        print("=" * 60)
        print(f"📐 TESTING OBJECTIVES: m = {m}")

        n_v, n_c = m * 10, m
        fixed_pm_val = 1.0 / n_v
        ai_pm_initial = AI.predict_rate(n_v, m, nb_generation, n_c, pop_size)
        print(f"   Baseline Pm: {fixed_pm_val:.5f} | AI Initial Pm: {ai_pm_initial:.5f}")

        run_data_vault = {alg: [] for alg in algorithms}
        all_feasible_f_vals = []

        # --- PARALLEL EXECUTION PHASE ---
        for alg in algorithms:
            print(f"\n Running: {alg} (Parallel Processing...)")
            strategy = "AI-Adaptive" if "AI" in alg else "Fixed-Rate"
            pm_rate = ai_pm_initial if "AI" in alg else fixed_pm_val

            # 🚀 joblib.Parallel handles all runs simultaneously utilizing all CPU Cores
            results = Parallel(n_jobs=-1)(
                delayed(execute_single_run)(alg, m, pop_size, nb_generation, strategy, pm_rate)
                for _ in range(runs_per_setting)
            )

            # Unpack results
            for f_vals, exec_time in results:
                if len(f_vals) > 0: all_feasible_f_vals.append(f_vals)
                run_data_vault[alg].append({'f_vals': f_vals, 'exec_time': exec_time})

        # --- GLOBAL NORMALIZATION PHASE ---
        global_min, global_max, denom = None, None, None
        if len(all_feasible_f_vals) > 0:
            stacked_f = np.vstack(all_feasible_f_vals)
            global_min = stacked_f.min(axis=0)
            global_max = stacked_f.max(axis=0)
            denom = global_max - global_min
            denom[denom == 0] = 1.0
            print(f"\n Global Bounds (m={m}) Generated.")
        else:
            print(f"\n  No feasible solutions found for m={m}.")

        # --- HV CALCULATION & SAVING ---
        for alg in algorithms:
            hv_list = []
            for run_data in run_data_vault[alg]:
                f_vals = run_data['f_vals']
                run_hv = 0.0
                if len(f_vals) > 0 and global_min is not None:
                    f_norm = (f_vals - global_min) / denom
                    try:
                        run_hv = HV(ref_point=np.array([1.1] * m)).do(f_norm)
                    except:
                        pass
                hv_list.append(run_hv)

            mean_hv, std_hv = np.mean(hv_list), np.std(hv_list)
            print(f"      ✅ {alg}: Mean HV = {mean_hv:.4f} | Std = {std_hv:.4e}")

            result_row = {'Objectives_m': m, 'Algorithm': alg, 'Mean_HV': mean_hv, 'Std_HV': std_hv}
            file_exists = os.path.isfile(csv_filename)
            with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=result_row.keys())
                if not file_exists: w.writeheader()
                w.writerow(result_row)

    print(f"\nAll M-Objective experiments complete! Data saved to '{csv_filename}'.")
