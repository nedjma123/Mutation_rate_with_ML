import joblib
import random
import math
import pandas as pd
import numpy as np
import warnings
import gc
import csv
import sys
import os
import time
from pymoo.indicators.hv import HV
from scipy.stats import mannwhitneyu
import matplotlib

matplotlib.use('Agg')  
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==========================================
#  ACTIVE LEARNING LOGGER
# ==========================================
def active_learning_logger(
        problem, problem_type, algorithm, pop_size, n_constr, n_obj, archive_size,
        mutation_type, crossover_rate, crossover_type, n_gen, n_var, ref_point,
        exec_time, hypervolume, target_mutation_rate, filename="CMOEA_DMA_Randomized_Dataset_1.csv"
):
    file_exists = os.path.isfile(filename)
    data_row = {
        'Problem': problem, 'Problem Type': problem_type, 'Algorithm': algorithm,
        'Population Size': pop_size, 'Constraints Number': n_constr, 'Objectives Number': n_obj,
        'Archive Size': archive_size, 'Mutation Type': mutation_type, 'Crossover Rate': crossover_rate,
        'Crossover Type': crossover_type, 'Number of Generations': n_gen, 'Decision Variables Number': n_var,
        'Reference Point': str(ref_point), 'Execution Time': round(exec_time, 4),
        'Hypervolume': hypervolume, 'Mutation Rate': target_mutation_rate
    }
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data_row.keys()))
            if not file_exists: writer.writeheader()
            writer.writerow(data_row)
        print(f"      📝 Active Learning: Added champion run (HV={hypervolume:.4f}) to '{filename}'.")
    except Exception as e:
        print(f"⚠️ Logger failed to write: {e}")


# ==========================================
# 2. AI BRAIN
# ==========================================
class AIBrain:
    def __init__(self):
        self.model = None
        self.loaded = False
        try:
            self.model = joblib.load("honest_ai_model.pkl")
            self.loaded = True
            print("AI Brain Loaded.")
        except:
            print("Brain not found. Using Heuristic Fallback.")
            self.loaded = False

    def predict_rate(self, n_var, n_obj, nb_Gen, n_constr, pop_size):
        default_rate = 1.0 / n_var
        if not self.loaded:
            if n_constr > 0: return default_rate * 2.0
            return default_rate

        k_values = [1.5, 2, 2.5, 3, 4, 5]
        test_rates = [k / n_var for k in k_values]
        test_rates = [r for r in test_rates if r <= 0.5]
        if not test_rates: return default_rate

        rows = []
        for rate in test_rates:
            rows.append({
                'Population Size': pop_size, 'Archive Size': pop_size,
                'Constraints Number': n_constr, 'Objectives Number': n_obj,
                'Decision Variables Number': n_var, 'Crossover Rate': 1.0,
                'Number of Generations': nb_Gen, 'Mutation Rate': rate,
                'Complexity_Index': n_obj * n_var,
                'Constraint_Density': n_constr / (n_var + 1e-5),
                'Mutation_Strength': rate * n_var
            })

        input_data = pd.DataFrame(rows)
        try:
            predicted_scores = self.model.predict(input_data)
            best_idx = np.argmax(predicted_scores)
            return test_rates[best_idx]
        except:
            return default_rate

AI = AIBrain()


# ==========================================
# TRI-PHASE CONTROLLER
# ==========================================
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
#  INDIVIDUAL CLASS
# ==========================================
class Individual:
    def __init__(self, m, p_name='m-cdtlz'):
        self.p_name = p_name
        self.m = m
        self.X = []
        self.g = []
        self.f = []
        self.feasible = False
        self.eta_m = random.uniform(20.0, 100.0)

        if "Welded Beam" in p_name:
            self.n = 4
            self.bounds = [(0.125, 5), (0.1, 10), (0.1, 10), (0.125, 5)]
        elif "Pressure Vessel" in p_name:
            self.n = 4
            self.bounds = [(0, 99), (0, 99), (10, 200), (10, 200)]
        elif "BICOP1" in p_name or "LIR-CMOP" in p_name:
            self.n = 30
            self.bounds = [(0, 1)] * self.n
        else:
            self.n = m * 10
            self.bounds = [(0, 1)] * self.n

    def d_variables(self):
        self.X = [random.uniform(b[0], b[1]) for b in self.bounds]

    def fitness(self, m):
        self.f = []
        x = self.X
        if "Welded Beam" in self.p_name:
            f1 = 1.10471 * (x[0] ** 2) * x[1] + 0.04811 * x[2] * x[3] * (14 + x[1])
            delta = (4 * 6000 * (14 ** 3)) / (30e6 * x[3] * (x[2] ** 3))
            self.f = [f1, delta]
        elif "Pressure Vessel" in self.p_name:
            f1 = (0.6224 * x[0] * x[2] * x[3]) + (1.7781 * x[1] * (x[2] ** 2)) + (3.1661 * (x[0] ** 2) * x[3]) + (19.84 * (x[0] ** 2) * x[2])
            volume = (math.pi * (x[2] ** 2) * x[3]) + ((4 / 3) * math.pi * (x[2] ** 3))
            self.f = [f1, -volume]
        elif "LIR-CMOP2" in self.p_name:
            g_val = 1.0 + 9.0 * sum(x[1:]) / (self.n - 1)
            f1 = x[0]
            f2 = g_val * (1.0 - math.sqrt(f1 / g_val))
            self.f = [f1, f2]
        elif "BICOP1" in self.p_name:
            g_val = 1.0 + 9.0 * sum(x[1:]) / (self.n - 1)
            f1 = g_val * x[0]
            val = f1 / g_val
            if val < 0: val = 0.0
            f2 = g_val * (1.0 - math.sqrt(val))
            self.f = [f1, f2]
        else:
            for i in range(1, m + 1):
                s = 0.0
                k = int((i - 1) * (self.n / m))
                r = int(i * (self.n / m))
                for l in range(k, r):
                    s += self.X[l] ** 0.5
                self.f.append((1.0 / (self.n / m)) * s)

    def constraints_g(self, m):
        self.g = []
        x = self.X
        if "Welded Beam" in self.p_name:
            h, l, t, b = x
            P, L, E, G = 6000, 14, 30e6, 12e6
            tau_prime = P / (math.sqrt(2) * h * l)
            R = math.sqrt(l ** 2 / 4 + ((h + t) / 2) ** 2)
            J = 2 * (math.sqrt(2) * h * l * (l ** 2 / 12 + ((h + t) / 2) ** 2)) + 1e-6
            tau = math.sqrt(tau_prime ** 2 + 2 * tau_prime * (P * (L + l / 2) * R / J) * l / (2 * R) + ((P * (L + l / 2) * R / J) ** 2))
            sigma = 6 * P * L / (b * t ** 2)
            pc = 4.013 * E * math.sqrt(t ** 2 * b ** 6 / 36) / L ** 2 * (1 - t / (2 * L) * math.sqrt(E / (4 * G)))
            self.g = [13600 - tau, 30000 - sigma, b - h, pc - P, h - 0.125, l - 0.1, t - 0.1, b - 0.125]
        elif "Pressure Vessel" in self.p_name:
            self.g = [x[0] - 0.0193 * x[2], x[1] - 0.00954 * x[2], (math.pi * x[2] ** 2 * x[3] + 4 / 3 * math.pi * x[2] ** 3) - 1296000, 240 - x[3]]
        elif "LIR-CMOP2" in self.p_name:
            c1 = 0.25 - (((self.f[0] - 0.5) ** 2) + ((self.f[1] - 0.5) ** 2))
            c2 = (((self.f[0] - 0.6) ** 2) + ((self.f[1] - 0.6) ** 2)) - 0.1
            self.g = [c1, c2]
        elif "BICOP1" in self.p_name:
            self.g = [0.0]
        else:
            for i in range(len(self.f)):
                r = []
                for j in range(len(self.f)):
                    d = 0.0
                    for l in range(len(self.f)):
                        if l != j: d = d + (self.f[l] ** 2)
                    g1 = ((self.f[j] ** 2) + (4 * d)) - 1
                    r.append(g1)
                self.g.append(r)

    def feasibility(self):
        s = 0.0
        for item in self.g:
            if isinstance(item, list):
                for val in item:
                    if val < 0: s += abs(val)
            else:
                if item < 0: s += abs(item)
        self.feasible = (s <= 1e-6)


# ==========================================
# 5. POPULATION CLASS
# ==========================================
class Population:
    def __init__(self, n, t, q, alpha, nb_g, Pc, Pm, p_name="m-cdtlz"):
        self.p_name = p_name
        self.individuals = []
        self.N = n
        self.weights = []
        self.T = t
        self.neighbours = []
        self.z = []
        self.A = [[] for _ in range(n)]
        self.Alph = alpha
        self.m = q
        self.nbGen = nb_g
        self.Pc = Pc
        self.Pm = Pm
        self.final_pm = Pm

    def generateWeightVectors(self):
        if self.m == 2:
            self.weights = [[i / (self.N - 1), 1.0 - i / (self.N - 1)] for i in range(self.N)]
        else:
            self.weights = np.random.dirichlet(np.ones(self.m), self.N).tolist()

    def generate_neighbors(self):
        for i in range(self.N):
            dists = [(math.sqrt(sum([(a - b) ** 2 for a, b in zip(self.weights[i], w)])), j) for j, w in enumerate(self.weights)]
            dists.sort()
            self.neighbours.append([d[1] for d in dists[:self.T]])

    def init_Z(self):
        self.z = [float('inf')] * self.m

    def Tchebychef(self, x, ind):
        return max([self.weights[ind][i] * abs(x.f[i] - self.z[i]) for i in range(self.m)])

    def dominance(self, x, y):
        x_f = x.f
        y_f = y.f
        not_worse = all(x_f[i] <= y_f[i] for i in range(len(x_f)))
        strictly_better = any(x_f[i] < y_f[i] for i in range(len(x_f)))
        return not_worse and strictly_better

    def makeInitialPopulation(self):
        for i in range(self.N):
            idiv = Individual(self.m, self.p_name)
            idiv.d_variables()
            idiv.fitness(self.m)
            idiv.constraints_g(self.m)
            idiv.feasibility()
            self.individuals.append(idiv)

    def best_point_z(self, y):
        for i in range(self.m):
            y_f = y.f[i]
            if y.feasible:
                self.z[i] = y_f
                for j in range(self.N):
                    if self.individuals[j].feasible:
                        current_f = self.individuals[j].f[i]
                        if self.z[i] > current_f: self.z[i] = current_f
            else:
                self.z[i] = self.individuals[0].f[i]
                for j in range(1, self.N):
                    if self.individuals[j].feasible:
                        current_f = self.individuals[j].f[i]
                        if self.z[i] > current_f: self.z[i] = current_f
                if not any(ind.feasible for ind in self.individuals):
                    if self.z[i] > y_f: self.z[i] = y_f
                    for j in range(1, self.N):
                        current_f = self.individuals[j].f[i]
                        if self.z[i] > current_f: self.z[i] = current_f

    def exchange(self, y, k):
        self.individuals[k] = y

    def Update_Solution_Archive(self, y, ind, alpha):
        improved = False
        d = self.neighbours[ind]
        for i in range(len(d)):
            k = d[i]
            r = self.Tchebychef(y, k)
            r1 = self.Tchebychef(self.individuals[k], k)
            
            if self.individuals[k].feasible and y.feasible and (r < r1):
                self.exchange(y, k)
                self.A[k] = [sol for sol in self.A[k] if not self.dominance(y, sol)]
                improved = True
            elif self.individuals[k].feasible and not y.feasible and (r < r1):
                if len(self.A[k]) < alpha:
                    self.A[k].append(y)
                else:
                    worst_idx = np.argmax([self.Tchebychef(sol, k) for sol in self.A[k]])
                    self.A[k][worst_idx] = y
            elif not self.individuals[k].feasible and y.feasible:
                self.exchange(y, k)
                improved = True
            elif not self.individuals[k].feasible and not y.feasible:
                viol_y = sum([abs(v) for item in y.g for v in (item if isinstance(item, list) else [item])] if y.g else [0])
                viol_xj = sum([abs(v) for item in self.individuals[k].g for v in (item if isinstance(item, list) else [item])] if self.individuals[k].g else [0])
                if viol_y < viol_xj:
                    self.exchange(y, k)
                    improved = True
                elif viol_y == viol_xj and (r < r1):
                    self.exchange(y, k)
                    improved = True
        return improved

    def gentic_operators(self, x, y, Pm, Pc, strategy_type="Fixed-Rate"):
        if strategy_type == "Self-Adaptive":
            avg_eta = (x.eta_m + y.eta_m) / 2.0
            new_eta = avg_eta + np.random.normal(0, 1)
            new_eta = max(20.0, min(100.0, new_eta))
            x.eta_m = new_eta
            y.eta_m = new_eta

        def poly_mut_deterministic(ind_in, rate):
            new_ind = Individual(self.m, self.p_name)
            new_ind.X = list(ind_in.X)
            new_ind.eta_m = ind_in.eta_m
            n_vars = len(new_ind.X)
            eta = new_ind.eta_m if strategy_type == "Self-Adaptive" else 20.0
            mut_expected = n_vars * rate
            base_mut = int(mut_expected)
            remainder = mut_expected - base_mut
            num_mutations = base_mut + 1 if random.random() < remainder else base_mut
            num_mutations = max(1, num_mutations)
            target_indices = random.sample(range(n_vars), min(num_mutations, n_vars))
            
            for i in target_indices:
                u = random.random()
                beta = (2 * u) ** (1 / (eta + 1)) - 1 if u <= 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                low, high = new_ind.bounds[i]
                val = new_ind.X[i] + beta * (high - low)
                new_ind.X[i] = max(low, min(high, val))
            
            new_ind.fitness(self.m)
            new_ind.constraints_g(self.m)
            new_ind.feasibility()
            return new_ind

        def sbx(p1, p2, rate):
            child = Individual(self.m, self.p_name)
            child.d_variables()
            child.eta_m = p1.eta_m
            if rate <= 1e-6:
                child.X = list(p1.X)
            else:
                for i in range(len(p1.X)):
                    if random.random() < rate and abs(p1.X[i] - p2.X[i]) > 1e-8:
                        u = random.random()
                        beta = (2 * u) ** (1 / 16) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / 16)
                        val = 0.5 * ((1 + beta) * p1.X[i] + (1 - beta) * p2.X[i])
                        low, high = child.bounds[i]
                        child.X[i] = max(low, min(high, val))
                    else:
                        child.X[i] = p1.X[i]
            child.fitness(self.m)
            child.constraints_g(self.m)
            child.feasibility()
            return child

        ind1 = poly_mut_deterministic(x, Pm)
        ind2 = poly_mut_deterministic(y, Pm)
        return sbx(ind1, ind2, Pc)

    
    def CMOEA_D_DMA(self, strategy_type="Fixed-Rate"):
        self.generateWeightVectors()
        self.generate_neighbors()
        self.makeInitialPopulation()
        self.init_Z()
        for ind in self.individuals:
            self.best_point_z(ind)

        controller = TriPhaseController(self.Pm, self.individuals[0].n)
        successful_pm_history = []

        for j in range(self.nbGen):
            success_count = 0
            for i in range(self.N):
                pa = self.individuals[i]
                if pa.feasible and len(self.A[i]) > 0:
                    pb = random.choice(self.A[i])
                else:
                    pb = self.individuals[random.choice(self.neighbours[i])]

                y = self.gentic_operators(pa, pb, self.Pm, self.Pc, strategy_type)
                self.best_point_z(y)
                
                if self.Update_Solution_Archive(y, i, self.Alph):
                    success_count += 1

            success_rate = success_count / self.N
            if success_count > 0: successful_pm_history.append(self.Pm)

            # --- APPLY SELECTED MUTATION STRATEGY ---
            if strategy_type == "AI-Adaptive":
                self.Pm = controller.update(success_rate)
            elif strategy_type == "Deterministic-Decay":
                start_pm, end_pm = 0.50, 0.05
                self.Pm = start_pm - (start_pm - end_pm) * (j / self.nbGen)
            elif strategy_type == "1/5th-Rule":
                if success_rate > 0.20:
                    self.Pm = min(0.50, self.Pm / 0.85)
                elif success_rate < 0.20:
                    self.Pm = max(0.01, self.Pm * 0.85)
            # If "Fixed-Rate" or "Self-Adaptive", self.Pm remains unchanged

        self.final_pm = np.mean(successful_pm_history) if successful_pm_history else self.Pm
        return self.individuals


# ==========================================
# MAIN EXECUTION (Safe 5-Way Loop with GLOBAL NORMALIZATION)
# ==========================================
if __name__ == '__main__':
    # --- Configuration ---
    PROBLEMS = ["LIR-CMOP2", "Welded Beam", "Pressure Vessel", "m-cdtlz", "BICOP1"]
    pop_size = 300
    neighbour_size = 10
    nb_generation = 1000
    Pc_experiment = 1.0
    runs_per_strategy = 30
    csv_filename = "final_5way_comparative_results.csv"
    alph = 3

    print(f"Starting 5-Way Comparative Study (Global Normalization)", flush=True)

    for p_name in PROBLEMS:
        print(f"\n" + "=" * 60, flush=True)
        print(f"PROBLEM: {p_name}", flush=True)

        if "m-cdtlz" in p_name:
            m, n_c, n_v = 3, 3, 30
        elif "Welded Beam" in p_name:
            m, n_c, n_v = 2, 8, 4
        elif "Pressure Vessel" in p_name:
            m, n_c, n_v = 2, 4, 4
        elif "LIR-CMOP" in p_name:
            m, n_c, n_v = 2, 2, 30
        elif "BICOP1" in p_name:
            m, n_c, n_v = 2, 1, 30
        else:
            m, n_c, n_v = 2, 0, 30

        ai_pm_initial = AI.predict_rate(n_v, m, nb_generation, n_c, pop_size)
        fixed_pm_val = 1.0 / n_v

       
        strategies = [
            ("AI-Adaptive", ai_pm_initial),
            ("Fixed-Rate", fixed_pm_val),
            ("Deterministic-Decay", 0.50),
            ("1/5th-Rule", fixed_pm_val),
            ("Self-Adaptive", fixed_pm_val)
        ]

        # ---------------------------------------------------------
        # PHASE 1: EXECUTION & DATA COLLECTION
        # ---------------------------------------------------------
        run_data_vault = {}
        all_feasible_f_vals_for_problem = []

        for strategy_name, pm_rate in strategies:
            print(f"\n Running Strategy: {strategy_name}", flush=True)
            run_data_vault[strategy_name] = []

            for run_i in range(runs_per_strategy):
                print(f"Run {run_i + 1}/{runs_per_strategy} executing...", end="\r", flush=True)

                pop = Population(pop_size, neighbour_size, m, alph, nb_generation, Pc_experiment, pm_rate, p_name)

                start_time = time.time()
                res = pop.CMOEA_D_DMA(strategy_type=strategy_name)
                exec_time = time.time() - start_time

                # Extract only feasible objective values
                f_vals = np.array([ind.f for ind in res if ind.feasible])

                if len(f_vals) > 0:
                    all_feasible_f_vals_for_problem.append(f_vals)

                # Save raw data for Phase 3
                run_data_vault[strategy_name].append({
                    'f_vals': f_vals,
                    'exec_time': exec_time,
                    'final_pm': pop.final_pm
                })

                # 🧹 MEMORY NUKE
                del pop
                del res
                gc.collect()

        # ---------------------------------------------------------
        # PHASE 2: CALCULATE GLOBAL BOUNDS
        # ---------------------------------------------------------
        global_min, global_max, denom = None, None, None

        if len(all_feasible_f_vals_for_problem) > 0:
            # Stack every single feasible point found by all strategies
            stacked_all_f = np.vstack(all_feasible_f_vals_for_problem)
            global_min = stacked_all_f.min(axis=0)
            global_max = stacked_all_f.max(axis=0)

            denom = global_max - global_min
            denom[denom == 0] = 1.0  # Prevent division by zero
            print(f"\n Global Bounds Found! Min: {np.round(global_min, 2)} | Max: {np.round(global_max, 2)}", flush=True)
        else:
            print(f"\n  WARNING: No feasible solutions found by ANY strategy for {p_name}.", flush=True)

        # ---------------------------------------------------------
        # PHASE 3: GLOBAL NORMALIZATION, HV & LOGGING
        # ---------------------------------------------------------
        problem_hv_data = {}

        for strategy_name, _ in strategies:
            hv_list = []
            success_count_list = []
            exec_time_list = []      
            final_pm_list = []       
            for run_data in run_data_vault[strategy_name]:
                f_vals = run_data['f_vals']
                exec_time_list.append(run_data['exec_time'])
                final_pm_list.append(run_data['final_pm'])
                success_count_list.append(len(f_vals))

                run_hv = 0.0
                # Only calculate HV if the run found feasible points AND we have global bounds
                if len(f_vals) > 0 and global_min is not None:
                    f_norm = (f_vals - global_min) / denom
                    try:
                        ref_point = np.array([1.1] * m)
                        indicator = HV(ref_point=ref_point)
                        run_hv = indicator.do(f_norm)
                    except:
                        run_hv = 0.0

                hv_list.append(run_hv)

            # --- Save CSV Results for Strategy ---
            mean_hv = np.mean(hv_list) if len(hv_list) > 0 else 0.0
            std_hv = np.std(hv_list) if len(hv_list) > 0 else 0.0
            avg_success = np.mean(success_count_list) if len(success_count_list) > 0 else 0.0
            
            print(f"      ✅ {strategy_name} Result: Mean HV = {mean_hv:.4f} | Std = {std_hv:.4e}", flush=True)
            problem_hv_data[strategy_name] = hv_list

            result_row = {
                'Problem': p_name,
                'Strategy': strategy_name,
                'Mean_HV': mean_hv,
                'Std_HV': std_hv,
                'Success_Solutions': avg_success
            }
            
            file_exists = os.path.isfile(csv_filename)
            try:
                with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=result_row.keys())
                    if not file_exists: w.writeheader()
                    w.writerow(result_row)
            except Exception as e:
                print(f"⚠️ Could not save to CSV: {e}")

            # --- ACTIVE LEARNING LOGGER (Only log the absolute BEST run) ---
            if len(hv_list) > 0:
                best_run_index = np.argmax(hv_list)
                best_hv = hv_list[best_run_index]
                best_exec_time = exec_time_list[best_run_index]
                best_final_pm = final_pm_list[best_run_index]

                # We save the best run from ALL strategies to enrich the dataset!
                if best_hv > 0:
                    active_learning_logger(
                        problem=p_name, 
                        problem_type="Engineering" if "Welded" in p_name or "Pressure" in p_name else "Math", 
                        algorithm="CMOEA/D-DMA", 
                        pop_size=pop_size, 
                        n_constr=n_c, 
                        n_obj=m, 
                        archive_size=pop_size,
                        mutation_type="Polynomial", 
                        crossover_rate=Pc_experiment, 
                        crossover_type="SBX",
                        n_gen=nb_generation, 
                        n_var=n_v, 
                        ref_point=[1.1] * m,
                        exec_time=best_exec_time, 
                        hypervolume=best_hv, 
                        target_mutation_rate=best_final_pm,
                        filename="CMOEA_DMA_Randomized_Dataset_1.csv" 

        # ---------------------------------------------------------
        # STATISTICAL COMPARISON (Mann-Whitney U)
        # ---------------------------------------------------------
        print("\n   --- STATISTICAL SIGNIFICANCE (Mann-Whitney U) ---", flush=True)
        ai_data = problem_hv_data.get("AI-Adaptive", [])
        baselines = ["Fixed-Rate", "Deterministic-Decay", "1/5th-Rule", "Self-Adaptive"]

        for baseline in baselines:
            baseline_data = problem_hv_data.get(baseline, [])

            if len(ai_data) == 0 or len(baseline_data) == 0:
                print(f"      ⚠️ SKIP vs {baseline:<20} | Error: Missing data arrays.", flush=True)
                continue

            ai_mean = np.mean(ai_data)
            base_mean = np.mean(baseline_data)

            if np.allclose(ai_data, baseline_data):
                p_value = 1.0
            else:
                try:
                    stat, p_value = mannwhitneyu(ai_data, baseline_data, alternative='two-sided', method='asymptotic')
                except Exception as e:
                    print(f"  STATS CRASH vs {baseline:<20} | Error: {e}", flush=True)
                    p_value = 1.0

            if p_value < 0.05:
                winner = "AI-Adaptive" if ai_mean > base_mean else baseline
                indicator = " WIN" if winner == "AI-Adaptive" else " LOSS"
                print(f"      {indicator} vs {baseline:<20} | p-value: {p_value:.4e}", flush=True)
            else:
                print(f"      ⚖️ TIE vs {baseline:<20} | p-value: {p_value:.4e}", flush=True)

    print(f"\n All 5-way experiments complete! Results saved successfully.", flush=True)
