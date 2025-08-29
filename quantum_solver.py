"""
Complete Quantum TSP/VRP Solver with Real Road Network Integration
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
import requests
import folium
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Quantum Computing Imports
try:
    from qiskit import QuantumCircuit
    from qiskit.providers.basic_provider import BasicProvider
    from qiskit.primitives import Sampler, Estimator
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as IBMSampler, Estimator as IBMEstimator, Batch
        IBM_RUNTIME_AVAILABLE = True
    except ImportError:
        IBM_RUNTIME_AVAILABLE = False
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import Sampler as AerSampler, Estimator as AerEstimator
        AER_AVAILABLE = True
    except ImportError:
        AER_AVAILABLE = False
    try:
        from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP
        QISKIT_ALGORITHMS_AVAILABLE = True
    except ImportError:
        QISKIT_ALGORITHMS_AVAILABLE = False
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("❌ Qiskit not available. Install with: pip install qiskit qiskit-ibm-runtime qiskit-aer qiskit-algorithms")

# Problem Definition Classes
@dataclass
class Node:
    """Represents a node in the VRP/TSP problem"""
    id: int
    x: float  # Longitude
    y: float  # Latitude
    demand: float = 0
    time_window: Optional[Tuple[float, float]] = None
    service_time: float = 0

@dataclass
class Vehicle:
    """Represents a vehicle in the VRP"""
    id: int
    capacity: float
    max_time: float = float('inf')
    start_depot: int = 0
    end_depot: int = 0
    type: str = 'car' # Added vehicle type
    speed: float = 1.0 # Added speed
    fuel_consumption: float = 1.0 # Added fuel consumption per unit distance

class Problem:
    """Base class for routing problems with real road network integration"""
    def __init__(self, nodes: List[Node], distance_matrix: np.ndarray = None):
        self.nodes = nodes
        self.n_nodes = len(nodes)
        if distance_matrix is None:
            self.distance_matrix = self._calculate_distance_matrix()
        else:
            self.distance_matrix = distance_matrix
        self.metrics = {
            "qubo_size": 0,
            "solution_quality": [],
            "runtime": [],
            "circuit_depth": [],
            "qubits_used": []
        }

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix using real road network distances via OSRM"""
        n = len(self.nodes)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = self._get_osrm_distance(self.nodes[i], self.nodes[j])
                    distances[i][j] = distance if distance is not None else np.inf
        return distances

    def _get_osrm_distance(self, start_node: Node, end_node: Node) -> Optional[float]:
        """Query OSRM API for real road distance between two nodes"""
        start_lon, start_lat = start_node.x, start_node.y
        end_lon, end_lat = end_node.x, end_node.y
        loc_param = f"{start_lon},{start_lat};{end_lon},{end_lat}"
        url = f"http://router.project-osrm.org/route/v1/driving/{loc_param}"
        params = {"overview": "false"}  # We only need distance
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data['code'] == 'Ok':
                distance = data['routes'][0]['distance'] / 1000.0  # Convert meters to kilometers
                return distance
            else:
                logger.warning(f"OSRM could not find route from node {start_node.id} to {end_node.id}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"OSRM API error: {e}")
            return None

class TSPProblem(Problem):
    """Traveling Salesman Problem"""
    def postprocess_quantum_output(self, solution):
        """Apply postprocessing to quantum output"""
        post = PostProcessor()
        if isinstance(solution, dict):
            return post.repair_tsp_solution_advanced(solution, self)
        return list(solution)

class VRPProblem(Problem):
    """Vehicle Routing Problem"""
    def __init__(self, nodes: List[Node], vehicles: List[Vehicle], distance_matrix: np.ndarray = None):
        super().__init__(nodes, distance_matrix)
        self.vehicles = vehicles
        self.n_vehicles = len(vehicles)
    def postprocess_quantum_output(self, solution):
        return list(solution)

class QUBOFormulator:
    """Converts routing problems to QUBO formulations"""
    def __init__(self, problem_obj: Optional[Problem] = None):
        self.problem_obj = problem_obj

    def vrp_qubo(self, problem: VRPProblem, penalty: float = None):
        n = problem.n_nodes
        K = problem.n_vehicles
        W = problem.distance_matrix
        if penalty is None:
            penalty = np.max(W) * n * K
        Q = {}
        def add_qubo_term(i, j, coeff):
            Q[(i, j)] = Q.get((i, j), 0) + coeff
        def var_index(i, j, k):
            return k * (n * n) + i * n + j

        for k in range(K):
            vehicle_cost_multiplier = problem.vehicles[k].fuel_consumption / problem.vehicles[k].speed
            for i in range(n):
                for j in range(n):
                    if i != j:
                        var = var_index(i, j, k)
                        add_qubo_term(var, var, W[i, j] * vehicle_cost_multiplier)

        for j in range(1, n):
            vars_for_node = [var_index(i, j, k) for i in range(n) for k in range(K) if i != j]
            self._add_quadratic_penalty(Q, vars_for_node, penalty)

        for k in range(K):
            for i in range(n):
                vars_incoming = [var_index(j, i, k) for j in range(n) if i != j]
                vars_outgoing = [var_index(i, j, k) for j in range(n) if i != j]
                for u in vars_incoming:
                    for v in vars_outgoing:
                        add_qubo_term(u, v, -2 * penalty)
                for u in vars_incoming:
                    add_qubo_term(u, u, penalty)
                for v in vars_outgoing:
                    add_qubo_term(v, v, penalty)

        for k in range(K):
            vehicle = problem.vehicles[k]
            for i in range(1, n):
                var = var_index(0, i, k)
                demand = problem.nodes[i].demand
                capacity_penalty = penalty * (demand > vehicle.capacity)
                add_qubo_term(var, var, capacity_penalty)

        if self.problem_obj:
            self.problem_obj.metrics['qubo_size'] = len(Q)
            logger.info(f"Heterogeneous VRP QUBO generated with {len(Q)} terms")
        return Q

    def simplified_tsp_qubo(self, problem: TSPProblem) -> Dict[Tuple[int, int], float]:
        n = problem.n_nodes
        penalty = np.max(problem.distance_matrix) * n * 3 if n > 1 else 1.0
        Q = {}
        var_map = {}
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    var_map[(i, j)] = k
                    k += 1
        for i in range(n):
            for j in range(n):
                if i != j:
                    var_ij = var_map[(i, j)]
                    Q[(var_ij, var_ij)] = Q.get((var_ij, var_ij), 0) + problem.distance_matrix[i][j]
        for i in range(n):
            outgoing_vars = [var_map[(i, j)] for j in range(n) if i != j]
            self._add_quadratic_penalty(Q, outgoing_vars, penalty)
        for j in range(n):
            incoming_vars = [var_map[(i, j)] for i in range(n) if i != j]
            self._add_quadratic_penalty(Q, incoming_vars, penalty)
        if self.problem_obj:
            self.problem_obj.metrics['qubo_size'] = len(Q)
            logger.info(f"Simplified TSP QUBO generated with {len(Q)} terms")
        return Q

    def _add_quadratic_penalty(self, Q: Dict, variables: List[int], penalty: float):
        for i in range(len(variables)):
            var_i = variables[i]
            Q[(var_i, var_i)] = Q.get((var_i, var_i), 0) - penalty
            for j in range(i + 1, len(variables)):
                var_j = variables[j]
                v1, v2 = (var_i, var_j) if var_i < var_j else (var_j, var_i)
                Q[(v1, v2)] = Q.get((v1, v2), 0) + 2 * penalty

class QiskitQAOASolver:
    def __init__(self, backend_name='basic_simulator', shots=1024, p_layers=2,
                 optimizer='COBYLA', use_real_device=False, use_aer=True, problem_obj: Problem = None):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not installed")
        self.shots = shots
        self.p_layers = p_layers
        self.use_real_device = use_real_device
        self.problem_obj = problem_obj
        if use_real_device and IBM_RUNTIME_AVAILABLE:
            try:
                service = QiskitRuntimeService()
                backend = service.least_busy(operational=True, simulator=False)
                self.sampler = IBMSampler(backend=backend)
                self.estimator = IBMEstimator(backend=backend)
                self.backend = backend
                logger.info(f"Using IBM Quantum device: {backend.name}")
            except Exception as e:
                logger.warning(f"Failed to connect to IBM Quantum: {e}")
                self._setup_local_backend(use_aer)
        else:
            self._setup_local_backend(use_aer)
        if QISKIT_ALGORITHMS_AVAILABLE:
            if optimizer == 'COBYLA':
                self.optimizer = COBYLA(maxiter=200)
            elif optimizer == 'SPSA':
                self.optimizer = SPSA(maxiter=100)
            else:
                self.optimizer = SLSQP(maxiter=100)
        else:
            self.optimizer = None

    def _setup_local_backend(self, use_aer):
        if use_aer and AER_AVAILABLE:
            self.backend = AerSimulator()
            self.sampler = AerSampler()
            self.estimator = AerEstimator()
            logger.info("Using Aer simulator")
        else:
            provider = BasicProvider()
            self.backend = provider.get_backend('basic_simulator')
            self.sampler = Sampler()
            self.estimator = Estimator()
            logger.info("Using basic Qiskit simulator")

    def qubo_to_ising_hamiltonian(self, Q: Dict[Tuple[int, int], float]) -> Tuple[SparsePauliOp, List[int]]:
        variables = set()
        for (i, j) in Q.keys():
            variables.add(i)
            variables.add(j)
        var_list = sorted(list(variables))
        n_vars = len(var_list)
        var_to_idx = {var: idx for idx, var in enumerate(var_list)}
        pauli_terms = []
        coeffs = []
        for (i, j), weight in Q.items():
            i_idx = var_to_idx[i]
            j_idx = var_to_idx[j]
            if i == j:
                z_pauli = ['I'] * n_vars
                z_pauli[i_idx] = 'Z'
                pauli_terms.append(''.join(z_pauli))
                coeffs.append(-weight / 2)
            else:
                zz_pauli = ['I'] * n_vars
                zz_pauli[i_idx] = 'Z'
                zz_pauli[j_idx] = 'Z'
                pauli_terms.append(''.join(zz_pauli))
                coeffs.append(weight / 4)
                z_i_pauli = ['I'] * n_vars
                z_i_pauli[i_idx] = 'Z'
                pauli_terms.append(''.join(z_i_pauli))
                coeffs.append(-weight / 4)
                z_j_pauli = ['I'] * n_vars
                z_j_pauli[j_idx] = 'Z'
                pauli_terms.append(''.join(z_j_pauli))
                coeffs.append(-weight / 4)
        if pauli_terms:
            hamiltonian = SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))
        else:
            hamiltonian = SparsePauliOp.from_list([('I' * max(1, n_vars), 0)])
        if self.problem_obj:
            self.problem_obj.metrics['qubo_size'] = n_vars
        return hamiltonian, var_list

    def create_qaoa_circuit(self, hamiltonian: SparsePauliOp, beta: List[float], gamma: List[float], add_measures: bool = True) -> QuantumCircuit:
        n_qubits = hamiltonian.num_qubits
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        p = len(beta)  # Dynamically determine p from input parameters
        for layer in range(p):
            for term, weight in zip(hamiltonian.paulis, hamiltonian.coeffs):
                pauli_str = str(term)
                if pauli_str.count('Z') == 1:
                    indices = [i for i, c in enumerate(reversed(pauli_str)) if c == 'Z']
                    if len(indices) == 1:
                        qc.rz(2 * gamma[layer] * weight.real, indices[0])
                elif pauli_str.count('Z') == 2:
                    indices = [i for i, c in enumerate(reversed(pauli_str)) if c == 'Z']
                    if len(indices) == 2:
                        qc.rzz(2 * gamma[layer] * weight.real, indices[0], indices[1])
            for i in range(n_qubits):
                qc.rx(2 * beta[layer], i)
        if add_measures:
            qc.measure_all()
        if self.use_real_device and IBM_RUNTIME_AVAILABLE:
            try:
                from qiskit.transpiler import PassManager
                from qiskit.transpiler.passes import VF2Layout
                from qiskit import transpile
                vf2_pass = VF2Layout(coupling_map=self.backend.configuration().coupling_map)
                pm = PassManager([vf2_pass])
                qc = transpile(qc, backend=self.backend, pass_manager=pm, optimization_level=1)
                logger.info("Applied VF2Layout for circuit")
            except Exception as e:
                logger.warning(f"VF2Layout failed: {e}, using default transpilation")
                qc = transpile(qc, backend=self.backend, optimization_level=1)
        if self.problem_obj:
            self.problem_obj.metrics['circuit_depth'].append(qc.depth())
            self.problem_obj.metrics['qubits_used'].append(n_qubits)
        return qc

    def compute_expectation(self, hamiltonian, beta, gamma):
        qc = self.create_qaoa_circuit(hamiltonian, beta, gamma, add_measures=False)
        job = self.estimator.run([qc], [hamiltonian])
        result = job.result()
        return result.values[0]

    def is_valid_tsp_solution(self, solution: Dict[int, int], var_list: List[int], n: int) -> bool:
        # Build var_map
        var_map = {}
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    var_map[(i, j)] = k
                    k += 1
        inv_var_map = {v: k for k, v in var_map.items()}
        # Edges
        edges = []
        outgoing = [0] * n
        incoming = [0] * n
        for var, val in solution.items():
            if val == 1:
                if var in inv_var_map:
                    (i, j) = inv_var_map[var]
                    edges.append((i, j))
                    outgoing[i] += 1
                    incoming[j] += 1
        if any(o != 1 for o in outgoing) or any(i != 1 for i in incoming):
            return False
        # Build graph
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        # Check connected
        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(graph[node])
        if len(visited) != n:
            return False
        # Since in/out degree 1 and connected, it's a single cycle
        return True

    def qaoa_solve(self, Q: Dict[Tuple[int, int], float], num_runs: int = 10) -> Dict:
        try:
            hamiltonian, var_list = self.qubo_to_ising_hamiltonian(Q)
            n_qubits = len(var_list)
            if n_qubits > 20:
                logger.warning(f"Problem size {n_qubits} may be too large for QAOA")
                return self._fallback_solution(var_list)
            best_solution = None
            best_energy = float('inf')
            quantum_results = []

            if self.optimizer is None or (self.use_real_device and IBM_RUNTIME_AVAILABLE):
                logger.warning("Using random parameter sampling instead of optimization" + (" due to real device" if self.use_real_device else ""))
                circuits = []
                betas = []
                gammas = []
                for run in range(num_runs):
                    logger.info(f"Preparing QAOA circuit for run {run + 1}/{num_runs}")
                    beta = [random.uniform(0, np.pi) for _ in range(self.p_layers)]
                    gamma = [random.uniform(0, np.pi/2) for _ in range(self.p_layers)]
                    qc = self.create_qaoa_circuit(hamiltonian, beta, gamma)
                    circuits.append(qc)
                    betas.append(beta)
                    gammas.append(gamma)
                    logger.info(f"Circuit {run + 1} prepared, qubits: {n_qubits}, depth: {qc.depth()}")
                if self.use_real_device and IBM_RUNTIME_AVAILABLE:
                    try:
                        logger.info(f"Submitting {num_runs} circuits in batch mode to {self.backend.name}")
                        with Batch(backend=self.backend) as batch:
                            jobs = [self.sampler.run([qc], shots=self.shots) for qc in circuits]
                    except Exception as e:
                        logger.error(f"Batch submission failed: {e}, falling back to individual runs")
                        jobs = [self.sampler.run([qc], shots=self.shots) for qc in circuits]
                else:
                    logger.info(f"Submitting {num_runs} circuits individually to {self.backend.name}")
                    jobs = [self.sampler.run([qc], shots=self.shots) for qc in circuits]
                for run, (job, beta, gamma) in enumerate(zip(jobs, betas, gammas)):
                    logger.info(f"Processing results for run {run + 1}/{num_runs}")
                    start_time = time.time()
                    try:
                        result = job.result()
                        logger.info(f"Job completed for run {run + 1}")
                        if hasattr(result, "quasi_dists"):
                            if isinstance(result.quasi_dists, list):
                                counts = result.quasi_dists[0]
                            else:
                                counts = result.quasi_dists.get(0, {})
                        else:
                            counts = {}
                            logger.warning(f"Unknown sampler result format for run {run + 1}, cannot get counts")
                        if not counts:
                            logger.warning(f"No counts obtained for run {run + 1}")
                            continue
                        for bitstring, count in counts.items():
                            if count > 0:
                                bitstring_str = format(int(bitstring), f"0{n_qubits}b") if isinstance(bitstring, (int, str)) else str(bitstring)
                                solution = {var_list[i]: int(bit) for i, bit in enumerate(reversed(bitstring_str))}
                                energy = self._calculate_qubo_energy(Q, solution)
                                logger.info(f"Run {run + 1}: Found solution with energy {energy}")
                                quantum_results.append({
                                    'solution': solution,
                                    'energy': energy,
                                    'beta': beta,
                                    'gamma': gamma
                                })
                                if energy < best_energy:
                                    best_energy = energy
                                    best_solution = solution
                        if self.problem_obj:
                            self.problem_obj.metrics['runtime'].append(time.time() - start_time)
                            self.problem_obj.metrics['solution_quality'].append(best_energy)
                    except Exception as e:
                        logger.error(f"Run {run + 1} failed: {e}")
                        continue
            else:
                import numpy as np
                for run in range(num_runs):
                    logger.info(f"QAOA optimization run {run + 1}/{num_runs}")
                    start_time = time.time()
                    opt_beta = []
                    opt_gamma = []
                    opt_energy_per_layer = []
                    for layer in range(1, self.p_layers + 1):
                        new_beta_init = random.uniform(0, np.pi)
                        new_gamma_init = random.uniform(0, 2 * np.pi)
                        initial_params = np.array([new_beta_init, new_gamma_init])
                        def cost_fn(params):
                            current_beta = opt_beta + [params[0]]
                            current_gamma = opt_gamma + [params[1]]
                            exp = self.compute_expectation(hamiltonian, current_beta, current_gamma)
                            logger.debug(f"Current expectation: {exp}")
                            return exp
                        try:
                            opt_result = self.optimizer.minimize(cost_fn, initial_params)
                            opt_beta.append(opt_result.x[0])
                            opt_gamma.append(opt_result.x[1])
                            opt_energy_per_layer.append(opt_result.fun)
                            logger.info(f"Layer {layer} optimization complete. Energy: {opt_result.fun}")
                        except Exception as e:
                            logger.error(f"Layer {layer} optimization failed: {e}")
                            break
                    if len(opt_beta) != self.p_layers:
                        continue
                    opt_energy = opt_energy_per_layer[-1]
                    logger.info(f"Layer-wise optimization complete. Final energy: {opt_energy}")
                    qc = self.create_qaoa_circuit(hamiltonian, opt_beta, opt_gamma)
                    job = self.sampler.run([qc], shots=self.shots)
                    result = job.result()
                    if hasattr(result, "quasi_dists"):
                        counts = result.quasi_dists[0] if isinstance(result.quasi_dists, list) else result.quasi_dists
                    else:
                        counts = {}
                    run_best_energy = float('inf')
                    run_best_solution = None
                    for bitstring, prob in counts.items():
                        if prob > 0.01:
                            bitstring_str = format(int(bitstring), f"0{n_qubits}b")
                            solution = {var_list[i]: int(bit) for i, bit in enumerate(reversed(bitstring_str))}
                            energy = self._calculate_qubo_energy(Q, solution)
                            if energy < run_best_energy:
                                run_best_energy = energy
                                run_best_solution = solution
                    if run_best_solution:
                        quantum_results.append({
                            'solution': run_best_solution,
                            'energy': run_best_energy,
                            'beta': opt_beta,
                            'gamma': opt_gamma,
                            'opt_energy': opt_energy
                        })
                        if run_best_energy < best_energy:
                            best_energy = run_best_energy
                            best_solution = run_best_solution
                    if self.problem_obj:
                        self.problem_obj.metrics['runtime'].append(time.time() - start_time)
                        self.problem_obj.metrics['solution_quality'].append(run_best_energy)

            # After collecting all quantum_results, filter valid TSP solutions if applicable
            if quantum_results:
                num_vars = n_qubits
                n = int((1 + math.sqrt(1 + 4 * num_vars)) / 2)
                if n * (n - 1) == num_vars:
                    # This is likely a TSP problem
                    valid_results = [r for r in quantum_results if self.is_valid_tsp_solution(r['solution'], var_list, n)]
                    if valid_results:
                        quantum_results = valid_results
                        logger.info(f"Selected {len(valid_results)} valid TSP solutions out of {len(quantum_results)}")
                best_result = min(quantum_results, key=lambda r: r['energy'])
                best_energy = best_result['energy']
                best_solution = best_result['solution']

            if best_solution is None:
                logger.warning("No valid solutions found, using fallback")
                return self._fallback_solution(var_list)
            if self.problem_obj:
                best_solution = self.problem_obj.postprocess_quantum_output(best_solution)
            logger.info(f"Best solution found with energy {best_energy}")
            return {
                'solution': best_solution,
                'energy': best_energy,
                'method': 'QAOA',
                'num_qubits': n_qubits,
                'quantum_results': quantum_results
            }
        except Exception as e:
            logger.error(f"QAOA failed: {e}")
            try:
                _, var_list = self.qubo_to_ising_hamiltonian(Q)
            except:
                var_list = []
            return self._fallback_solution(var_list)

    def _manual_qaoa_solve(self, Q: Dict[Tuple[int, int], float],
                           hamiltonian: SparsePauliOp, var_list: List[int],
                           num_runs: int) -> Dict:
        best_solution = None
        best_energy = float('inf')
        n_qubits = len(var_list)
        qc_template = QuantumCircuit(n_qubits)
        qc_template.h(range(n_qubits))
        for i in range(n_qubits):
            qc_template.rz(Parameter(f'gamma_{i}'), i)
        for i in range(n_qubits):
            qc_template.rx(Parameter(f'beta_{i}'), i)
        qc_template.measure_all()
        param_names = [f'gamma_{i}' for i in range(n_qubits)] + [f'beta_{i}' for i in range(n_qubits)]
        parameters = [Parameter(name) for name in param_names]
        for run in range(num_runs):
            logger.info(f"QAOA run {run + 1}/{num_runs}")
            param_binds = {param: random.uniform(0, np.pi if 'gamma' in param.name else np.pi/2)
                           for param in parameters}
            qc = qc_template.assign_parameters(param_binds)
            job = self.sampler.run([qc], shots=self.shots)
            result = job.result()
            if hasattr(result, "quasi_dists"):
                if isinstance(result.quasi_dists, list):
                    counts = result.quasi_dists[0]
                else:
                    counts = result.quasi_dists.get(0, {})
            elif hasattr(result, "get_counts"):
                counts = result.get_counts()[0]
            else:
                counts = {}
                logger.warning("Unknown sampler result format, cannot get counts.")
            for bitstring, count in counts.items():
                if count > 0:
                    bitstring_str = format(int(bitstring), f"0{n_qubits}b") if isinstance(bitstring, (int, str)) else str(bitstring)
                    solution = {}
                    for i, var in enumerate(var_list):
                        bit_idx = -(i + 1)
                        if len(bitstring_str) > abs(bit_idx):
                            solution[var] = int(bitstring_str[bit_idx])
                        else:
                            solution[var] = 0
                    energy = self._calculate_qubo_energy(Q, solution)
                    if energy < best_energy:
                        best_energy = energy
                        best_solution = solution
        return {
            'solution': best_solution or {var: 0 for var in var_list},
            'energy': best_energy,
            'method': 'manual_qaoa',
            'num_qubits': n_qubits
        }

    def _fallback_solution(self, variables: List[int]) -> Dict:
        solution = {var: random.choice([0, 1]) for var in variables}
        return {
            'solution': solution,
            'energy': 0,
            'method': 'classical_fallback',
            'num_qubits': len(variables)
        }

    def _calculate_qubo_energy(self, Q: Dict[Tuple[int, int], float], solution: Dict[int, int]) -> float:
        energy = 0
        for (i, j), weight in Q.items():
            xi = solution.get(i, 0)
            xj = solution.get(j, 0)
            energy += weight * xi * xj
        return energy

class ClassicalPreprocessor:
    def cluster_customers_advanced(self, problem: VRPProblem, method='capacity_aware') -> Dict[int, List[int]]:
        """Advanced clustering ensuring all vehicles are used if possible."""
        logger.debug(f"Starting clustering for {problem.n_nodes} nodes, {problem.n_vehicles} vehicles")
        
        # Validate input
        total_demand = sum(node.demand for node in problem.nodes[1:])
        total_capacity = sum(vehicle.capacity for vehicle in problem.vehicles)
        logger.debug(f"Total demand: {total_demand}, Total capacity: {total_capacity}")
        if total_demand > total_capacity:
            logger.warning(f"Total demand ({total_demand}) exceeds total capacity ({total_capacity}).")
        
        # Initialize clusters as a dictionary
        customers = list(range(1, problem.n_nodes))
        clusters = {v.id: [] for v in problem.vehicles}
        remaining_customers = list(customers)
        random.shuffle(remaining_customers)
        
        logger.debug(f"Initial clusters: {clusters}, type: {type(clusters)}")
        logger.debug(f"Customers to assign: {remaining_customers}, count: {len(remaining_customers)}")

        # Sort vehicles by capacity in descending order
        vehicles_sorted = sorted(problem.vehicles, key=lambda v: v.capacity, reverse=True)
        n_vehicles = len(vehicles_sorted)
        customers_per_vehicle = max(1, len(customers) // n_vehicles)
        logger.debug(f"Customers per vehicle: {customers_per_vehicle}")

        # Phase 1: Assign at least one customer to each vehicle if possible
        for vehicle in vehicles_sorted:
            current_load = 0
            customers_in_cluster = set()
            customers_to_consider = list(remaining_customers)
            random.shuffle(customers_to_consider)
            assigned = 0
            for customer_idx in customers_to_consider:
                customer = problem.nodes[customer_idx]
                if current_load + customer.demand <= vehicle.capacity and customer_idx in remaining_customers:
                    clusters[vehicle.id].append(customer_idx)
                    current_load += customer.demand
                    customers_in_cluster.add(customer_idx)
                    assigned += 1
                    if assigned >= customers_per_vehicle:
                        break
            remaining_customers = [c for c in remaining_customers if c not in customers_in_cluster]
            logger.debug(f"Phase 1, vehicle {vehicle.id} (capacity={vehicle.capacity}): cluster={clusters[vehicle.id]}, load={current_load}, remaining={len(remaining_customers)}")

        # Phase 2: Assign remaining customers to vehicles with available capacity
        for vehicle in vehicles_sorted:
            current_load = sum(problem.nodes[c].demand for c in clusters[vehicle.id])
            customers_in_cluster = set(clusters[vehicle.id])
            customers_to_consider = list(remaining_customers)
            random.shuffle(customers_to_consider)
            for customer_idx in customers_to_consider:
                customer = problem.nodes[customer_idx]
                if current_load + customer.demand <= vehicle.capacity and customer_idx in remaining_customers:
                    clusters[vehicle.id].append(customer_idx)
                    current_load += customer.demand
                    customers_in_cluster.add(customer_idx)
            remaining_customers = [c for c in remaining_customers if c not in customers_in_cluster]
            logger.debug(f"Phase 2, vehicle {vehicle.id} (capacity={vehicle.capacity}): cluster={clusters[vehicle.id]}, load={current_load}, remaining={len(remaining_customers)}")

        # Phase 3: Handle any remaining customers
        if remaining_customers:
            logger.warning(f"Could not assign {len(remaining_customers)} customers: {remaining_customers}")
            for customer_idx in remaining_customers:
                candidate_vehicles = [
                    v for v in vehicles_sorted
                    if sum(problem.nodes[c].demand for c in clusters[v.id]) + problem.nodes[customer_idx].demand <= v.capacity
                ]
                if candidate_vehicles:
                    vehicle = min(candidate_vehicles, key=lambda v: len(clusters[v.id]))
                    clusters[vehicle.id].append(customer_idx)
                    logger.debug(f"Phase 3: Assigned customer {customer_idx} to vehicle {vehicle.id}")
                else:
                    largest_vehicle = vehicles_sorted[0]
                    clusters[vehicle.id].append(customer_idx)
                    logger.warning(f"Phase 3: Assigned customer {customer_idx} to vehicle {largest_vehicle.id} despite capacity constraints")
            remaining_customers = [c for c in remaining_customers if c not in sum(clusters.values(), [])]
            logger.debug(f"After Phase 3, remaining customers: {remaining_customers}")

        # Filter non-empty clusters
        logger.debug(f"Clusters before filtering: {clusters}, type: {type(clusters)}")
        final_clusters = {v_id: c for v_id, c in clusters.items() if c}
        
        # Fallback if no clusters are formed
        if not final_clusters:
            logger.warning("No valid clusters formed. Returning dictionary with empty lists for all vehicles.")
            final_clusters = {v.id: [] for v in problem.vehicles}
        
        # Final type check
        if not isinstance(final_clusters, dict):
            logger.error(f"final_clusters is not a dict: type={type(final_clusters)}, value={final_clusters}")
            raise ValueError(f"Clustering returned invalid format; expected dict, got {type(final_clusters)}")
        
        logger.info(f"Final clusters: {final_clusters}, type: {type(final_clusters)}")
        return final_clusters     
class PostProcessor:
    """Enhanced post-processing for solution repair"""
    def repair_tsp_solution_advanced(self, solution: Dict[int, int], problem: TSPProblem) -> List[int]:
        n = problem.n_nodes
        tour_edges = []
        var_map = {}
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    var_map[(i, j)] = k
                    k += 1
        inv_var_map = {v: k for k, v in var_map.items()}

        for var_index, value in solution.items():
            if value == 1:
                if var_index in inv_var_map:
                    tour_edges.append(inv_var_map[var_index])
                else:
                    logger.warning(f"Skipping invalid var_index {var_index} from solution.")
        tour = []
        if tour_edges:
            start_node = tour_edges[0][0]
            tour.append(start_node)
            current_node = start_node
            remaining_edges = list(tour_edges)
            while remaining_edges:
                next_edge = None
                for i, (u, v) in enumerate(remaining_edges):
                    if u == current_node:
                        next_edge = (u, v)
                        break
                if next_edge:
                    tour.append(next_edge[1])
                    current_node = next_edge[1]
                    remaining_edges.remove(next_edge)
                else:
                    logger.warning("Cannot form a simple tour from edges, falling back to greedy.")
                    remaining_nodes = set(range(n)) - set(tour)
                    if remaining_nodes:
                        tour.extend(list(remaining_nodes))
                        break
                    else:
                        break
        if len(set(tour)) != n:
            logger.warning("Extracted tour is invalid (missing or duplicate nodes), falling back to greedy.")
            tour = list(range(n))
            random.shuffle(tour)
        tour = self._two_opt_improvement(tour, problem.distance_matrix)
        return tour

    def _two_opt_improvement(self, tour: List[int], distance_matrix: np.ndarray) -> List[int]:
        def tour_distance(t):
            if len(t) <= 1: return 0
            return sum(distance_matrix[t[i]][t[(i+1) % len(t)]] for i in range(len(t)))
        best_tour = tour[:]
        best_distance = tour_distance(best_tour)
        improved = True
        k = 0
        while improved and k < 1000:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    if j - i >= 1:
                        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                        new_distance = tour_distance(new_tour)
                        if new_distance < best_distance:
                            best_tour = new_tour[:]
                            best_distance = new_distance
                            tour = new_tour[:]
                            improved = True
                            break
                if improved:
                    break
            k += 1
        return best_tour

class QuantumHybridSolver:
    def __init__(self, quantum_backend='qiskit', **kwargs):
        self.formulator = QUBOFormulator(kwargs.get('problem_obj'))
        self.preprocessor = ClassicalPreprocessor()
        self.postprocessor = PostProcessor()
        self.quantum_backend = quantum_backend
        if quantum_backend == 'qiskit' and QISKIT_AVAILABLE:
            self.quantum_solver = QiskitQAOASolver(**kwargs)
        else:
            logger.warning(f"Quantum backend {quantum_backend} not available, using Qiskit simulator as fallback.")
            if QISKIT_AVAILABLE:
                self.quantum_solver = QiskitQAOASolver(use_real_device=False, use_aer=True, **kwargs)
                self.quantum_backend = 'qiskit_simulator'
            else:
                raise ValueError(f"Qiskit not available, cannot use any Qiskit-based solver.")
        logger.info(f"Initialized quantum solver: {self.quantum_backend}")

    def solve_tsp_quantum(self, problem: TSPProblem, num_runs: int = 5,
                          use_simplified_qubo: bool = True) -> Tuple[List[int], float, Dict]:
        logger.info(f"Solving TSP with {problem.n_nodes} nodes using {self.quantum_backend}")
        if use_simplified_qubo or problem.n_nodes <= 6:
            qubo = self.formulator.simplified_tsp_qubo(problem)
        else:
            qubo = self.formulator.tsp_to_qubo(problem)
        logger.info(f"QUBO size: {len(qubo)} terms")
        logger.info(f"Estimated number of qubits for TSP: {len(set(var for edge in qubo.keys() for var in [edge[0], edge[1]]))}")
        best_solution = None
        best_energy = float('inf')
        quantum_results = []
        num_qubits = 0
        for run in range(num_runs):
            logger.info(f"Quantum run {run + 1}/{num_runs}")
            try:
                result = self.quantum_solver.qaoa_solve(qubo)
                quantum_results.append(result)
                if 'num_qubits' in result:
                    num_qubits = result['num_qubits']
                if result['energy'] < best_energy:
                    best_energy = result['energy']
                    best_solution = result['solution']
            except Exception as e:
                logger.error(f"Quantum run {run + 1} failed: {e}")
                continue
        if best_solution is None:
            logger.warning("All quantum runs failed, using classical fallback")
            tour, tour_cost, results = self._classical_tsp_fallback(problem)
            results['num_qubits'] = 0
            return tour, tour_cost, results
        tour = self.postprocessor.repair_tsp_solution_advanced(best_solution, problem)
        tour_cost = self._calculate_tour_cost(tour, problem.distance_matrix)
        results = {
            'quantum_results': quantum_results,
            'best_quantum_energy': best_energy,
            'post_processed_cost': tour_cost,
            'method': self.quantum_backend,
            'num_successful_runs': len(quantum_results),
            'num_qubits': num_qubits
        }
        return tour, tour_cost, results

    def solve_vrp_quantum(self, problem: VRPProblem, num_runs: int = 3,
                      clustering_method: str = 'capacity_aware') -> Tuple[Dict[int, List[int]], float, Dict]:
        logger.info(f"Solving VRP with {problem.n_nodes} nodes, {problem.n_vehicles} vehicles")
        
        # Get clusters
        clusters_by_vehicle = self.preprocessor.cluster_customers_advanced(problem, clustering_method)
        logger.debug(f"clusters_by_vehicle: {clusters_by_vehicle}, type: {type(clusters_by_vehicle)}")
        if not isinstance(clusters_by_vehicle, dict):
            logger.error(f"clusters_by_vehicle is not a dict: type={type(clusters_by_vehicle)}, value={clusters_by_vehicle}")
            raise ValueError(f"Clustering returned invalid format; expected dict, got {type(clusters_by_vehicle)}")
        logger.info(f"Created {len(clusters_by_vehicle)} clusters: {clusters_by_vehicle}")
        
        total_cost = 0
        all_routes_by_vehicle = {}  # Initialize as dict
        all_results = []
        total_qubits_estimated = 0

        for vehicle_id, cluster in clusters_by_vehicle.items():
            if not cluster:
                logger.debug(f"Vehicle {vehicle_id} has empty cluster, skipping")
                all_routes_by_vehicle[vehicle_id] = [0]  # Assign depot-only route
                continue
            logger.info(f"Solving cluster for Vehicle {vehicle_id}: nodes {cluster}")
            cluster_node_indices = [0] + cluster
            cluster_nodes = [problem.nodes[idx] for idx in cluster_node_indices]
            cluster_distances = self._extract_subproblem_distances(problem, cluster_node_indices)
            sub_problem = TSPProblem(cluster_nodes, cluster_distances)

            tour, cost, sub_results = self.solve_tsp_quantum(
                sub_problem,
                max(1, num_runs // len(clusters_by_vehicle)),
                use_simplified_qubo=True
            )

            # Ensure tour is a list
            if not isinstance(tour, list):
                logger.error(f"TSP tour for vehicle {vehicle_id} is not a list: type={type(tour)}, value={tour}")
                tour = list(tour) if hasattr(tour, '__iter__') else [0]
            
            original_tour = [cluster_node_indices[node_idx] for node_idx in tour]
            all_routes_by_vehicle[vehicle_id] = original_tour
            total_cost += cost
            all_results.append(sub_results)
            total_qubits_estimated += sub_results.get('num_qubits', 0)
            logger.debug(f"Vehicle {vehicle_id} route: {original_tour}, cost: {cost}")

        # Validate output
        if not isinstance(all_routes_by_vehicle, dict):
            logger.error(f"all_routes_by_vehicle is not a dict: type={type(all_routes_by_vehicle)}, value={all_routes_by_vehicle}")
            raise ValueError(f"Routes output invalid; expected dict, got {type(all_routes_by_vehicle)}")
        
        vrp_results = {
            'clustering_method': clustering_method,
            'clusters': clusters_by_vehicle,
            'subproblem_results': all_results,
            'total_cost': total_cost,
            'num_routes': len(all_routes_by_vehicle),
            'total_estimated_qubits': total_qubits_estimated
        }
        
        logger.debug(f"VRP results: routes={all_routes_by_vehicle}, total_cost={total_cost}, vrp_results={vrp_results}")
        return all_routes_by_vehicle, total_cost, vrp_results

    def _extract_subproblem_distances(self, problem: Problem, node_indices: List[int]) -> np.ndarray:
        n = len(node_indices)
        sub_distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                orig_i = node_indices[i]
                orig_j = node_indices[j]
                sub_distances[i][j] = problem.distance_matrix[orig_i][orig_j]
        return sub_distances

    def _calculate_tour_cost(self, tour: List[int], distance_matrix: np.ndarray) -> float:
        if len(tour) <= 1:
            return 0
        cost = 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            if tour[i] < distance_matrix.shape[0] and tour[j] < distance_matrix.shape[1]:
                cost += distance_matrix[tour[i]][tour[j]]
            else:
                logger.warning(f"Invalid index in tour: {tour[i]} or {tour[j]}. Distance matrix shape: {distance_matrix.shape}")
        return cost

    def _classical_tsp_fallback(self, problem: TSPProblem) -> Tuple[List[int], float, Dict]:
        logger.info("Using classical nearest neighbor fallback")
        tour = [0]
        unvisited = set(range(1, problem.n_nodes))
        current = 0
        while unvisited:
            nearest = min(unvisited, key=lambda x: problem.distance_matrix[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        tour.append(0)
        cost = self._calculate_tour_cost(tour, problem.distance_matrix)
        if tour and tour[0] == tour[-1] and len(tour) > 1:
            tour = tour[:-1]
        return tour, cost, {'method': 'classical_nearest_neighbor'}

def create_random_vrp(n_nodes: int, n_vehicles: int, seed: int = None) -> VRPProblem:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    depot_lon, depot_lat = -73.9857, 40.7484
    nodes = [Node(0, depot_lon, depot_lat, demand=0)]
    for i in range(1, n_nodes):
        lon = depot_lon + random.uniform(-0.1, 0.1)
        lat = depot_lat + random.uniform(-0.1, 0.1)
        nodes.append(Node(i, lon, lat, demand=10))

    vehicles = [
        Vehicle(0, capacity=100, type='truck', speed=0.7, fuel_consumption=2.5, max_time=8.0),
        Vehicle(1, capacity=50, type='car', speed=1.0, fuel_consumption=1.0, max_time=8.0),
        Vehicle(2, capacity=25, type='car', speed=1.2, fuel_consumption=0.8, max_time=8.0),
        Vehicle(3, capacity=10, type='bike', speed=0.5, fuel_consumption=0.2, max_time=4.0),
        Vehicle(4, capacity=5, type='bike', speed=0.8, fuel_consumption=0.1, max_time=4.0)
    ]

    if n_vehicles > len(vehicles):
        logger.warning(f"Number of vehicles requested ({n_vehicles}) exceeds defined types. Reusing vehicles.")
        vehicles.extend([Vehicle(i, capacity=24, max_time=8.0) for i in range(len(vehicles), n_vehicles)])
    elif n_vehicles < len(vehicles):
        vehicles = vehicles[:n_vehicles]

    return VRPProblem(nodes, vehicles)

def create_test_problems():
    tsp_nodes = [
        Node(0, -73.9857, 40.7484),
        Node(1, -73.9680, 40.7850),
        Node(2, -73.9934, 40.7589),
        Node(3, -74.0134, 40.7041),
        Node(4, -73.9597, 40.7661)
    ]
    tsp_problem = TSPProblem(tsp_nodes)
    vrp_nodes = [
        Node(0, -73.9857, 40.7484, demand=0, service_time=0.5),
        Node(1, -73.9680, 40.7850, demand=10, service_time=0.5),
        Node(2, -73.9934, 40.7589, demand=10, service_time=0.5),
        Node(3, -74.0134, 40.7041, demand=10, service_time=0.5),
        Node(4, -73.9597, 40.7661, demand=10, service_time=0.5),
        Node(5, -73.9978, 40.7209, demand=10, service_time=0.5),
        Node(6, -73.9813, 40.7736, demand=10, service_time=0.5),
        Node(7, -74.0059, 40.7403, demand=10, service_time=0.5),
        Node(8, -73.9772, 40.7587, demand=10, service_time=0.5),
        Node(9, -74.0091, 40.7128, demand=10, service_time=0.5),
        Node(10, -73.9867, 40.7305, demand=10, service_time=0.5),
        Node(11, -73.9442, 40.7789, demand=10, service_time=0.5),
        Node(12, -73.9911, 40.7505, demand=10, service_time=0.5)
    ]

    vehicles = [
        Vehicle(0, capacity=60, type='truck', speed=0.7, fuel_consumption=2.5, max_time=8.0),
        Vehicle(1, capacity=30, type='truck', speed=1.0, fuel_consumption=1.0, max_time=8.0),
        Vehicle(2, capacity=30, type='car', speed=1.0, fuel_consumption=1.0, max_time=8.0),
        Vehicle(3, capacity=10, type='bike', speed=0.5, fuel_consumption=0.2, max_time=4.0),
        Vehicle(4, capacity=10, type='bike', speed=0.5, fuel_consumption=0.2, max_time=4.0)
    ]
    vrp_problem = VRPProblem(vrp_nodes, vehicles)
    return tsp_problem, vrp_problem

def get_route_geometry_from_osrm(start_node: Node, end_node: Node) -> List[tuple]:
    start_lon, start_lat = start_node.x, start_node.y
    end_lon, end_lat = end_node.x, end_node.y
    loc_param = f"{start_lon},{start_lat};{end_lon},{end_lat}"
    url = f"http://router.project-osrm.org/route/v1/driving/{loc_param}"
    params = {"overview": "full", "geometries": "geojson"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['code'] != 'Ok':
            logger.warning(f"OSRM could not find a route from node {start_node.id} to {end_node.id}")
            return None
        route_geometry = data['routes'][0]['geometry']['coordinates']
        return [(lat, lon) for lon, lat in route_geometry]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OSRM API: {e}")
        return None

def visualize_solution_with_folium(problem, routes_by_vehicle, title, vehicle_info_map):
    import folium
    depot_node = problem.nodes[0]
    map_center = [depot_node.y, depot_node.x]
    m = folium.Map(location=map_center, zoom_start=13, tiles="cartodbpositron")

    folium.Marker(
        location=[depot_node.y, depot_node.x],
        popup="<strong>Depot (Node 0)</strong>",
        tooltip="Depot",
        icon=folium.Icon(color='black', icon='industry', prefix='fa')
    ).add_to(m)
    for node in problem.nodes[1:]:
        folium.Marker(
            location=[node.y, node.x],
            popup=f"<strong>Customer (Node {node.id})</strong>",
            tooltip=f"Customer {node.id}",
            icon=folium.Icon(color='gray', icon='user', prefix='fa')
        ).add_to(m)

    for vehicle_id, route in routes_by_vehicle.items():
        if not route:
            continue

        vehicle_details = vehicle_info_map.get(vehicle_id)
        if not vehicle_details:
            logger.warning(f"No color/label map for vehicle {vehicle_id}, skipping visualization.")
            continue

        route_color = vehicle_details['color']
        route_label = vehicle_details['label']

        viz_route = route[:]
        if viz_route[0] != 0:
            viz_route.insert(0, 0)
        if viz_route[-1] != 0:
            viz_route.append(0)

        for j in range(len(viz_route) - 1):
            start_node_id = viz_route[j]
            end_node_id = viz_route[j + 1]
            start_node = problem.nodes[start_node_id]
            end_node = problem.nodes[end_node_id]
            path_geometry = get_route_geometry_from_osrm(start_node, end_node)

            if path_geometry:
                folium.PolyLine(
                    locations=path_geometry,
                    color=route_color,
                    weight=4,
                    opacity=0.8,
                    tooltip=f"{route_label}: Segment {start_node.id} → {end_node.id}"
                ).add_to(m)

    filename = f"{title.replace(' ', '_').lower()}.html"
    m.save(filename)
    print(f"\n✅ Visualization with color-coded routes saved to '{filename}'")


def check_quantum_connectivity():
    print("🔍 Checking IBM Quantum connectivity...")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        print("✓ Successfully connected to IBM Quantum!")
        print("\nAvailable backends:")
        backends = service.backends()
        quantum_backends = []
        simulator_backends = []
        for backend in backends:
            if backend.simulator:
                simulator_backends.append(backend)
            else:
                quantum_backends.append(backend)
            try:
                status = "🟢" if backend.status().operational else "🔴"
                queue = backend.status().pending_jobs if hasattr(backend.status(), 'pending_jobs') else 0
                qubits = backend.num_qubits if hasattr(backend, 'num_qubits') else 'N/A'
                print(f"  {status} {backend.name} | Qubits: {qubits} | Queue: {queue}")
            except Exception as e:
                logger.warning(f"Could not get status for backend {backend.name}: {e}")
                print(f"  ❓ {backend.name} | Qubits: N/A | Queue: N/A (Status unavailable)")
        print(f"\n📊 Summary:")
        print(f"  Quantum devices: {len(quantum_backends)}")
        print(f"  Simulators: {len(simulator_backends)}")
        if quantum_backends:
            try:
                operational_backends = [b for b in quantum_backends if b.status().operational and hasattr(b, 'num_qubits') and b.num_qubits > 0]
                if operational_backends:
                    recommended = min(operational_backends, key=lambda b: b.status().pending_jobs if hasattr(b.status(), 'pending_jobs') else float('inf'))
                    print(f"  Recommended for small problems: {recommended.name} ({recommended.num_qubits} qubits)")
                else:
                    print("  No operational quantum devices available.")
            except Exception as e:
                logger.warning(f"Could not determine recommended backend: {e}")
                print("  Could not determine recommended quantum device.")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to IBM Quantum: {e}")
        print("    Please ensure your IBM Quantum Experience account is set up and saved.")
        print("    You can save your account using QiskitRuntimeService.save_account(...)")
        return False

def find_vrp_qubit_count(n_nodes: int, n_vehicles: int):
    print(f"\nCalculating estimated qubit count for VRP with {n_nodes} nodes and {n_vehicles} vehicles...")
    vrp_problem = create_random_vrp(n_nodes, n_vehicles, seed=123)
    preprocessor = ClassicalPreprocessor()
    clusters = preprocessor.cluster_customers_advanced(vrp_problem)
    max_qubits_needed = 0
    if clusters:
        largest_cluster_size = max(len(cluster) for cluster in clusters.values())
        logger.info(f"Largest cluster size (excluding depot): {largest_cluster_size}")
        nodes_in_largest_subproblem = largest_cluster_size + 1
        if nodes_in_largest_subproblem > 1:
            estimated_qubits = nodes_in_largest_subproblem * (nodes_in_largest_subproblem - 1)
            max_qubits_needed = estimated_qubits
            print(f"Estimated qubits needed for the largest VRP subproblem (simplified QUBO): {estimated_qubits}")
        else:
            print("Largest subproblem is too small (1 node including depot), 0 qubits needed for optimization.")
    else:
        logger.warning("No clusters formed for the given VRP problem.")
        print("Could not estimate qubits as no clusters were formed.")
    return max_qubits_needed

# New function to calculate route details
def calculate_route_details(problem: VRPProblem, vehicle: Vehicle, route: List[int]) -> Dict:
    DISTANCE_TO_TIME_FACTOR = 0.1
    total_distance = 0
    total_travel_time = 0
    total_service_time = 0
    total_fuel_cost = 0

    if len(route) > 1:
        full_route = route[:]
        if full_route[0] != 0: full_route.insert(0, 0)
        if full_route[-1] != 0: full_route.append(0)

        for i in range(len(full_route) - 1):
            start_node_id = full_route[i]
            end_node_id = full_route[i+1]
            distance = problem.distance_matrix[start_node_id][end_node_id]
            total_distance += distance
            total_travel_time += distance * (DISTANCE_TO_TIME_FACTOR / vehicle.speed)

    total_service_time = sum(problem.nodes[node_id].service_time for node_id in route if node_id != 0)
    total_route_duration = total_travel_time + total_service_time
    total_fuel_cost = total_distance * vehicle.fuel_consumption

    return {
        'total_distance': total_distance,
        'total_travel_time': total_travel_time,
        'total_service_time': total_service_time,
        'total_duration': total_route_duration,
        'total_fuel_cost': total_fuel_cost,
        'route': route
    }

# New function to print the summary
def print_solution_schedules(vrp_problem: VRPProblem, routes: Dict[int, List[int]]) -> str:
    if not isinstance(routes, dict):
        logger.error(f"Routes is not a dict in print_solution_schedules: type={type(routes)}, value={routes}")
        raise ValueError(f"Routes must be a dictionary, got {type(routes)}")
    
    schedule = []
    for vehicle_id, route in routes.items():
        vehicle = next(v for v in vrp_problem.vehicles if v.id == vehicle_id)
        route_str = f"Vehicle {vehicle_id} ({vehicle.type}): {' -> '.join(str(node_id) for node_id in route)}"
        schedule.append(route_str)
    return "\n".join(schedule)

def print_detailed_vehicle_summary(vrp_problem: VRPProblem, routes: Dict[int, List[int]], total_cost: float) -> str:
    if not isinstance(routes, dict):
        logger.error(f"Routes is not a dict in print_detailed_vehicle_summary: type={type(routes)}, value={routes}")
        raise ValueError(f"Routes must be a dictionary, got {type(routes)}")
    
    summary = [f"Total Cost: {total_cost:.2f}"]
    for vehicle_id, route in routes.items():
        vehicle = next(v for v in vrp_problem.vehicles if v.id == vehicle_id)
        total_demand = sum(vrp_problem.nodes[node_id].demand for node_id in route if node_id != 0)
        summary.append(f"Vehicle {vehicle_id} ({vehicle.type}):")
        summary.append(f"  Route: {' -> '.join(str(node_id) for node_id in route)}")
        summary.append(f"  Total Demand: {total_demand}/{vehicle.capacity}")
    return "\n".join(summary)
def run_ibm_quantum_experiments():
    print("=" * 80)
    print("RUNNING ON IBM QUANTUM HARDWARE")
    print("=" * 80)
    tsp_problem, vrp_problem = create_test_problems()
    print(f"TSP Problem: {tsp_problem.n_nodes} nodes")
    print(f"VRP Problem: {vrp_problem.n_nodes} nodes, {vrp_problem.n_vehicles} vehicles")

    vehicle_color_map = {
        0: {'color': 'red', 'label': 'Truck 1'},
        1: {'color': 'blue', 'label': 'Car 1'},
        2: {'color': 'green', 'label': 'Car 2'},
        3: {'color': 'orange', 'label': 'Bike 1'},
        4: {'color': 'purple', 'label': 'Bike 2'}
    }

    test_configs = [
        {
            'name': 'IBM Quantum (Real Device)',
            'use_real_device': True,
            'shots': 1024,
            'p_layers': 1,
            'optimizer': 'COBYLA'
        },
       
    ]
    results = {}
    for config in test_configs:
        print(f"\n{'-' * 60}")
        print(f"TESTING: {config['name']}")
        print(f"{'-' * 60}")
        try:
            solver = QuantumHybridSolver(
                quantum_backend='qiskit',
                use_real_device=config['use_real_device'],
                use_aer=config.get('use_aer', True),
                shots=config['shots'],
                p_layers=config['p_layers'],
                optimizer=config['optimizer'],
                problem_obj=tsp_problem if config['name'] == 'IBM Quantum (Real Device)' else None
            )
            initialization_successful = True
        except Exception as init_e:
            logger.error(f"Solver initialization failed for {config['name']}: {init_e}")
            results[config['name']] = {'initialization_error': str(init_e)}
            initialization_successful = False
        if initialization_successful:
            config_results = {'tsp': None, 'vrp': None}

            print(f"\n🔬 Solving TSP ({tsp_problem.n_nodes} nodes)...")
            start_time = time.time()
            try:
                tsp_tour, tsp_cost, tsp_details = solver.solve_tsp_quantum(
                    tsp_problem,
                    num_runs=3,
                    use_simplified_qubo=True
                )
                tsp_time = time.time() - start_time
                config_results['tsp'] = {
                    'tour': tsp_tour,
                    'cost': tsp_cost,
                    'time': tsp_time,
                    'details': tsp_details,
                    'success': True,
                    'num_qubits': tsp_details.get('num_qubits', 'N/A')
                }
                print(f"✓ TSP Solution: Cost = {tsp_cost:.2f} km, Time = {tsp_time:.2f}s, Qubits = {config_results['tsp']['num_qubits']}")
                print(f"  Tour: {tsp_tour}")
                print(f"  Method: {tsp_details.get('method', 'unknown')}")
            except Exception as e:
                print(f"❌ TSP failed: {e}")
                config_results['tsp'] = {'success': False, 'error': str(e)}

            print(f"\n🚚 Solving VRP ({vrp_problem.n_nodes} nodes, {vrp_problem.n_vehicles} vehicles)...")
            start_time = time.time()
            try:
                vrp_routes, vrp_cost, vrp_details = solver.solve_vrp_quantum(
                    vrp_problem,
                    num_runs=2,
                    clustering_method='capacity_aware'
                )
                vrp_time = time.time() - start_time
                config_results['vrp'] = {
                    'routes': vrp_routes,
                    'cost': vrp_cost,
                    'time': vrp_time,
                    'details': vrp_details,
                    'success': True,
                    'total_estimated_qubits': vrp_details.get('total_estimated_qubits', 'N/A')
                }
                print(f"✓ VRP Solution: Cost = {vrp_cost:.2f} km, Time = {vrp_time:.2f}s, Est. Qubits = {config_results['vrp']['total_estimated_qubits']}")
                print(f"  Routes: {vrp_routes}")
                print(f"  Clusters: {vrp_details.get('clusters', [])}")

                if vrp_routes:
                    print_solution_schedules(vrp_problem, vrp_routes)
                    print_detailed_vehicle_summary(vrp_problem, vrp_routes)

                visualize_solution_with_folium(vrp_problem, vrp_routes, f"VRP - {config['name']}", vehicle_color_map)

            except Exception as e:
                print(f"❌ VRP failed: {e}")
                config_results['vrp'] = {'success': False, 'error': str(e)}
            results[config['name']] = config_results


    print(f"\n{'=' * 80}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'Configuration':<25} | {'TSP Cost':<10} | {'TSP Time':<10} | {'TSP Qubits':<12} | {'VRP Cost':<10} | {'VRP Time':<10} | {'VRP Est. Qubits':<15}")
    print("-" * 120)
    for config_name, config_results in results.items():
        if 'initialization_error' in config_results:
            print(f"{config_name:<25} | INITIALIZATION ERROR: {config_results['initialization_error']}")
            continue
        tsp_cost = "FAILED"
        tsp_time = "N/A"
        tsp_qubits = "N/A"
        vrp_cost = "FAILED"
        vrp_time = "N/A"
        vrp_est_qubits = "N/A"
        if config_results.get('tsp', {}).get('success'):
            tsp_cost = f"{config_results['tsp']['cost']:.2f}"
            tsp_time = f"{config_results['tsp']['time']:.2f}s"
            tsp_qubits = config_results['tsp']['num_qubits']
        if config_results.get('vrp', {}).get('success'):
            vrp_cost = f"{config_results['vrp']['cost']:.2f}"
            vrp_time = f"{config_results['vrp']['time']:.2f}s"
            vrp_est_qubits = config_results['vrp']['total_estimated_qubits']
        print(f"{config_name:<25} | {tsp_cost:<10} | {tsp_time:<10} | {str(tsp_qubits):<12} | {vrp_cost:<10} | {vrp_time:<10} | {str(vrp_est_qubits):<15}")
    print(f"\n{'=' * 60}")
    print("QUANTUM ADVANTAGE ANALYSIS")
    print(f"{'=' * 60}")
    print("\n🔄 Running classical benchmarks...")
    try:
        classical_solver = ClassicalSolver()
        classical_tsp_tour, classical_tsp_cost = classical_solver.solve_tsp_2opt(tsp_problem)
        print(f"Classical TSP ({tsp_problem.n_nodes} nodes): Cost = {classical_tsp_cost:.2f} km")
        quantum_tsp_costs = []
        for config_name, config_results in results.items():
            if config_results.get('tsp', {}).get('success'):
                quantum_tsp_costs.append((config_name, config_results['tsp']['cost']))
        if quantum_tsp_costs:
            print("Quantum TSP Results:")
            for name, cost in quantum_tsp_costs:
                if classical_tsp_cost != 0:
                    improvement = ((classical_tsp_cost - cost) / classical_tsp_cost) * 100
                    print(f"  {name}: {cost:.2f} km ({improvement:+.1f}% vs classical)")
                else:
                    print(f"  {name}: {cost:.2f} km (Classical cost is zero, cannot calculate improvement)")
    except Exception as e:
        print(f"Classical benchmark failed: {e}")
    print(f"\n🎯 Key Insights:")
    print("1. Real quantum devices may have higher error rates due to noise")
    print("2. Simulators provide exact results but don't show quantum advantage")
    print("3. Problem size is limited by current quantum hardware capabilities (check qubit counts)")
    print("4. Hybrid approach combines quantum optimization with classical post-processing and preprocessing")
    return results

class ClassicalSolver:
    def solve_tsp_2opt(self, problem: TSPProblem) -> Tuple[List[int], float]:
        tour = [0]
        unvisited = set(range(1, problem.n_nodes))
        current = 0
        while unvisited:
            nearest = min(unvisited, key=lambda x: problem.distance_matrix[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        def tour_distance(t):
            if len(t) <= 1: return 0
            return sum(problem.distance_matrix[t[i]][t[(i+1) % len(t)]] for i in range(len(t)))
        best_tour = tour[:]
        best_distance = tour_distance(best_tour)
        improved = True
        k = 0
        while improved and k < 1000:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    if j - i >= 1:
                        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                        new_distance = tour_distance(new_tour)
                        if new_distance < best_distance:
                            best_tour = new_tour[:]
                            best_distance = new_distance
                            tour = new_tour[:]
                            improved = True
                            break
                if improved:
                    break
            k += 1
        return best_tour, best_distance

def find_vrp_qubit_requirement(n_nodes: int, n_vehicles: int):
    print(f"\nEstimating qubit count for VRP with {n_nodes} nodes and {n_vehicles} vehicles...")
    vrp_problem = create_random_vrp(n_nodes, n_vehicles, seed=456)
    preprocessor = ClassicalPreprocessor()
    clusters = preprocessor.cluster_customers_advanced(vrp_problem)
    max_qubits_needed = 0
    if clusters:
        largest_cluster_size = max(len(cluster) for cluster in clusters.values())
        logger.info(f"Largest cluster size (excluding depot): {largest_cluster_size}")
        nodes_in_largest_subproblem = largest_cluster_size + 1
        if nodes_in_largest_subproblem > 1:
            estimated_qubits = nodes_in_largest_subproblem * (nodes_in_largest_subproblem - 1)
            max_qubits_needed = estimated_qubits
            print(f"Estimated qubits needed for the largest VRP subproblem (simplified QUBO): {estimated_qubits}")
        else:
            print("Largest subproblem is too small (1 node including depot), 0 qubits needed for optimization.")
    else:
        logger.warning("No clusters formed for the VRP problem.")
        print("Could not estimate qubits as no clusters were formed.")
    return max_qubits_needed

if __name__ == "__main__":
    print("🚀 IBM Quantum TSP/VRP Solver with Real Road Network")
    print("=" * 50)
    if not QISKIT_AVAILABLE:
        print("❌ Qiskit not available. Please install:")
        print("pip install qiskit qiskit-ibm-runtime qiskit-aer qiskit-algorithms")
    if QISKIT_AVAILABLE and IBM_RUNTIME_AVAILABLE:
        check_quantum_connectivity()
    else:
        print("\nSkipping IBM Quantum connectivity check (Qiskit or Runtime not available).")
    print(f"\n🎯 Starting quantum experiments...")
    try:
        QiskitRuntimeService.delete_account()
        QiskitRuntimeService.save_account(
            channel="ibm_cloud",
            token="S24GFhGFFjQ54O839cRmwC7bvMY4SNObsh-DaU22uL6g",
            instance="Path_optimization",
            overwrite=True
        )
        service = QiskitRuntimeService()
        
        results = run_ibm_quantum_experiments()
        print(f"\n🎉 Experiments completed!")
        vrp_qubit_count_13_5 = find_vrp_qubit_requirement(13, 5)
        print(f"Result: Estimated qubits for VRP (13 nodes, 5 vehicles): {vrp_qubit_count_13_5}")
        print(f"\n🔬 Quantum TSP/VRP Solver - Experiment Complete!")
    except KeyboardInterrupt:
        print(f"\n⏹ Experiments interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred during experiments: {e}")
        import traceback
        traceback.print_exc()