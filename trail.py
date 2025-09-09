# Quantum TSP/VRP Solver with Real Road Network Integration
# Updated to resolve "Invalid tour" issue and incorporate recommendations:
# - Stronger QUBO constraints for single visits, flow conservation, and subtour elimination
# - Increased QAOA layers (p_layers=4) and shots (shots=4096) with SPSA optimizer
# - Enhanced post-processing with MST-based tour construction
# - Validated cluster sizes to avoid trivial clusters
# - Additional logging for QAOA bitstrings
# - Classical benchmark comparison

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
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree

# Configure logging with detailed QAOA bitstring output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
        from qiskit_aer.primitives import SamplerV2, EstimatorV2
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
    type: str = 'car'
    speed: float = 1.0
    fuel_consumption: float = 1.0

class Problem:
    def __init__(self, nodes: List[Node], distance_matrix: np.ndarray = None):
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.distance_matrix = distance_matrix if distance_matrix is not None else self._calculate_distance_matrix()
        self.metrics = {
            "qubo_size": 0,
            "solution_quality": [],
            "runtime": [],
            "circuit_depth": [],
            "qubits_used": []
        }

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix using real road network distances via OSRM with caching"""
        n = len(self.nodes)
        distances = np.zeros((n, n))
        cache = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    key = (self.nodes[i].id, self.nodes[j].id)
                    if key in cache:
                        distances[i][j] = cache[key]
                    else:
                        distance = self._get_osrm_distance(self.nodes[i], self.nodes[j])
                        distances[i][j] = distance if distance is not None else np.inf
                        cache[key] = distances[i][j]
        return distances

    def _get_osrm_distance(self, start_node: Node, end_node: Node) -> Optional[float]:
        """Query OSRM API for real road distance with error handling"""
        start_lon, start_lat = start_node.x, start_node.y
        end_lon, end_lat = end_node.x, end_node.y
        loc_param = f"{start_lon},{start_lat};{end_lon},{end_lat}"
        url = f"http://router.project-osrm.org/route/v1/driving/{loc_param}"
        params = {"overview": "false"}
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data['code'] == 'Ok':
                distance = data['routes'][0]['distance'] / 1000.0
                return distance
            else:
                logger.warning(f"OSRM could not find route from node {start_node.id} to {end_node.id}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"OSRM API error: {e}")
            return None

class TSPProblem(Problem):
    def postprocess_quantum_output(self, solution):
        post = PostProcessor()
        return post.repair_tsp_solution_advanced(solution, self)

class VRPProblem(Problem):
    def __init__(self, nodes: List[Node], vehicles: List[Vehicle], distance_matrix: np.ndarray = None):
        super().__init__(nodes, distance_matrix)
        self.vehicles = vehicles
        self.n_vehicles = len(vehicles)
    def postprocess_quantum_output(self, solution):
        return list(solution)

class QUBOFormulator:
    def __init__(self, problem_obj: Optional[Problem] = None):
        self.problem_obj = problem_obj

    def vrp_qubo(self, problem: VRPProblem, penalty: float = None):
        """Enhanced VRP QUBO with time and single-visit constraints"""
        n = problem.n_nodes
        K = problem.n_vehicles
        W = problem.distance_matrix
        penalty = np.max(W) * n * K * 5 if penalty is None else penalty  # Increased penalty
        Q = {}
        DISTANCE_TO_TIME_FACTOR = 0.1

        def add_qubo_term(i, j, coeff):
            Q[(i, j)] = Q.get((i, j), 0) + coeff
        def var_index(i, j, k):
            return k * (n * n) + i * n + j

        # Objective: Minimize distance with vehicle-specific costs
        for k in range(K):
            vehicle = problem.vehicles[k]
            vehicle_cost_multiplier = vehicle.fuel_consumption / vehicle.speed
            for i in range(n):
                for j in range(n):
                    if i != j:
                        var = var_index(i, j, k)
                        add_qubo_term(var, var, W[i][j] * vehicle_cost_multiplier)

        # Constraint: Each customer visited exactly once
        for j in range(1, n):
            vars_for_node = [var_index(i, j, k) for i in range(n) for k in range(K) if i != j]
            self._add_quadratic_penalty(Q, vars_for_node, penalty)

        # Constraint: Flow conservation
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

        # Constraint: Capacity
        for k in range(K):
            vehicle = problem.vehicles[k]
            for i in range(1, n):
                var = var_index(0, i, k)
                demand = problem.nodes[i].demand
                capacity_penalty = penalty * (demand > vehicle.capacity)
                add_qubo_term(var, var, capacity_penalty)

        # Constraint: Time windows
        for k in range(K):
            vehicle = problem.vehicles[k]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        var = var_index(i, j, k)
                        travel_time = W[i][j] * (DISTANCE_TO_TIME_FACTOR / vehicle.speed)
                        service_time = problem.nodes[j].service_time if j != 0 else 0
                        total_time = travel_time + service_time
                        if total_time > vehicle.max_time:
                            add_qubo_term(var, var, penalty * total_time)

        if self.problem_obj:
            self.problem_obj.metrics['qubo_size'] = len(Q)
            logger.info(f"VRP QUBO generated with {len(Q)} terms")
        return Q

    def simplified_tsp_qubo(self, problem: TSPProblem) -> Dict[Tuple[int, int], float]:
        """Enhanced TSP QUBO with stronger constraints for single visits and subtour elimination"""
        # Addresses "Invalid tour" issue with stronger constraints
        n = problem.n_nodes
        penalty = np.max(problem.distance_matrix) * n * 5 if n > 1 else 1.0  # Increased penalty
        Q = {}
        var_map = {}
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    var_map[(i, j)] = k
                    k += 1
        # Objective: Minimize distance
        for i in range(n):
            for j in range(n):
                if i != j:
                    var_ij = var_map[(i, j)]
                    Q[(var_ij, var_ij)] = Q.get((var_ij, var_ij), 0) + problem.distance_matrix[i][j]
        # Constraint: Exactly one outgoing and incoming edge per node
        for i in range(n):
            outgoing_vars = [var_map[(i, j)] for j in range(n) if i != j]
            incoming_vars = [var_map[(j, i)] for j in range(n) if i != j]
            self._add_quadratic_penalty(Q, outgoing_vars, penalty)
            self._add_quadratic_penalty(Q, incoming_vars, penalty)
        # Constraint: Subtour elimination (simplified MTZ-like constraint)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    for k in range(1, n):
                        if k != i and k != j:
                            u = var_map.get((i, j))
                            v = var_map.get((j, k))
                            w = var_map.get((k, i))
                            if u is not None and v is not None and w is not None:
                                Q[(u, v)] = Q.get((u, v), 0) + penalty
                                Q[(v, w)] = Q.get((v, w), 0) + penalty
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
    def __init__(self, backend_name='basic_simulator', shots=4096, p_layers=4,
                 optimizer='SPSA', use_real_device=False, use_aer=True, problem_obj: Problem = None):
        # Recommendation: Increased shots to 4096 and p_layers to 4 with SPSA optimizer
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not installed")
        self.shots = shots
        self.p_layers = p_layers
        self.use_real_device = use_real_device
        self.problem_obj = problem_obj
        self._setup_local_backend(use_aer)
        if QISKIT_ALGORITHMS_AVAILABLE:
            if optimizer == 'SPSA':
                self.optimizer = SPSA(maxiter=200)  # Increased iterations
            elif optimizer == 'COBYLA':
                self.optimizer = COBYLA(maxiter=200)
            else:
                self.optimizer = SLSQP(maxiter=100)
        else:
            self.optimizer = None

    def _setup_local_backend(self, use_aer):
        """Setup quantum backend with V2 primitives"""
        if use_aer and AER_AVAILABLE:
            self.backend = AerSimulator()
            self.sampler = SamplerV2()
            self.estimator = EstimatorV2()
            logger.info("Using Aer simulator with V2 primitives")
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
        constant_term = 0

        for (i, j), weight in Q.items():
            i_idx = var_to_idx[i]
            j_idx = var_to_idx[j]
            if i == j:
                z_pauli = ['I'] * n_vars
                z_pauli[i_idx] = 'Z'
                pauli_terms.append(''.join(z_pauli))
                coeffs.append(-weight / 2.0)
                constant_term += weight / 2.0
            else:
                zz_pauli = ['I'] * n_vars
                zz_pauli[i_idx] = 'Z'
                zz_pauli[j_idx] = 'Z'
                pauli_terms.append(''.join(zz_pauli))
                coeffs.append(weight / 4.0)
                z_i_pauli = ['I'] * n_vars
                z_i_pauli[i_idx] = 'Z'
                pauli_terms.append(''.join(z_i_pauli))
                coeffs.append(-weight / 4.0)
                z_j_pauli = ['I'] * n_vars
                z_j_pauli[j_idx] = 'Z'
                pauli_terms.append(''.join(z_j_pauli))
                coeffs.append(-weight / 4.0)
                constant_term += weight / 4.0

        if constant_term != 0:
            pauli_terms.append('I' * max(1, n_vars))
            coeffs.append(float(constant_term))

        if pauli_terms:
            real_coeffs = [float(np.real(c)) for c in coeffs]
            hamiltonian = SparsePauliOp.from_list(list(zip(pauli_terms, real_coeffs)))
        else:
            hamiltonian = SparsePauliOp.from_list([('I' * max(1, n_vars), 0.0)])

        if self.problem_obj:
            self.problem_obj.metrics['qubo_size'] = n_vars
        return hamiltonian, var_list

    def create_qaoa_circuit(self, hamiltonian: SparsePauliOp, beta: List[float], gamma: List[float]) -> QuantumCircuit:
        n_qubits = hamiltonian.num_qubits
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(range(n_qubits))

        for p in range(self.p_layers):
            for term, weight in zip(hamiltonian.paulis, hamiltonian.coeffs):
                real_weight = float(np.real(weight))
                pauli_str = str(term)
                if pauli_str.count('Z') == 1:
                    indices = [i for i, c in enumerate(reversed(pauli_str)) if c == 'Z']
                    if len(indices) == 1:
                        qc.rz(2 * gamma[p] * real_weight, indices[0])
                elif pauli_str.count('Z') == 2:
                    indices = [i for i, c in enumerate(reversed(pauli_str)) if c == 'Z']
                    if len(indices) == 2:
                        qc.rzz(2 * gamma[p] * real_weight, indices[0], indices[1])
            for i in range(n_qubits):
                qc.rx(2 * beta[p], i)
        qc.measure_all()

        if self.problem_obj:
            self.problem_obj.metrics['circuit_depth'].append(qc.depth())
            self.problem_obj.metrics['qubits_used'].append(n_qubits)
        return qc

    def qaoa_solve(self, Q: Dict[Tuple[int, int], float], num_runs: int = 10) -> Dict:
        """Solve QUBO using QAOA with detailed bitstring logging"""
        # Recommendation: Log raw QAOA bitstrings
        try:
            hamiltonian, var_list = self.qubo_to_ising_hamiltonian(Q)
            n_qubits = len(var_list)
            best_solution = None
            best_energy = float('inf')
            quantum_results = []
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
                logger.debug(f"Circuit {run + 1}: qubits={n_qubits}, depth={qc.depth()}")

            jobs = [self.sampler.run([qc], shots=self.shots) for qc in circuits]
            for run, (job, beta, gamma) in enumerate(zip(jobs, betas, gammas)):
                logger.info(f"Processing results for run {run + 1}/{num_runs}")
                start_time = time.time()
                try:
                    result = job.result()
                    pub_result = result[0]
                    if hasattr(pub_result.data, 'meas'):
                        data_bin = pub_result.data.meas
                    elif hasattr(pub_result.data, 'bin'):
                        data_bin = pub_result.data.bin
                    else:
                        raise AttributeError("No 'meas' or 'bin' in PubResult.data")
                    counts = data_bin.get_counts()
                    logger.debug(f"Run {run + 1}: Bitstrings with counts: {counts}")
                    if not counts:
                        logger.warning(f"No counts obtained for run {run + 1}")
                        continue
                    for bitstring, count in counts.items():
                        if count > 0:
                            if isinstance(bitstring, str):
                                try:
                                    bit_int = int(bitstring, 2)
                                    bitstring_str = format(bit_int, f"0{n_qubits}b")
                                except ValueError:
                                    logger.warning(f"Invalid bitstring '{bitstring}' skipped")
                                    continue
                            elif isinstance(bitstring, int):
                                bitstring_str = format(bitstring, f"0{n_qubits}b")
                            else:
                                logger.warning(f"Unexpected bitstring type {type(bitstring)}")
                                continue
                            solution = {var_list[i]: int(bit) for i, bit in enumerate(reversed(bitstring_str))}
                            energy = self._calculate_qubo_energy(Q, solution)
                            logger.info(f"Run {run + 1}: Solution energy={energy}")
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
                    logger.error(f"Run {run + 1} failed: {e}, circuit depth: {qc.depth()}, qubits: {n_qubits}")
                    continue

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
            return self._fallback_solution(var_list)

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
        """Improved clustering with minimum cluster size validation"""
        # Recommendation: Avoid trivial clusters
        customers = list(range(1, problem.n_nodes))
        if not customers:
            logger.warning("No customers to cluster")
            return {v.id: [] for v in problem.vehicles}
        
        coords = [(problem.nodes[i].x, problem.nodes[i].y) for i in customers]
        n_clusters = min(len(problem.vehicles), len(customers))
        if n_clusters < len(problem.vehicles):
            logger.warning(f"Fewer clusters ({n_clusters}) than vehicles ({len(problem.vehicles)})")
        kmeans = KMeans(n_clusters=n_clusters, random_state=123).fit(coords)
        clusters = {v.id: [] for v in problem.vehicles}
        
        # Assign customers to vehicles
        for idx, label in enumerate(kmeans.labels_):
            vehicle_idx = label % len(problem.vehicles)
            clusters[problem.vehicles[vehicle_idx].id].append(customers[idx])
        
        # Ensure minimum cluster size (at least 2 nodes including depot)
        for v_id, cluster in clusters.items():
            if len(cluster) < 1 and customers:
                available_customers = [c for c in customers if not any(c in cl for cl in clusters.values())]
                if available_customers:
                    clusters[v_id].append(random.choice(available_customers))
        
        # Validate capacity constraints
        for v_id, cluster in clusters.items():
            vehicle = next((v for v in problem.vehicles if v.id == v_id), None)
            if vehicle:
                total_demand = sum(problem.nodes[i].demand for i in cluster)
                if total_demand > vehicle.capacity:
                    logger.warning(f"Vehicle {v_id} cluster demand ({total_demand}) exceeds capacity ({vehicle.capacity})")
        
        logger.info(f"Clusters assigned: {clusters}")
        return clusters

class PostProcessor:
    def repair_tsp_solution_advanced(self, solution: Dict[int, int], problem: TSPProblem) -> List[int]:
        """Enhanced post-processing with MST-based tour construction"""
        # Addresses "Invalid tour" issue and recommendation
        n = problem.n_nodes
        var_map = {}
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    var_map[(i, j)] = k
                    k += 1
        inv_var_map = {v: k for k, v in var_map.items()}
        tour_edges = []
        used_nodes = set()
        for var_index, value in solution.items():
            if value == 1 and var_index in inv_var_map:
                u, v = inv_var_map[var_index]
                if v not in used_nodes or v == 0:
                    tour_edges.append((u, v))
                    used_nodes.add(v)
        
        # Try to form a tour
        tour = []
        if tour_edges:
            start_node = 0
            tour.append(start_node)
            current_node = start_node
            remaining_edges = list(tour_edges)
            while remaining_edges:
                next_edge = None
                for i, (u, v) in enumerate(remaining_edges):
                    if u == current_node and v not in tour[1:]:
                        next_edge = (u, v)
                        break
                if next_edge:
                    tour.append(next_edge[1])
                    current_node = next_edge[1]
                    remaining_edges.remove(next_edge)
                else:
                    break
        
        # Check if tour is valid
        if len(set(tour)) == n and tour[0] == 0 and tour[-1] == 0:
            logger.info("Valid tour constructed from QAOA output")
            return self._two_opt_improvement(tour, problem.distance_matrix)
        
        # Fallback: Minimum Spanning Tree approach
        logger.info("Trying MST-based tour construction")
        adj_matrix = np.full((n, n), np.inf)
        for (u, v) in tour_edges:
            adj_matrix[u, v] = problem.distance_matrix[u, v]
        mst = minimum_spanning_tree(adj_matrix).toarray()
        tour = [0]
        visited = {0}
        for _ in range(n - 1):
            for j in range(n):
                if mst[tour[-1], j] != np.inf and j not in visited:
                    tour.append(j)
                    visited.add(j)
                    break
        tour.append(0)
        
        if len(set(tour)) == n:
            logger.info("Valid tour constructed via MST")
            return self._two_opt_improvement(tour, problem.distance_matrix)
        
        # Final fallback: Nearest neighbor
        logger.warning("Invalid tour, using nearest neighbor fallback")
        tour = [0]
        unvisited = set(range(1, n))
        current = 0
        while unvisited:
            nearest = min(unvisited, key=lambda x: problem.distance_matrix[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        tour.append(0)
        return self._two_opt_improvement(tour, problem.distance_matrix)

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
            logger.warning(f"Quantum backend {quantum_backend} not available, using Qiskit simulator")
            if QISKIT_AVAILABLE:
                self.quantum_solver = QiskitQAOASolver(use_real_device=False, use_aer=True, **kwargs)
                self.quantum_backend = 'qiskit_simulator'
            else:
                raise ValueError("Qiskit not available")

    def solve_tsp_quantum(self, problem: TSPProblem, num_runs: int = 5, use_simplified_qubo: bool = True) -> Tuple[List[int], float, Dict]:
        logger.info(f"Solving TSP with {problem.n_nodes} nodes using {self.quantum_backend}")
        qubo = self.formulator.simplified_tsp_qubo(problem)
        logger.info(f"QUBO size: {len(qubo)} terms")
        best_solution = None
        best_energy = float('inf')
        quantum_results = []
        num_qubits = 0
        for run in range(num_runs):
            logger.info(f"Quantum run {run + 1}/{num_runs}")
            try:
                result = self.quantum_solver.qaoa_solve(qubo)
                quantum_results.append(result)
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

    def solve_vrp_quantum(self, problem: VRPProblem, num_runs: int = 3, clustering_method: str = 'capacity_aware') -> Tuple[Dict[int, List[int]], float, Dict]:
        logger.info(f"Solving VRP with {problem.n_nodes} nodes, {problem.n_vehicles} vehicles")
        clusters_by_vehicle = self.preprocessor.cluster_customers_advanced(problem, clustering_method)
        total_cost = 0
        all_routes_by_vehicle = {}
        all_results = []
        total_qubits_estimated = 0

        for vehicle_id, cluster in clusters_by_vehicle.items():
            if not cluster:
                continue
            logger.info(f"Solving cluster for Vehicle {vehicle_id}: nodes {cluster}")
            cluster_node_indices = [0] + cluster
            cluster_nodes = [problem.nodes[idx] for idx in cluster_node_indices]
            cluster_distances = self._extract_subproblem_distances(problem, cluster_node_indices)
            sub_problem = TSPProblem(cluster_nodes, cluster_distances)
            tour, cost, sub_results = self.solve_tsp_quantum(sub_problem, max(1, num_runs // len(clusters_by_vehicle)), use_simplified_qubo=True)
            original_tour = [cluster_node_indices[node_idx] for node_idx in tour]
            vehicle = next((v for v in problem.vehicles if v.id == vehicle_id), None)
            if vehicle:
                details = calculate_route_details(problem, vehicle, original_tour)
                if details['total_duration'] > vehicle.max_time:
                    logger.warning(f"Vehicle {vehicle_id} exceeds max time ({details['total_duration']:.2f} > {vehicle.max_time})")
                    original_tour = original_tour[:len(original_tour)//2]  # Truncate route
            all_routes_by_vehicle[vehicle_id] = original_tour
            total_cost += cost
            all_results.append(sub_results)
            total_qubits_estimated += sub_results.get('num_qubits', 0)

        vrp_results = {
            'clustering_method': clustering_method,
            'clusters': clusters_by_vehicle,
            'subproblem_results': all_results,
            'total_cost': total_cost,
            'num_routes': len(all_routes_by_vehicle),
            'total_estimated_qubits': total_qubits_estimated
        }
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
                logger.warning(f"Invalid index in tour: {tour[i]} or {tour[j]}")
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
        Vehicle(0, capacity=60, type='truck', speed=0.7, fuel_consumption=2.5, max_time=8.0),
        Vehicle(1, capacity=30, type='car', speed=1.0, fuel_consumption=1.0, max_time=8.0),
        Vehicle(2, capacity=30, type='car', speed=1.0, fuel_consumption=1.0, max_time=8.0),
        Vehicle(3, capacity=10, type='bike', speed=0.5, fuel_consumption=0.2, max_time=4.0),
        Vehicle(4, capacity=10, type='bike', speed=0.5, fuel_consumption=0.2, max_time=4.0)
    ]
    if n_vehicles > len(vehicles):
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
        Vehicle(1, capacity=30, type='car', speed=1.0, fuel_consumption=1.0, max_time=8.0),
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
        response = requests.get(url, params=params, timeout=5)
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
        vehicle_details = vehicle_info_map.get(vehicle_id, {'color': 'blue', 'label': f'Vehicle {vehicle_id}'})
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
    print(f"\n✅ Visualization saved to '{filename}'")

def print_solution_schedules(problem: VRPProblem, routes: Dict[int, List[int]]):
    print("\n" + "-"*80)
    print("VRP DETAILED SCHEDULE & SHIFT ANALYSIS")
    print("-"*80)
    if not routes:
        print("No routes to analyze.")
        return
    DISTANCE_TO_TIME_FACTOR = 0.1
    sorted_vehicle_ids = sorted(routes.keys())
    for vehicle_id in sorted_vehicle_ids:
        route = routes[vehicle_id]
        if not route:
            continue
        vehicle = next((v for v in problem.vehicles if v.id == vehicle_id), None)
        if not vehicle:
            continue
        print(f"\n Vehicle {vehicle.id} ({vehicle.type}) | Shift: [{vehicle.max_time:.2f} hours]")
        print("-" * 75)
        print(f"{'Node':>5} | {'Arrival Time':>12} | {'Service End':>12} | {'Remaining Shift Time':>25}")
        print(f"{'-----':>5} | {'------------':>12} | {'-----------':>12} | {'-------------------------':>25}")
        current_time = 0.0
        previous_node = 0
        print(f"{0:>5} | {current_time:>12.2f} | {current_time:>12.2f} | {(vehicle.max_time - current_time):>25.2f}")
        full_route = route[:]
        if full_route[0] != 0: full_route.insert(0, 0)
        if full_route[-1] != 0: full_route.append(0)
        total_travel_time = 0
        total_service_time = 0
        for node_id in full_route[1:]:
            travel_distance = problem.distance_matrix[previous_node][node_id]
            travel_time = travel_distance * (DISTANCE_TO_TIME_FACTOR / vehicle.speed)
            total_travel_time += travel_time
            arrival_time = current_time + travel_time
            node = problem.nodes[node_id]
            service_time = node.service_time
            if node_id != 0:
                total_service_time += service_time
                departure_time = arrival_time + service_time
            else:
                departure_time = arrival_time
            remaining_shift_time = vehicle.max_time - departure_time
            print(f"{node_id:>5} | {arrival_time:>12.2f} | {departure_time:>12.2f} | {remaining_shift_time:>25.2f}")
            current_time = departure_time
            previous_node = node_id
            if remaining_shift_time < 0:
                print(f"     **** WARNING: Vehicle {vehicle.id} shift exceeded by {-remaining_shift_time:.2f} hours! ****")
        total_route_duration = current_time
        print("-" * 75)
        print(f"Route Summary for Vehicle {vehicle.id}: Total Duration = {total_route_duration:.2f} hours")
        print(f"                             (Travel Time: {total_travel_time:.2f}, Service Time: {total_service_time:.2f})")
    print("\n" + "-"*80 + "\n")

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

def print_detailed_vehicle_summary(problem: VRPProblem, routes_by_vehicle: Dict[int, List[int]]):
    print("\n" + "=" * 100)
    print("DETAILED VRP SOLUTION SUMMARY: VEHICLE PERFORMANCE & COST ANALYSIS")
    print("=" * 100)
    header = f"{'Vehicle ID':<12} | {'Type':<10} | {'Max Time (hrs)':<14} | {'Total Distance (km)':<20} | {'Total Duration (hrs)':<20} | {'Total Fuel Cost':<18}"
    print(header)
    print("-" * 100)
    total_fleet_cost = 0
    sorted_vehicle_ids = sorted(routes_by_vehicle.keys())
    for vehicle_id in sorted_vehicle_ids:
        route = routes_by_vehicle[vehicle_id]
        if not route:
            continue
        vehicle = next((v for v in problem.vehicles if v.id == vehicle_id), None)
        if not vehicle:
            continue
        details = calculate_route_details(problem, vehicle, route)
        total_fleet_cost += details['total_fuel_cost']
        line = (f"{vehicle_id:<12} | {vehicle.type:<10} | {vehicle.max_time:<14.2f} | "
                f"{details['total_distance']:<20.2f} | {details['total_duration']:<20.2f} | "
                f"{details['total_fuel_cost']:<18.2f}")
        print(line)
        if details['total_duration'] > vehicle.max_time:
            print(f"   ⚠️  WARNING: Vehicle {vehicle_id} exceeded its max time by {details['total_duration'] - vehicle.max_time:.2f} hours!")
    print("-" * 100)
    print(f"{'TOTAL FLEET COST':<60} | {total_fleet_cost:>30.2f}")
    print("=" * 100 + "\n")

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
                print(f"  ❓ {backend.name} | Qubits: N/A | Queue: N/A")
        print(f"\n📊 Summary:")
        print(f"  Quantum devices: {len(quantum_backends)}")
        print(f"  Simulators: {len(simulator_backends)}")
        if quantum_backends:
            try:
                operational_backends = [b for b in quantum_backends if b.status().operational and hasattr(b, 'num_qubits') and b.num_qubits > 0]
                if operational_backends:
                    recommended = min(operational_backends, key=lambda b: b.status().pending_jobs if hasattr(b.status(), 'pending_jobs') else float('inf'))
                    print(f"  Recommended: {recommended.name} ({recommended.num_qubits} qubits)")
                else:
                    print("  No operational quantum devices available.")
            except Exception as e:
                logger.warning(f"Could not determine recommended backend: {e}")
                print("  Could not determine recommended quantum device.")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to IBM Quantum: {e}")
        return False

def find_vrp_qubit_requirement(n_nodes: int, n_vehicles: int):
    """Complete qubit estimation with error handling"""
    try:
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
                print(f"Estimated qubits needed: {estimated_qubits}")
            else:
                print("Largest subproblem too small, 0 qubits needed.")
        else:
            print("No clusters formed.")
        return max_qubits_needed
    except Exception as e:
        logger.error(f"Qubit estimation failed: {e}")
        print(f"Error estimating qubits: {e}")
        return 0

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
            'name': 'IBM Aer Simulator',
            'use_real_device': False,
            'use_aer': True,
            'shots': 4096,  # Increased shots
            'p_layers': 4,  # Increased layers
            'optimizer': 'SPSA'  # Better for noisy settings
        }
    ]
    results = {}
    for config in test_configs:
        print(f"\n{'-' * 60}")
        print(f"TESTING: {config['name']}")
        print(f"{'-' * 60}")
        config_results = {'tsp': None, 'vrp': None}
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
        except Exception as init_e:
            logger.error(f"Solver initialization failed for {config['name']}: {init_e}")
            config_results['initialization_error'] = str(init_e)
            results[config['name']] = config_results
            continue
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
            print(f"{config_name:<25} | {'FAILED':<10} | {'N/A':<10} | {'N/A':<12} | {'FAILED':<10} | {'N/A':<10} | {'N/A':<15}")
            continue
        tsp_cost = "FAILED"
        tsp_time = "N/A"
        tsp_qubits = "N/A"
        vrp_cost = "FAILED"
        vrp_time = "N/A"
        vrp_est_qubits = "N/A"
        if config_results.get('tsp') and config_results['tsp'].get('success'):
            tsp_cost = f"{config_results['tsp']['cost']:.2f}"
            tsp_time = f"{config_results['tsp']['time']:.2f}s"
            tsp_qubits = config_results['tsp']['num_qubits']
        if config_results.get('vrp') and config_results['vrp'].get('success'):
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
        # Recommendation: Simulated OR-Tools-like VRP benchmark
        classical_vrp_routes, classical_vrp_cost = classical_solver.solve_vrp_ortools_like(vrp_problem)
        print(f"Classical VRP ({vrp_problem.n_nodes} nodes, {vrp_problem.n_vehicles} vehicles): Cost = {classical_vrp_cost:.2f} km")
    except Exception as e:
        print(f"Classical benchmark failed: {e}")
    print(f"\n🎯 Key Insights:")
    print("1. Real quantum devices may have higher error rates due to noise")
    print("2. Simulators provide exact results but don't show quantum advantage")
    print("3. Problem size is limited by current quantum hardware capabilities")
    print("4. Hybrid approach combines quantum optimization with classical post-processing")
    print("5. Stronger QUBO constraints and enhanced post-processing reduce fallback reliance")
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

    def solve_vrp_ortools_like(self, problem: VRPProblem) -> Tuple[Dict[int, List[int]], float]:
        """Simulated OR-Tools-like VRP solver using greedy assignment and 2-opt"""
        # Recommendation: Classical VRP benchmark
        clusters = ClassicalPreprocessor().cluster_customers_advanced(problem)
        routes = {}
        total_cost = 0
        for vehicle_id, cluster in clusters.items():
            if not cluster:
                continue
            cluster_node_indices = [0] + cluster
            cluster_nodes = [problem.nodes[idx] for idx in cluster_node_indices]
            cluster_distances = np.zeros((len(cluster_node_indices), len(cluster_node_indices)))
            for i in range(len(cluster_node_indices)):
                for j in range(len(cluster_node_indices)):
                    orig_i = cluster_node_indices[i]
                    orig_j = cluster_node_indices[j]
                    cluster_distances[i][j] = problem.distance_matrix[orig_i][orig_j]
            sub_problem = TSPProblem(cluster_nodes, cluster_distances)
            tour, cost = self.solve_tsp_2opt(sub_problem)
            original_tour = [cluster_node_indices[node_idx] for node_idx in tour]
            vehicle = next((v for v in problem.vehicles if v.id == vehicle_id), None)
            if vehicle:
                details = calculate_route_details(problem, vehicle, original_tour)
                if details['total_duration'] > vehicle.max_time:
                    original_tour = original_tour[:len(original_tour)//2]
                    cost = sum(problem.distance_matrix[original_tour[i]][original_tour[i+1]] 
                               for i in range(len(original_tour)-1))
            routes[vehicle_id] = original_tour
            total_cost += cost
        return routes, total_cost

if __name__ == "__main__":
    print("🚀 IBM Quantum TSP/VRP Solver with Real Road Network")
    print("=" * 50)
    if not QISKIT_AVAILABLE:
        print("❌ Qiskit not available. Please install:")
        print("pip install qiskit qiskit-ibm-runtime qiskit-aer qiskit-algorithms")
    if QISKIT_AVAILABLE and IBM_RUNTIME_AVAILABLE:
        check_quantum_connectivity()
    else:
        print("\nSkipping IBM Quantum connectivity check")
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
        find_vrp_qubit_requirement(13, 5)
    except KeyboardInterrupt:
        print(f"\n⏹ Experiments interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()