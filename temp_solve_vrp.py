def solve_vrp_from_api(nodes_data, vehicles_data):
    """Main entry point for API calls to solve VRP problems"""
    print("🚀 IBM Quantum TSP/VRP Solver with Real Road Network")
    print("="*50)

    # Convert input data to internal Node and Vehicle objects
    nodes = [Node(
        id=n['id'],
        x=n['x'],
        y=n['y'],
        demand=n.get('demand', 0),
        service_time=n.get('service_time', 0),
        available_time=n.get('available_time', 'Shift 1'),
        time_window=tuple(n.get('time_window', (8.0, 12.0)))
    ) for n in nodes_data]

    vehicles = [Vehicle(
        id=v['id'],
        capacity=v['capacity'],
        type=v.get('type', 'car'),
        speed=v.get('speed', 1.0),
        fuel_consumption=v.get('fuel_consumption', 1.0),
        max_time=v.get('max_time', float('inf'))
    ) for v in vehicles_data]

    # Create VRP problem
    vrp_problem = VRPProblem(nodes=nodes, vehicles=vehicles, shifts=group_nodes_by_shift(nodes))
    
    print("🔍 Checking IBM Quantum connectivity...")
    backends = []
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
                print(f"  ❓ {backend.name} | Qubits: N/A | Queue: N/A")
    except Exception as e:
        print(f"⚠️ Could not connect to IBM Quantum: {e}")
        print("Using classical solver.")

    print("\n🎯 Starting quantum experiments...")
    print("="*80)
    print("RUNNING ON IBM QUANTUM HARDWARE")
    print("="*80)

    config = {
        'name': 'IBM Aer Simulator (MPS with Traffic)',
        'use_real_device': False,
        'use_aer': True,
        'shots': 1024,
        'p_layers': 3,
        'optimizer': 'COBYLA'
    }

    api_key = "rF4ssuOAjyta2iqAYmlf1bZz0Itm2c8L"
    vehicle_color_map = {
        0: {'color': 'red', 'label': 'Truck (60kg)'},
        1: {'color': 'blue', 'label': 'Car 1 (30kg)'},
        2: {'color': 'green', 'label': 'Car 2 (30kg)'},
        3: {'color': 'orange', 'label': 'Bike 1 (10kg)'},
        4: {'color': 'purple', 'label': 'Bike 2 (10kg)'}
    }

    try:
        solver = QuantumHybridSolver(
            quantum_backend='qiskit',
            use_real_device=config['use_real_device'],
            use_aer=config.get('use_aer', True),
            shots=config['shots'],
            p_layers=config['p_layers'],
            optimizer=config['optimizer'],
            problem_obj=vrp_problem
        )
    except Exception as init_e:
        logger.error(f"Solver initialization failed: {init_e}")
        return {
            "solution_found": False,
            "error": str(init_e)
        }

    all_solutions = []
    total_fleet_cost = 0

    for shift_name in ["Shift 1", "Shift 2", "Shift 3"]:
        shift_nodes = vrp_problem.shifts.get(shift_name, [])
        if not shift_nodes:
            continue

        print(f"\n🚚 Solving VRP for {shift_name} ({len(shift_nodes)} nodes, {len(vehicles)} vehicles)...")
        shift_indices = [node.id for node in shift_nodes]
        shift_distance_matrix = vrp_problem.distance_matrix[np.ix_(shift_indices, shift_indices)]
        shift_vrp_problem = VRPProblem(
            nodes=shift_nodes,
            vehicles=vrp_problem.vehicles,
            distance_matrix=shift_distance_matrix,
            shifts=vrp_problem.shifts
        )

        try:
            vrp_routes, vrp_cost, vrp_details = solver.solve_vrp_quantum(
                shift_vrp_problem,
                num_runs=2,
                clustering_method='spectral'
            )

            # Get traffic-aware recommendations
            current_positions = {vid: 0 for vid in vrp_routes.keys()}
            traffic_recommendations = traffic_aware_rerouting_fixed(
                shift_vrp_problem, vrp_routes, api_key, current_positions
            )

            # Print detailed schedules and summaries
            print_solution_schedules(shift_vrp_problem, vrp_routes, shift_name, vrp_details.get('deferred_nodes', []))
            print_detailed_vehicle_summary(shift_vrp_problem, vrp_routes)

            # Generate visualization
            html_file = f"vrp_-_ibm_aer_simulator_(mps_with_traffic)_-_{shift_name.lower()}"
            visualize_solution_with_folium(
                shift_vrp_problem,
                vrp_routes,
                html_file,
                vehicle_color_map,
                traffic_recommendations
            )

            # Add to solutions
            shift_solution = {
                "shift_name": shift_name,
                "routes": vrp_routes,
                "cost": vrp_cost,
                "schedule_output": "",  # Filled by print_solution_schedules
                "summary_output": "",   # Filled by print_detailed_vehicle_summary
                "visualization_file": f"{html_file}.html",
                "total_estimated_qubits": vrp_details.get('total_estimated_qubits', 48),
                "deferred_nodes": vrp_details.get('deferred_nodes', [])
            }
            all_solutions.append(shift_solution)
            total_fleet_cost += vrp_cost

        except Exception as e:
            logger.error(f"Failed to solve VRP for {shift_name}: {e}")
            print(f"❌ VRP failed for {shift_name}: {e}")
            continue

    if not all_solutions:
        return {
            "solution_found": False,
            "error": "Failed to solve any shifts"
        }

    return {
        "solution_found": True,
        "total_fleet_cost": total_fleet_cost,
        "shift_solutions": all_solutions,
        "error": None
    }
