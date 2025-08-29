from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
from fastapi.middleware.cors import CORSMiddleware  # Add CORS middleware
from dataclasses import dataclass
import uuid
import logging
import numpy as np
import time
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
# Response models
class VRPSolution(BaseModel):
    cost: float
    time: float
    estimated_qubits: int
    routes: Dict[int, List[int]]
    clusters: Dict[int, List[int]]
    method: str
    schedule: str
    summary: str
    visualization_file: str

class VRPResponse(BaseModel):
    status: str
    vrp_solution: VRPSolution
import os
# Assuming the provided quantum solver code is available in a module named quantum_solver
from quantum_solver import VRPProblem, Node, Vehicle, QuantumHybridSolver, print_solution_schedules, print_detailed_vehicle_summary, visualize_solution_with_folium

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Quantum VRP Solver API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow OPTIONS and POST
    allow_headers=["Content-Type"],
)
app.mount("/static", StaticFiles(directory="."), name="static")
# Pydantic models for input validation
class NodeInput(BaseModel):
    id: int
    x: float
    y: float
    demand: float = 0
    service_time: float = 0
    time_window: Optional[Tuple[float, float]] = None

class VehicleInput(BaseModel):
    id: int
    capacity: float
    max_time: float = float('inf')
    type: str = 'car'
    speed: float = 1.0
    fuel_consumption: float = 1.0
    start_depot: int = 0
    end_depot: int = 0

class VRPInput(BaseModel):
    nodes: List[NodeInput]
    vehicles: List[VehicleInput]
    num_runs: int = 2
    clustering_method: str = 'capacity_aware'

# Convert Pydantic input to solver-compatible dataclasses
def convert_to_solver_nodes(nodes: List[NodeInput]) -> List[Node]:
    return [
        Node(
            id=node.id,
            x=node.x,
            y=node.y,
            demand=node.demand,
            service_time=node.service_time,
            time_window=node.time_window
        ) for node in nodes
    ]

def convert_to_solver_vehicles(vehicles: List[VehicleInput]) -> List[Vehicle]:
    return [
        Vehicle(
            id=vehicle.id,
            capacity=vehicle.capacity,
            max_time=vehicle.max_time,
            type=vehicle.type,
            speed=vehicle.speed,
            fuel_consumption=vehicle.fuel_consumption,
            start_depot=vehicle.start_depot,
            end_depot=vehicle.end_depot
        ) for vehicle in vehicles
    ]

# Define vehicle color map for visualization
vehicle_color_map = {
    0: {'color': 'red', 'label': 'Truck 1'},
    1: {'color': 'blue', 'label': 'Truck 2'},
    2: {'color': 'green', 'label': 'Car 1'},
    3: {'color': 'orange', 'label': 'Bike 1'},
    4: {'color': 'purple', 'label': 'Bike 2'}
}

@app.post("/solve_vrp/", response_model=VRPResponse)
async def solve_vrp(vrp_input: VRPInput):
    try:
        start_time = time.time()
        logger.info(f"Received VRP input with {len(vrp_input.nodes)} nodes and {len(vrp_input.vehicles)} vehicles")
        
        vrp_problem = VRPProblem(
            nodes=[Node(**node.dict()) for node in vrp_input.nodes],
            vehicles=[Vehicle(**vehicle.dict()) for vehicle in vrp_input.vehicles]
        )
        
        solver = QuantumHybridSolver()
        routes, total_cost, vrp_details = solver.solve_vrp_quantum(
            vrp_problem,
            num_runs=vrp_input.num_runs,
            clustering_method=vrp_input.clustering_method
        )
        
        # Validate routes
        if not isinstance(routes, dict):
            logger.error(f"Routes returned from solver is not a dict: type={type(routes)}, value={routes}")
            raise ValueError(f"Routes must be a dictionary, got {type(routes)}")
        
        schedule = print_solution_schedules(vrp_problem, routes)
        summary = print_detailed_vehicle_summary(vrp_problem, routes, total_cost)
        
        # Generate visualization using folium
        map_filename = f"vrp_solution_{uuid.uuid4()}.html"
        visualize_solution_with_folium(vrp_problem, routes, "VRP Solution", vehicle_color_map)
        
        vrp_solution = VRPSolution(
            cost=total_cost,
            time=time.time() - start_time,
            estimated_qubits=vrp_details.get('total_estimated_qubits', 0),
            routes=routes,
            clusters=vrp_details.get('clusters', {}),
            method=vrp_details.get('method', 'qiskit_simulator'),
            schedule=schedule,
            summary=summary,
            visualization_file=map_filename
        )
        
        return VRPResponse(status="success", vrp_solution=vrp_solution)
            
        return response
    except Exception as e:
        logger.error(f"VRP solving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VRP solving failed: {str(e)}")
@app.get("/download_visualization/{filename}")
async def download_visualization(filename: str):
    file_path = filename
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='text/html', filename=filename)
    else:
        raise HTTPException(status_code=404, detail="Visualization file not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)