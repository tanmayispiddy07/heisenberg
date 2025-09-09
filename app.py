# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
from fastapi.middleware.cors import CORSMiddleware
import uuid
import logging
import os
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# --- Pydantic models for input validation ---
class NodeInput(BaseModel):
    id: int
    x: float
    y: float
    demand: float = 0
    service_time: float = 0
    available_time: Optional[str] = "Shift 1"
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

# --- Pydantic models for the response ---
class ShiftSolution(BaseModel):
    shift_name: str
    cost: float
    routes: Dict[str, List[int]]
    schedule_output: str
    summary_output: str

class VRPResponse(BaseModel):
    solution_found: bool
    total_fleet_cost: Optional[float] = None
    shift_solutions: Optional[List[ShiftSolution]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None

# --- API App Setup ---
app = FastAPI(title="Quantum VRP Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# You must mount a directory that *actually exists*
# Create a folder named 'visualizations' and use that.
if not os.path.exists("visualizations"):
    os.makedirs("visualizations")
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import solver logic ---
import sys
import os
# This ensures FastAPI can find your quantum_solver.py file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
try:
    from quantum_solver import solve_vrp_from_api
except ImportError as e:
    raise RuntimeError(f"Could not import from quantum_solver.py. Make sure it's in the same directory. Error: {e}")

# --- API Endpoints ---
@app.post("/solve_vrp/", response_model=VRPResponse)
async def solve_vrp(vrp_input: VRPInput):
    logger.info(f"Received VRP input with {len(vrp_input.nodes)} nodes and {len(vrp_input.vehicles)} vehicles")

    # Call the solver function from your other file
    result = solve_vrp_from_api(
        nodes_data=[node.dict() for node in vrp_input.nodes],
        vehicles_data=[vehicle.dict() for vehicle in vrp_input.vehicles]
    )

    # Note: I have commented out the visualization logic as it requires
    # significant changes in the original quantum_solver.py functions.
    # The primary focus is getting the API to work with your inputs.
    
    return JSONResponse(content=result)

@app.get("/visualizations/{filename}")
async def get_visualization(filename: str):
    file_path = os.path.join("visualizations", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='text/html', filename=filename)
    else:
        raise HTTPException(status_code=404, detail="Visualization file not found")

# --- No `if __name__ == "__main__"` here! ---
# The server is started from quantum_solver.py