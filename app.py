from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from quantum_solver import api_solve

app = FastAPI(
    title="Quantum VRP Solver API",
    description="API for quantum-based Vehicle Routing Problem solver using IBM Quantum and FastAPI/Uvicorn",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="C:/Users/tanma/Desktop/trail"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"  # Vite default port (if using Vite)
    ],
    allow_credentials=True,  # For future authentication if needed
    allow_methods=["POST", "GET"],  # Allow POST for /solve_vrp, GET for static files
    allow_headers=["Content-Type"],  # Allow JSON headers
)
# Pydantic models for request validation
class Node(BaseModel):
    id: int
    x: float
    y: float
    demand: float
    service_time: float
    available_time: str
    time_window: List[float]

class Vehicle(BaseModel):
    id: int
    capacity: float
    type: str
    speed: float
    fuel_consumption: float
    max_time: float

class VRPSolveRequest(BaseModel):
    nodes: List[Node]
    vehicles: List[Vehicle]
    choice: str

# Solve VRP endpoint
@app.post("/solve_vrp")
async def solve_vrp(request: VRPSolveRequest):
    try:
        # Convert Pydantic models to dict for api_solve
        nodes_data = [node.dict() for node in request.nodes]
        vehicles_data = [vehicle.dict() for vehicle in request.vehicles]
        choice = request.choice
        if choice not in ["overtime", "move"]:
            choice = "move"
        # Call+ the solver
        output = api_solve(nodes_data, vehicles_data,choice)
        
        return {"status": "success", "output": output}
    except Exception as e:
        return {"status": "error", "output": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)