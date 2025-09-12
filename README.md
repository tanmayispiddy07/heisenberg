# Quantum Path Planning for Delivery Vehicles  

> **Optimizing Vehicle Routing with Quantum Computing**  
This project explores how **quantum computing techniques** can revolutionize path planning for delivery vehicles. Our solution focuses on minimizing **travel time, fuel consumption, and operational costs**, while accounting for **traffic patterns, delivery time windows, and vehicle capacity constraints**.

---

## Project Overview  

Traditional vehicle routing algorithms often struggle as the number of vehicles, customers, and constraints grow.  
Our approach uses **Quantum Computing** (QUBO + Quantum Annealing) to solve this **Vehicle Routing Problem (VRP)** more efficiently, aiming for **optimal or near-optimal delivery routes**.  

---

## Work Accomplished  

### Quantum Algorithm Development  
- Designed a **QUBO-based model** for VRP.  
- Exploited **quantum annealing** to compute optimal routes efficiently.  

### Problem Formulation  
- Modeled delivery path planning as a **graph-based optimization problem**.  
- Incorporated constraints:  
  -  Road networks  
  -  Time-dependent traffic data  
  -  Delivery priorities & vehicle capacity  

### Simulation & Testing  
- Simulated solutions on a **quantum simulator** and compared against:  
  - Dijkstra’s Algorithm  
  - Genetic Algorithms  
- Achieved **~15% improvement in route efficiency** for small-scale delivery fleets.  

### Data Integration  
- Integrated **real-world datasets**:  
  - GPS-based road networks  
  - Historical traffic data  
- Created **realistic test scenarios** for evaluation.  

### Visualization  
- Developed interactive visualizations for **multi-vehicle coordination** and **dynamic constraints**.  
<p align="center">
 <img width="768" height="439" alt="image" src="https://github.com/user-attachments/assets/28114ed3-bacb-49ea-a251-efc795c08c50" />
</p>

### Prototype Deployment  
- Built a **prototype implementation** compatible with quantum hardware emulators.  
- Prepared a pipeline for **testing on actual quantum annealers**.
  
### Collaborative Analysis  
- Used **Jupyter Notebooks** for performance comparison between **quantum and classical approaches**.  
- Documented metrics such as:  
  -  Computation time  
  -  Route cost  
  -  Efficiency improvements  

---

## Live Demo  

**Check out our website:** [hisenberg.netlify.app](https://hisenberg.netlify.app/)

---

## Tech Stack  

| Category        | Tools & Frameworks |
|----------------|------------------|
| **Quantum Computing** | D-Wave Ocean SDK, QUBO Modeling |
| **Classical Benchmarks** | Dijkstra’s Algorithm, Genetic Algorithms |
| **Simulation** | Quantum Simulators |
| **Data** | GPS-based Road Networks, Historical Traffic Data |
| **Visualization** | Python (Matplotlib, Plotly) |
| **Deployment** | Netlify (for website), Quantum Hardware Emulator |

---

## Contributors  

👥 **Team Heisenberg**  
- Tanmayi Kona
- Maheswari Manikyalarao
- Jyothirmai Andluri
- Grishma Gayathri Vanama
- Maruthi Kumar Gude
- Balavardhan Tummalacherla

