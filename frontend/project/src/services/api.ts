export interface Node {
    id: number;
    latitude: number;
    longitude: number;
    demand: string;
    serviceTime: string;
    available: string;
}

export interface Vehicle {
    id: number;
    capacity: number;
    type: string;
    fuelConsumption: string;
    speed: string;
    maxTime: string;
}

export interface VRPSolution {
    success: boolean;
    visualizations: {
        [key: string]: {
            routes: { [key: string]: number[] };
            cost: number;
            time: number;
            qubits: number;
        }
    };
    metrics: {
        totalCost: number;
        totalTime: number;
        totalVehicles: number;
        totalNodes: number;
        averageTimePerShift: number;
        quantum: {
            totalQubits: number;
            runTime: number;
        }
    };
    driverShifts: {
        [key: string]: {
            routes: { [key: string]: number[] };
            cost: number;
            time: number;
            deferred_nodes: [number, string][];
        }
    };
    traffic: {
        [key: string]: {
            congested_segments: any[];
            alternative_routes: {
                vehicle_id: string;
                original_time: number;
                optimized_time: number;
                time_saved: number;
            }[];
            time_saved: number;
        }
    };
    error?: string;
}

const API_BASE_URL = 'http://127.0.0.1:8000';

const validateData = (nodes: Node[], vehicles: Vehicle[]) => {
    // Validate nodes
    if (!nodes.length) {
        throw new Error('At least one node is required');
    }

    // Check for valid coordinates
    nodes.forEach(node => {
        if (isNaN(node.latitude) || isNaN(node.longitude) ||
            node.latitude < -90 || node.latitude > 90 ||
            node.longitude < -180 || node.longitude > 180) {
            throw new Error(`Invalid coordinates for node ${node.id}`);
        }
    });

    // Validate vehicles
    if (!vehicles.length) {
        throw new Error('At least one vehicle is required');
    }

    vehicles.forEach(vehicle => {
        if (isNaN(vehicle.capacity) || vehicle.capacity <= 0) {
            throw new Error(`Invalid capacity for vehicle ${vehicle.id}`);
        }
        if (!['truck', 'car', 'bike', 'van', 'standard'].includes(vehicle.type.toLowerCase())) {
            throw new Error(`Invalid vehicle type: ${vehicle.type}`);
        }
    });
};

const processNodes = (nodes: Node[]): Node[] => {
    let processedNodes = [...nodes];
    const hasDepot = processedNodes.some(node => node.id === 0);
    
    if (!hasDepot) {
        processedNodes.unshift({
            id: 0,
            latitude: -73.9857,
            longitude: 40.7484,
            demand: '0',
            serviceTime: '0',
            available: '8-12'
        });
    }

    // Sort nodes to ensure depot is first
    processedNodes.sort((a, b) => {
        if (a.id === 0) return -1;
        if (b.id === 0) return 1;
        return a.id - b.id;
    });

    // Ensure all required fields have values
    return processedNodes.map(node => ({
        ...node,
        demand: node.demand || '0',
        serviceTime: node.serviceTime || '0',
        available: node.available || '8-12'
    }));
};

const processVehicles = (vehicles: Vehicle[]): Vehicle[] => {
    return vehicles.map((vehicle, idx) => ({
        ...vehicle,
        id: idx,
        type: vehicle.type.toLowerCase(),
        fuelConsumption: vehicle.fuelConsumption || '1.0',
        speed: vehicle.speed || '1.0',
        maxTime: vehicle.maxTime || '8.0'
    }));
};

export const solveVRP = async (nodes: Node[], vehicles: Vehicle[]): Promise<VRPSolution> => {
    try {
        // Validate input data
        validateData(nodes, vehicles);

        // Process nodes and vehicles
        const processedNodes = processNodes(nodes);
        const processedVehicles = processVehicles(vehicles);

        // Validate node IDs are sequential and unique
        const nodeIds = new Set(processedNodes.map(n => n.id));
        if (nodeIds.size !== processedNodes.length) {
            throw new Error('Duplicate node IDs detected');
        }

        const requestData = { 
            nodes: processedNodes,
            vehicles: processedVehicles
        };
        
        console.log('Sending data to backend:', JSON.stringify(requestData, null, 2));
        
        const response = await fetch(`${API_BASE_URL}/solve_vrp/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data as VRPSolution;
    } catch (error) {
        console.error('Error solving VRP:', error);
        throw error;
    }
};

export const getVisualization = async (filename: string): Promise<string> => {
    try {
        const response = await fetch(`${API_BASE_URL}/visualizations/${filename}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.text();
    } catch (error) {
        console.error('Error fetching visualization:', error);
        throw error;
    }
};
