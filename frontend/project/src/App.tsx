import React, { useState, useRef } from 'react';
import axios from 'axios';
import { MapPin, Truck, Play, Plus, ChevronDown, ChevronUp, ArrowDown, Loader2, AlertTriangle } from 'lucide-react';

interface Node {
  id: number;
  latitude: number;
  longitude: number;
  demand: number;
  serviceTime?: number;
  available?: string;
  timeWindow?: [number, number];
}

interface Vehicle {
  id: number;
  capacity: number;
  type: string;
  fuelConsumption: number;
  speed?: number;
  maxTime?: number;
}

interface ParsedOutput {
  connectivity: string;
  tspProblem: string;
  vrpProblem: string;
  solution: string;
  schedules: string;
  vehicleSummary: string;
  visualization: string;
  trafficAnalysis: string;
  experimentResults: string;
  mapFilename?: string;
}

function App() {
  const [currentStep, setCurrentStep] = useState(0);
  const [nodes, setNodes] = useState<Node[]>([
  // Shift 1
  { id: 0, latitude: 16.5193, longitude: 80.6185, demand: 0, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] },   // Vijayawada Railway Station
  { id: 1, latitude: 16.5250, longitude: 80.6095, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] },  // Kanaka Durga Temple
  { id: 2, latitude: 16.5130, longitude: 80.6287, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] },  // MG Road / Governorpet
  { id: 3, latitude: 16.4985, longitude: 80.6569, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] },  // Benz Circle
  { id: 4, latitude: 16.4926, longitude: 80.6715, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] },  // Autonagar
  { id: 5, latitude: 16.4824, longitude: 80.6957, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] },  // Poranki
  { id: 6, latitude: 16.5158, longitude: 80.6168, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] },  // Machavaram
  { id: 7, latitude: 16.5283, longitude: 80.6469, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] },  // Siddhartha Medical College
  { id: 8, latitude: 16.5325, longitude: 80.6360, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] },  // Gunadala
  { id: 9, latitude: 16.5105, longitude: 80.6380, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] },  // Labbipet
  { id: 10, latitude: 16.5134, longitude: 80.6255, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] }, // Raghavaiah Park
  { id: 11, latitude: 16.5028, longitude: 80.6305, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] }, // Krishnalanka
  { id: 12, latitude: 16.5169, longitude: 80.6671, demand: 10, serviceTime: 0.5, available: '8-12', timeWindow: [8.0, 12.0] }, // Ramavarappadu

  // Shift 2
  { id: 0, latitude: 16.5193, longitude: 80.6185, demand: 0, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 1, latitude: 16.5250, longitude: 80.6095, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 2, latitude: 16.5130, longitude: 80.6287, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 3, latitude: 16.4985, longitude: 80.6569, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 4, latitude: 16.4926, longitude: 80.6715, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 5, latitude: 16.4824, longitude: 80.6957, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 6, latitude: 16.5158, longitude: 80.6168, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 7, latitude: 16.5283, longitude: 80.6469, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 8, latitude: 16.5325, longitude: 80.6360, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 9, latitude: 16.5105, longitude: 80.6380, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 10, latitude: 16.5134, longitude: 80.6255, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 11, latitude: 16.5028, longitude: 80.6305, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },
  { id: 12, latitude: 16.5169, longitude: 80.6671, demand: 10, serviceTime: 0.5, available: '12-16', timeWindow: [12.0, 16.0] },

  // Shift 3
  { id: 0, latitude: 16.5193, longitude: 80.6185, demand: 0, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 1, latitude: 16.5250, longitude: 80.6095, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 2, latitude: 16.5130, longitude: 80.6287, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 3, latitude: 16.4985, longitude: 80.6569, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 4, latitude: 16.4926, longitude: 80.6715, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 5, latitude: 16.4824, longitude: 80.6957, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 6, latitude: 16.5158, longitude: 80.6168, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 7, latitude: 16.5283, longitude: 80.6469, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 8, latitude: 16.5325, longitude: 80.6360, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 9, latitude: 16.5105, longitude: 80.6380, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 10, latitude: 16.5134, longitude: 80.6255, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 11, latitude: 16.5028, longitude: 80.6305, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] },
  { id: 12, latitude: 16.5169, longitude: 80.6671, demand: 10, serviceTime: 0.5, available: '16-20', timeWindow: [16.0, 20.0] }
]);
  const [vehicles, setVehicles] = useState<Vehicle[]>([
  { id: 0, capacity: 60, type: 'truck', fuelConsumption: 2.5, speed: 0.7, maxTime: 8.0 },
  { id: 1, capacity: 30, type: 'car', fuelConsumption: 1.0, speed: 1.0, maxTime: 8.0 },
  { id: 2, capacity: 30, type: 'car', fuelConsumption: 1.0, speed: 1.0, maxTime: 8.0 },
  { id: 3, capacity: 10, type: 'bike', fuelConsumption: 0.2, speed: 0.5, maxTime: 4.0 },
  { id: 4, capacity: 10, type: 'bike', fuelConsumption: 0.2, speed: 0.5, maxTime: 4.0 }
]);
  const [expandedSections, setExpandedSections] = useState({
    nodes: true,
    vehicles: false,
    advanced: false
  });
  const [showOutput, setShowOutput] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [outputTab, setOutputTab] = useState<'graphs' | 'other'>('graphs');
  const [infoTab, setInfoTab] = useState<'driver' | 'traffic'>('driver');
  const [parsedOutput, setParsedOutput] = useState<ParsedOutput | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [choice, setChoice] = useState<'overtime' | 'move'>('move'); // New state for overtime/move choice
  const [output, setOutput] = useState<string>('');
  const inputSectionRef = useRef<HTMLDivElement>(null);

  const scrollToInputs = () => {
    inputSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const resetOutput = () => {
    setShowOutput(false);
    setParsedOutput(null);
    setError(null);
  };

  const addNode = () => {
    const newId = Math.max(...nodes.map(n => n.id)) + 1;
    setNodes(prev => {
      resetOutput();
      return [
        ...prev,
        { id: newId, latitude: 0, longitude: 0, demand: 0, serviceTime: 0, available: '8-12', timeWindow: [8.0, 12.0] }
      ];
    });
  };

  const addVehicle = () => {
    const newId = Math.max(...vehicles.map(v => v.id)) + 1;
    setVehicles(prev => {
      resetOutput();
      return [
        ...prev,
        { id: newId, capacity: 100, type: 'Car', fuelConsumption: 1.0, speed: 1.0, maxTime: 8.0 }
      ];
    });
  };

  const updateNode = (id: number, field: keyof Node, value: any) => {
    setNodes(nodes => {
      resetOutput();
      const updated = nodes.map(node => node.id === id ? { ...node, [field]: value } : node);
      updated.forEach(node => {
        if (node.id === id && field === 'available' && value) {
          const [start, end] = value.split('-').map(Number);
          node.timeWindow = [start + 0.0, end + 0.0] as [number, number];
        }
        if (node.id === 0) {
          node.demand = 0;
          node.serviceTime = 0;
        }
      });
      console.log('Updated nodes:', updated);
      return updated;
    });
  };

  const updateVehicle = (id: number, field: keyof Vehicle, value: any) => {
    setVehicles(vehicles => {
      resetOutput();
      return vehicles.map(vehicle => vehicle.id === id ? { ...vehicle, [field]: value } : vehicle);
    });
  };

  const removeNode = (id: number) => {
    if (nodes.length > 1 && id !== 0) {
      setNodes(nodes => {
        resetOutput();
        return nodes.filter(node => node.id !== id);
      });
    }
  };

  const removeVehicle = (id: number) => {
    if (vehicles.length > 1) {
      setVehicles(vehicles => {
        resetOutput();
        return vehicles.filter(vehicle => vehicle.id !== id);
      });
    }
  };

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const mapAvailableToShift = (available: string): { available_time: string; time_window: [number, number] } => {
    const mapping: Record<string, { available_time: string; time_window: [number, number] }> = {
      '8-12': { available_time: 'Shift 1', time_window: [8.0, 12.0] },
      '12-16': { available_time: 'Shift 2', time_window: [12.0, 16.0] },
      '16-20': { available_time: 'Shift 3', time_window: [16.0, 16.0] }
    };
    return mapping[available] || { available_time: 'Shift 1', time_window: [8.0, 12.0] };
  };

  const solveVRP = async () => {
    let finalNodes = nodes;
    if (!nodes.find(n => n.id === 0)) {
      finalNodes = [
        { id: 0, latitude: 40.7484, longitude: -73.9857, demand: 0, serviceTime: 0, available: '8-12', timeWindow: [8.0, 12.0] },
        ...nodes
      ];
      setNodes(finalNodes);
    }

    setLoading(true);
    setError(null);
    try {
      const apiNodes = finalNodes.map(node => {
        const { available_time, time_window } = mapAvailableToShift(node.available || '8-12');
        const nodeData = {
          id: node.id,
          x: node.longitude,
          y: node.latitude,
          demand: node.demand || 0,
          service_time: node.serviceTime || 0,
          available_time,
          time_window: [time_window[0] + 0.0, time_window[1] + 0.0]
        };
        return nodeData;
      });

      const apiVehicles = vehicles.map(vehicle => ({
        id: vehicle.id,
        capacity: vehicle.capacity,
        type: vehicle.type,
        speed: vehicle.speed || 1.0,
        fuel_consumption: vehicle.fuelConsumption || 1.0,
        max_time: vehicle.maxTime || 8.0
      }));

      const payload = { nodes: apiNodes, vehicles: apiVehicles ,choice };
      console.log('Sending payload:', JSON.stringify(payload, null, 2));

      const response = await axios.post('http://127.0.0.1:8000/solve_vrp', payload, {
        headers: { 'Content-Type': 'application/json' }
      });
      console.log('API response:', response.data);
      if (response.data.status === 'success') {
        const outputString = response.data.output;
        console.log('Raw output:', outputString); // Debug
        setParsedOutput(parseOutputString(outputString));
        setShowOutput(true);
        setShowModal(true);
      } else {
        setError(`API Error: ${response.data.output}`);
      }
    } catch (err: any) {
      setError(`API Error: ${err.response?.data?.output || err.message}`);
    } finally {
      setLoading(false);
    }
  };
  const parseOutputString = (output: string): ParsedOutput => {
  const sections: ParsedOutput = {
    connectivity: '',
    tspProblem: '',
    vrpProblem: '',
    solution: '',
    schedules: '',
    vehicleSummary: '',
    visualization: '',
    trafficAnalysis: '',
    experimentResults: '',
    mapFilename: undefined
  };

  const lines = output.split('\n');
  let currentSection: keyof ParsedOutput | null = null;
  let inSchedules = false;
  let inVehicleSummary = false;
  let inTrafficAnalysis = false;
  let inExperimentResults = false;

  lines.forEach((line, index) => {
    const trimmedLine = line.trim();

    // Handle connectivity section
    if (line.startsWith('🚀 IBM Quantum TSP/VRP Solver')) {
      currentSection = 'connectivity';
      sections.connectivity += line + '\n';
    } 
    else if (currentSection === 'connectivity' && 
             (line.match(/🔍|✓|Available backends:|📊 Summary:|Qubits:|Queue:|Quantum devices:|Simulators:|Recommended:/) || 
              line.includes('Successfully connected'))) {
      sections.connectivity += line + '\n';
    }

    // Handle TSP Problem
    else if (line.startsWith('TSP Problem:')) {
      currentSection = 'tspProblem';
      sections.tspProblem += line + '\n';
    }

    // Handle VRP Problem
    else if (line.startsWith('VRP Problem:')) {
      currentSection = 'vrpProblem';
      sections.vrpProblem += line + '\n';
    }

    // Handle solution section (starts with dashes and TESTING)
    else if (line.includes('TESTING:') || line.startsWith('-'.repeat(60))) {
      currentSection = 'solution';
      sections.solution += line + '\n';
    }
    else if (currentSection === 'solution' && 
             (line.includes('Solving VRP') || line.startsWith('✓ VRP Solution:') || 
              line.includes('Routes:') || line.includes('Clusters:'))) {
      sections.solution += line + '\n';
    }

    // Handle schedules section
    else if (line.startsWith('VRP DETAILED SCHEDULE & SHIFT ANALYSIS')) {
      inSchedules = true;
      currentSection = 'schedules';
      sections.schedules += line + '\n';
    }
    else if (inSchedules) {
      if (line.startsWith('DETAILED VRP SOLUTION SUMMARY')) {
        inSchedules = false;
        inVehicleSummary = true;
        currentSection = 'vehicleSummary';
        sections.vehicleSummary += line + '\n';
      } else {
        sections.schedules += line + '\n';
      }
    }

    // Handle vehicle summary section
    else if (inVehicleSummary) {
      if (line.includes('Visualization saved to')) {
        inVehicleSummary = false;
        currentSection = 'visualization';
        sections.visualization += line + '\n';
        // Extract filename
        const match = line.match(/'([^']+\.html)'/);
        if (match) sections.mapFilename = match[1];
      } else {
        sections.vehicleSummary += line + '\n';
      }
    }

    // Handle visualization
    else if (line.includes('Visualization saved to')) {
      currentSection = 'visualization';
      sections.visualization += line + '\n';
      const match = line.match(/'([^']+\.html)'/);
      if (match) sections.mapFilename = match[1];
    }

    // Handle traffic analysis section
    else if (line.startsWith('ENHANCED TRAFFIC-AWARE REROUTING ANALYSIS')) {
      inTrafficAnalysis = true;
      currentSection = 'trafficAnalysis';
      sections.trafficAnalysis += line + '\n';
    }
    else if (inTrafficAnalysis) {
      if (line.startsWith('EXPERIMENT RESULTS SUMMARY')) {
        inTrafficAnalysis = false;
        inExperimentResults = true;
        currentSection = 'experimentResults';
        sections.experimentResults += line + '\n';
      } else {
        sections.trafficAnalysis += line + '\n';
      }
    }

    // Handle experiment results section
    else if (inExperimentResults || line.startsWith('EXPERIMENT RESULTS SUMMARY')) {
      if (!inExperimentResults) {
        inExperimentResults = true;
        currentSection = 'experimentResults';
      }
      sections.experimentResults += line + '\n';
    }

    // If we're in a section, continue adding lines to it
    else if (currentSection && trimmedLine) {
      sections[currentSection] += line + '\n';
    }

    // Debug unassigned significant lines
    else if (trimmedLine && !line.startsWith('='.repeat(10)) && !line.startsWith('-'.repeat(10))) {
      console.warn('Unassigned line:', line);
    }
  });

  // Clean up sections (remove trailing newlines and empty sections)
  Object.keys(sections).forEach(key => {
    if (typeof sections[key as keyof ParsedOutput] === 'string') {
      sections[key as keyof ParsedOutput] = (sections[key as keyof ParsedOutput] as string).trim();
    }
  });

  console.log('Parsed sections summary:', {
    connectivity: sections.connectivity.length > 0 ? 'Present' : 'Empty',
    tspProblem: sections.tspProblem.length > 0 ? 'Present' : 'Empty',
    vrpProblem: sections.vrpProblem.length > 0 ? 'Present' : 'Empty',
    solution: sections.solution.length > 0 ? 'Present' : 'Empty',
    schedules: sections.schedules.length > 0 ? 'Present' : 'Empty',
    vehicleSummary: sections.vehicleSummary.length > 0 ? 'Present' : 'Empty',
    visualization: sections.visualization.length > 0 ? 'Present' : 'Empty',
    trafficAnalysis: sections.trafficAnalysis.length > 0 ? 'Present' : 'Empty',
    experimentResults: sections.experimentResults.length > 0 ? 'Present' : 'Empty',
    mapFilename: sections.mapFilename || 'Not found'
  });

  return sections;
};
  const handleModal = (action: 'yes' | 'reschedule') => {
    setShowModal(false);
    if (action === 'reschedule') {
      console.log('Reschedule logic: re-run with adjusted parameters');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-black text-white">
      {error && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-gradient-to-br from-gray-900/90 to-gray-800/90 border-2 border-red-400/40 rounded-2xl shadow-2xl p-8 min-w-[320px] max-w-sm flex flex-col items-center">
            <AlertTriangle className="w-12 h-12 text-red-400 mb-4" />
            <div className="text-lg font-bold mb-4 text-white">Error</div>
            <p className="text-sm text-gray-300 text-center mb-6">{error}</p>
            <button
              onClick={() => setError(null)}
              className="px-6 py-2 rounded-full bg-gradient-to-r from-red-400 to-purple-500 text-white font-semibold shadow-lg hover:scale-105 transition-all"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in-up">
          <div className="bg-gradient-to-br from-gray-900/90 to-gray-800/90 border-2 border-cyan-400/40 rounded-2xl shadow-2xl p-8 min-w-[320px] max-w-xs flex flex-col items-center glassmorphic">
            <div className="text-lg font-bold mb-4 text-white">Confirm Optimized Routes?</div>
            <div className="flex gap-4">
              <button
                onClick={() => handleModal('yes')}
                className="px-6 py-2 rounded-full bg-gradient-to-r from-cyan-400 to-purple-500 text-white font-semibold shadow-lg hover:scale-105 hover:shadow-cyan-400/40 transition-all"
              >
                Yes
              </button>
              <button
                onClick={() => handleModal('reschedule')}
                className="px-6 py-2 rounded-full bg-gradient-to-r from-gray-700 to-gray-900 text-cyan-300 font-semibold shadow-lg hover:scale-105 hover:shadow-purple-400/40 transition-all"
              >
                Reschedule
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="min-h-screen flex flex-col items-center justify-center px-6 relative overflow-hidden">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-80 h-80 bg-purple-600/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-1/6 right-1/2 translate-x-1/2 w-96 h-96 bg-cyan-400/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        </div>

        <div className="text-center max-w-4xl mx-auto relative z-10">
          <div className="mb-8">
            <div className="inline-flex items-center gap-3 px-6 py-3 bg-white/5 backdrop-blur-sm rounded-full border border-white/10 mb-8">
              <Truck className="w-6 h-6 text-purple-400" />
              <span className="text-sm font-medium text-gray-300">Advanced Route Optimization</span>
            </div>
          </div>

          <h1 className="text-6xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-white via-purple-200 to-cyan-200 bg-clip-text text-transparent leading-tight">
            What should we
            <br />
            <span className="bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
              optimize today?
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-gray-300 mb-12 leading-relaxed max-w-3xl mx-auto">
            Solve complex vehicle routing problems with AI-powered optimization.
            Create efficient delivery routes, minimize costs, and maximize customer satisfaction.
          </p>

          <button
            onClick={scrollToInputs}
            className="group inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-purple-600 to-cyan-500 rounded-full font-semibold text-lg hover:shadow-2xl hover:shadow-purple-500/25 transition-all duration-300 hover:scale-105"
          >
            <span>Start Optimizing</span>
            <ArrowDown className="w-5 h-5 group-hover:translate-y-1 transition-transform" />
          </button>
        </div>

        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center">
            <div className="w-1 h-3 bg-white/50 rounded-full mt-2 animate-pulse"></div>
          </div>
        </div>
      </div>

      <div ref={inputSectionRef} className="min-h-screen px-6 py-16">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
              Configure Your Route Optimization
            </h2>
            <p className="text-gray-400 text-lg">
              Set up your nodes, vehicles, and constraints to find the optimal solution
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-8">
                {['Nodes', 'Vehicles'].map((step, index) => (
                  <div key={step} className="flex items-center cursor-pointer group" onClick={() => setCurrentStep(index)}>
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-all duration-200 ${
                      index === currentStep
                        ? 'bg-gradient-to-r from-purple-600 to-cyan-500 text-white scale-110 shadow-lg'
                        : 'bg-gray-700 text-gray-400 group-hover:scale-105'
                    }`}>
                      {index + 1}
                    </div>
                    <span className={`ml-3 font-medium ${
                      index === currentStep ? 'text-white' : 'text-gray-500 group-hover:text-white'
                    }`}>
                      {step}
                    </span>
                    {index < 1 && (
                      <div className={`w-16 h-0.5 mx-4 ${
                        index < currentStep ? 'bg-gradient-to-r from-purple-600 to-cyan-500' : 'bg-gray-700'
                      }`} />
                    )}
                  </div>
                ))}
              </div>

              {currentStep === 0 && (
                <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
                  <button
                    onClick={() => toggleSection('nodes')}
                    className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/5 transition-colors rounded-full"
                  >
                    <div className="flex items-center gap-3">
                      <MapPin className="w-5 h-5 text-purple-400" />
                      <h3 className="text-xl font-semibold">Delivery Nodes</h3>
                      <span className="px-2 py-1 bg-purple-600/20 text-purple-300 rounded-full text-sm">
                        {nodes.length}
                      </span>
                    </div>
                    {expandedSections.nodes ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                  </button>

                  {expandedSections.nodes && (
                    <div className="px-6 pb-6">
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b border-white/10">
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Latitude</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Longitude</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Demand</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Service Time</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Available</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {nodes.map((node) => (
                              <tr key={node.id} className="border-b border-white/5">
                                <td className="py-1">
                                  <input
                                    type="number"
                                    step="0.000001"
                                    min="-90"
                                    max="90"
                                    value={node.latitude}
                                    onChange={(e) => updateNode(node.id, 'latitude', e.target.value === '' ? 0 : parseFloat(e.target.value))}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-purple-400 focus:ring-1 focus:ring-purple-400 transition-colors text-sm"
                                    placeholder="0.000000"
                                    disabled={node.id === 0}
                                  />
                                </td>
                                <td className="py-1">
                                  <input
                                    type="number"
                                    step="0.000001"
                                    min="-180"
                                    max="180"
                                    value={node.longitude}
                                    onChange={(e) => updateNode(node.id, 'longitude', e.target.value === '' ? 0 : parseFloat(e.target.value))}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-purple-400 focus:ring-1 focus:ring-purple-400 transition-colors text-sm"
                                    placeholder="0.000000"
                                    disabled={node.id === 0}
                                  />
                                </td>
                                <td className="py-1">
                                  <input
                                    type="number"
                                    min="0"
                                    value={node.demand}
                                    onChange={(e) => updateNode(node.id, 'demand', e.target.value === '' ? 0 : parseInt(e.target.value))}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-purple-400 focus:ring-1 focus:ring-purple-400 transition-colors text-sm"
                                    placeholder="0"
                                    disabled={node.id === 0}
                                  />
                                </td>
                                <td className="py-1">
                                  <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    value={node.serviceTime || 0}
                                    onChange={(e) => updateNode(node.id, 'serviceTime', e.target.value === '' ? 0 : parseFloat(e.target.value))}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-purple-400 focus:ring-1 focus:ring-purple-400 transition-colors text-sm"
                                    placeholder="0"
                                    disabled={node.id === 0}
                                  />
                                </td>
                                <td className="py-1">
                                  <select
                                    value={node.available || '8-12'}
                                    onChange={(e) => updateNode(node.id, 'available', e.target.value)}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white focus:border-purple-400 focus:ring-1 focus:ring-purple-400 transition-colors appearance-none text-sm"
                                  >
                                    <option value="8-12">8-12</option>
                                    <option value="12-16">12-16</option>
                                    <option value="16-20">16-20</option>
                                  </select>
                                </td>
                                <td className="py-1">
                                  <button
                                    onClick={() => removeNode(node.id)}
                                    disabled={nodes.length <= 1 || node.id === 0}
                                    className="px-3 py-1 text-red-400 hover:text-red-300 disabled:text-gray-600 disabled:cursor-not-allowed transition-colors rounded-full"
                                    title="Remove Node"
                                  >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3m5 0H6" />
                                    </svg>
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <p className="text-xs text-gray-400 mt-2">Note: Depot (ID 0) is fixed and cannot be edited.</p>
                      <button
                        onClick={addNode}
                        className="mt-4 flex items-center gap-2 px-4 py-2 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-600/30 rounded-full text-purple-300 transition-colors"
                      >
                        <Plus className="w-4 h-4" />
                        Add Node
                      </button>
                      <button
                        onClick={() => setCurrentStep(1)}
                        className="ml-4 mt-4 px-6 py-2 bg-gradient-to-r from-purple-600 to-cyan-500 rounded-full text-white font-semibold hover:scale-105 transition-all"
                      >
                        Next
                      </button>
                    </div>
                  )}
                </div>
              )}

              {currentStep === 1 && (
                <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
                  <button
                    onClick={() => toggleSection('vehicles')}
                    className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/5 transition-colors rounded-full"
                  >
                    <div className="flex items-center gap-3">
                      <Truck className="w-5 h-5 text-cyan-400" />
                      <h3 className="text-xl font-semibold">Vehicle Fleet</h3>
                      <span className="px-2 py-1 bg-cyan-600/20 text-cyan-300 rounded-full text-sm">
                        {vehicles.length}
                      </span>
                    </div>
                    {expandedSections.vehicles ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                  </button>

                  {expandedSections.vehicles && (
                    <div className="px-6 pb-6">
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b border-white/10">
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Capacity</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Type</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Fuel Cons.</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Speed</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Max Time</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {vehicles.map((vehicle) => (
                              <tr key={vehicle.id} className="border-b border-white/5">
                                <td className="py-1">
                                  <input
                                    type="number"
                                    min="0"
                                    value={vehicle.capacity}
                                    onChange={(e) => updateVehicle(vehicle.id, 'capacity', e.target.value === '' ? 0 : parseInt(e.target.value))}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-colors text-sm"
                                    placeholder="100"
                                  />
                                </td>
                                <td className="py-1">
                                  <select
                                    value={vehicle.type}
                                    onChange={(e) => updateVehicle(vehicle.id, 'type', e.target.value)}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-colors appearance-none text-sm"
                                  >
                                    <option value="Truck">Truck</option>
                                    <option value="Car">Car</option>
                                    <option value="Bike">Bike</option>
                                  </select>
                                </td>
                                <td className="py-1">
                                  <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    value={vehicle.fuelConsumption || 0}
                                    onChange={(e) => updateVehicle(vehicle.id, 'fuelConsumption', e.target.value === '' ? 0 : parseFloat(e.target.value))}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-colors text-sm"
                                    placeholder="0"
                                  />
                                </td>
                                <td className="py-1">
                                  <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    value={vehicle.speed || 1.0}
                                    onChange={(e) => updateVehicle(vehicle.id, 'speed', e.target.value === '' ? 1.0 : parseFloat(e.target.value))}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-colors text-sm"
                                    placeholder="1.0"
                                  />
                                </td>
                                <td className="py-1">
                                  <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    value={vehicle.maxTime || 8.0}
                                    onChange={(e) => updateVehicle(vehicle.id, 'maxTime', e.target.value === '' ? 8.0 : parseFloat(e.target.value))}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-colors text-sm"
                                    placeholder="8.0"
                                  />
                                </td>
                                <td className="py-1">
                                  <button
                                    onClick={() => removeVehicle(vehicle.id)}
                                    disabled={vehicles.length <= 1}
                                    className="px-3 py-1 text-red-400 hover:text-red-300 disabled:text-gray-600 disabled:cursor-not-allowed transition-colors rounded-full"
                                    title="Remove Vehicle"
                                  >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3m5 0H6" />
                                    </svg>
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <button
                        onClick={addVehicle}
                        className="mt-4 flex items-center gap-2 px-4 py-2 bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-600/30 rounded-full text-cyan-300 transition-colors"
                      >
                        <Plus className="w-4 h-4" />
                        Add Vehicle
                      </button>
                      <div className="mb-6">
        <label className="block text-lg font-semibold mb-2">Handle Exceeding Customers:</label>
        <select
          value={choice}
          onChange={(e) => setChoice(e.target.value as 'overtime' | 'move')}
          className="p-2 border rounded-md bg-white shadow-sm"
        >
          <option value="move">Move to Next Shift</option>
          <option value="overtime">Allow Overtime</option>
        </select>
      </div>

                      <button
                        onClick={solveVRP}
                        disabled={loading}
                        className={`ml-4 mt-4 px-6 py-2 bg-gradient-to-r from-purple-600 to-cyan-500 rounded-full text-white font-semibold hover:scale-105 transition-all flex items-center gap-2 ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        {loading ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Solving...
                          </>
                        ) : (
                          <>
                            <Play className="w-5 h-5" />
                            Solve Vehicle Routing Problem
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="lg:sticky lg:top-8 h-fit flex flex-col gap-8">
              {!showOutput ? (
                <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-cyan-400/30 shadow-lg p-6 glassmorphic">
                  <h3 className="text-xl font-semibold mb-4 flex items-center gap-2 text-cyan-300">
                    <MapPin className="w-5 h-5 text-cyan-400" />
                    Live Preview Map
                  </h3>
                  <div className="aspect-square bg-gray-800/50 rounded-xl border border-cyan-400/20 flex items-center justify-center">
                    <div className="text-center text-cyan-300">
                      <MapPin className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p className="text-sm">Map visualization will appear here</p>
                      <p className="text-xs mt-1">Add nodes to see them plotted</p>
                    </div>
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <div className="bg-white/5 rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold text-purple-400">{nodes.length}</div>
                      <div className="text-sm text-gray-400">Nodes</div>
                    </div>
                    <div className="bg-white/5 rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold text-cyan-400">{vehicles.length}</div>
                      <div className="text-sm text-gray-400">Vehicles</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="w-full animate-fade-in-up">
                  <div className="flex gap-2 mb-6">
                    <button
                      onClick={() => setOutputTab('graphs')}
                      className={`px-6 py-2 rounded-full font-semibold transition-all ${
                        outputTab === 'graphs' ? 'bg-gradient-to-r from-cyan-400 to-purple-500 text-white shadow-lg' : 'bg-gray-800 text-cyan-300 hover:bg-cyan-900/40'
                      }`}
                    >
                      Graphs
                    </button>
                    <button
                      onClick={() => setOutputTab('other')}
                      className={`px-6 py-2 rounded-full font-semibold transition-all ${
                        outputTab === 'other' ? 'bg-gradient-to-r from-purple-500 to-cyan-400 text-white shadow-lg' : 'bg-gray-800 text-purple-300 hover:bg-purple-900/40'
                      }`}
                    >
                      Other Info
                    </button>
                  </div>
                  {outputTab === 'graphs' && parsedOutput && (
                    <div className="bg-white/10 rounded-2xl border border-cyan-400/40 shadow-lg p-6 glassmorphic animate-fade-in-up mb-6">
                      <div className="font-bold text-cyan-300 mb-2 text-lg">Optimized Routes</div>
                      {parsedOutput.mapFilename ? (
                        <iframe
                        src="vrp_-_ibm_aer_simulator_(mps_with_traffic)_-_shift_2 (2).html"
                        title="Routing Before Traffic"
                        className="w-full aspect-video rounded-xl border border-cyan-400/20 bg-white"
                      />
                      ) : (
                        <div className="text-center text-gray-400">No visualization available</div>
                      )}
                    </div>
                  )}
                  {outputTab === 'other' && parsedOutput && (
                    <div className="mt-2">
                      <div className="flex gap-2 mb-4">
                        <button
                          onClick={() => setInfoTab('driver')}
                          className={`px-4 py-1 rounded-full font-semibold transition-all ${
                            infoTab === 'driver' ? 'bg-cyan-500 text-white shadow' : 'bg-gray-800 text-cyan-300 hover:bg-cyan-900/40'
                          }`}
                        >
                          Driver Shift
                        </button>
                        <button
                          onClick={() => setInfoTab('traffic')}
                          className={`px-4 py-1 rounded-full font-semibold transition-all ${
                            infoTab === 'traffic' ? 'bg-purple-500 text-white shadow' : 'bg-gray-800 text-purple-300 hover:bg-purple-900/40'
                          }`}
                        >
                          Traffic Analysis
                        </button>
                      </div>
                      {infoTab === 'driver' && (
                        <div className="bg-white/10 rounded-2xl border border-cyan-400/40 shadow-lg p-6 glassmorphic animate-fade-in-up">
                          <div className="font-bold text-cyan-300 mb-2 text-lg">Driver Shift Schedules</div>
                          <pre className="font-mono text-xs text-cyan-100 bg-black/30 rounded-xl p-4 max-h-60 overflow-auto whitespace-pre-wrap">
                            {parsedOutput.schedules || 'No schedule data available'}
                          </pre>
                        </div>
                      )}
                      {infoTab === 'traffic' && (
                        <div className="bg-white/10 rounded-2xl border border-purple-400/40 shadow-lg p-6 glassmorphic animate-fade-in-up">
                          <div className="font-bold text-purple-300 mb-2 text-lg">Traffic Analysis</div>
                          <pre className="font-mono text-xs text-purple-100 bg-black/30 rounded-xl p-4 max-h-60 overflow-auto whitespace-pre-wrap">
                            {parsedOutput.trafficAnalysis || 'No traffic analysis available'}
                          </pre>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;