import React, { useState, useRef } from 'react';
import { MapPin, Truck, Play, Plus, Info, ChevronDown, ChevronUp, ArrowDown } from 'lucide-react';

interface Node {
  id: number;
  latitude: number;
  longitude: number;
  demand: number;
  serviceTime?: number;
  available?: string;
}

interface Vehicle {
  id: number;
  capacity: number;
  type: string;
  fuelConsumption: number;
}

function App() {
  const [currentStep, setCurrentStep] = useState(0);
  const [nodes, setNodes] = useState<Node[]>([
    { id: 1, latitude: 0, longitude: 0, demand: 0, serviceTime: 0, available: '8-12' }
  ]);
  const [vehicles, setVehicles] = useState<Vehicle[]>([
    { id: 1, capacity: 100, type: 'Standard', fuelConsumption: 0 }
  ]);
  const [expandedSections, setExpandedSections] = useState({
    nodes: true,
    vehicles: false,
    advanced: false
  });

  const inputSectionRef = useRef<HTMLDivElement>(null);

  const scrollToInputs = () => {
    inputSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // If any input changes, reset output
  const [showOutput, setShowOutput] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [outputTab, setOutputTab] = useState<'graphs' | 'other'>('graphs');
  const [infoTab, setInfoTab] = useState<'driver' | 'traffic'>('driver');
  const resetOutput = () => setShowOutput(false);

  const addNode = () => {
    const newId = Math.max(...nodes.map(n => n.id)) + 1;
    setNodes(prev => {
      resetOutput();
      return [
        ...prev,
        { id: newId, latitude: 0, longitude: 0, demand: 0, serviceTime: 0, available: '8-12' }
      ];
    });
  };

  const addVehicle = () => {
    const newId = Math.max(...vehicles.map(v => v.id)) + 1;
    setVehicles(prev => {
      resetOutput();
      return [
        ...prev,
        { id: newId, capacity: 100, type: 'Standard', fuelConsumption: 0 }
      ];
    });
  };

  const updateNode = (id: number, field: keyof Node, value: any) => {
    setNodes(nodes => {
      resetOutput();
      return nodes.map(node => node.id === id ? { ...node, [field]: value } : node);
    });
  };

  const updateVehicle = (id: number, field: keyof Vehicle, value: any) => {
    setVehicles(vehicles => {
      resetOutput();
      return vehicles.map(vehicle => vehicle.id === id ? { ...vehicle, [field]: value } : vehicle);
    });
  };

  const removeNode = (id: number) => {
    if (nodes.length > 1) {
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

  const solveVRP = () => {
    // Replace with your VRP solving logic
    setShowModal(true);
  };

  const handleModal = (action: 'yes' | 'reschedule') => {
    setShowModal(false);
    setShowOutput(true); // Always show output regardless of action
    // Optionally handle reschedule logic here
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-black text-white">
      {/* Hero Section */}
      <div className="min-h-screen flex flex-col items-center justify-center px-6 relative overflow-hidden">
        {/* Animated background elements - fixed alignment */}
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

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center">
            <div className="w-1 h-3 bg-white/50 rounded-full mt-2 animate-pulse"></div>
          </div>
        </div>
      </div>

      {/* Input Section */}
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
            {/* Left Panel - Inputs */}
            <div className="space-y-6">
              {/* Progress Steps - clickable */}
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

              {/* Nodes Section */}
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
                                    value={node.latitude}
                                    onChange={(e) => updateNode(node.id, 'latitude', parseFloat(e.target.value) || 0)}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-purple-400 focus:ring-1 focus:ring-purple-400 transition-colors text-sm"
                                    placeholder="0.000000"
                                  />
                                </td>
                                <td className="py-1">
                                  <input
                                    type="number"
                                    step="0.000001"
                                    value={node.longitude}
                                    onChange={(e) => updateNode(node.id, 'longitude', parseFloat(e.target.value) || 0)}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-purple-400 focus:ring-1 focus:ring-purple-400 transition-colors text-sm"
                                    placeholder="0.000000"
                                  />
                                </td>
                                <td className="py-1">
                                  <input
                                    type="number"
                                    value={node.demand}
                                    onChange={(e) => updateNode(node.id, 'demand', parseInt(e.target.value) || 0)}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-purple-400 focus:ring-1 focus:ring-purple-400 transition-colors text-sm"
                                    placeholder="0"
                                  />
                                </td>
                                <td className="py-1">
                                  <input
                                    type="number"
                                    value={node.serviceTime || 0}
                                    onChange={(e) => updateNode(node.id, 'serviceTime', parseInt(e.target.value) || 0)}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-purple-400 focus:ring-1 focus:ring-purple-400 transition-colors text-sm"
                                    placeholder="0"
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
                                    disabled={nodes.length <= 1}
                                    className="px-3 py-1 text-red-400 hover:text-red-300 disabled:text-gray-600 disabled:cursor-not-allowed transition-colors rounded-full"
                                    title="Remove Node"
                                  >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3m5 0H6" /></svg>
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
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

              {/* Vehicles Section */}
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
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Fuel Consumption</th>
                              <th className="text-left py-3 text-sm font-medium text-gray-300">Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {vehicles.map((vehicle) => (
                              <tr key={vehicle.id} className="border-b border-white/5">
                                <td className="py-1">
                                  <input
                                    type="number"
                                    value={vehicle.capacity}
                                    onChange={(e) => updateVehicle(vehicle.id, 'capacity', parseInt(e.target.value) || 0)}
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
                                    <option value="Bike">Bike</option>
                                    <option value="Car">Car</option>
                                  </select>
                                </td>
                                <td className="py-1">
                                  <input
                                    type="number"
                                    value={vehicle.fuelConsumption || 0}
                                    onChange={(e) => updateVehicle(vehicle.id, 'fuelConsumption', parseFloat(e.target.value) || 0)}
                                    className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-full text-white placeholder-gray-400 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-colors text-sm"
                                    placeholder="0"
                                  />
                                </td>
                                <td className="py-1">
                                  <button
                                    onClick={() => removeVehicle(vehicle.id)}
                                    disabled={vehicles.length <= 1}
                                    className="px-3 py-1 text-red-400 hover:text-red-300 disabled:text-gray-600 disabled:cursor-not-allowed transition-colors rounded-full"
                                    title="Remove Vehicle"
                                  >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3m5 0H6" /></svg>
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
                      <button
                        onClick={solveVRP}
                        className="ml-4 mt-4 px-6 py-2 bg-gradient-to-r from-purple-600 to-cyan-500 rounded-full text-white font-semibold hover:scale-105 transition-all"
                      >
                        Solve Vehicle Routing Problem
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Solve Button removed, now part of Vehicles section */}
            </div>

            {/* Right Panel - Map or Output */}
            <div className="lg:sticky lg:top-8 h-fit flex flex-col gap-8">
              {/* Modal for delivery confirmation */}
              {showModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in-up">
                  <div className="bg-gradient-to-br from-gray-900/90 to-gray-800/90 border-2 border-cyan-400/40 rounded-2xl shadow-2xl p-8 min-w-[320px] max-w-xs flex flex-col items-center glassmorphic">
                    <div className="text-lg font-bold mb-4 text-white">Do you want your delivery?</div>
                    <div className="flex gap-4">
                      <button onClick={() => handleModal('yes')} className="px-6 py-2 rounded-full bg-gradient-to-r from-cyan-400 to-purple-500 text-white font-semibold shadow-lg hover:scale-105 hover:shadow-cyan-400/40 transition-all">Yes</button>
                      <button onClick={() => handleModal('reschedule')} className="px-6 py-2 rounded-full bg-gradient-to-r from-gray-700 to-gray-900 text-cyan-300 font-semibold shadow-lg hover:scale-105 hover:shadow-purple-400/40 transition-all">Reschedule</button>
                    </div>
                  </div>
                </div>
              )}
              {/* Show maps if not solved, else show output dashboard */}
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
                  {/* Stats */}
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
                // Output Dashboard
                <div className="w-full animate-fade-in-up">
                  {/* Tabs for Graphs/Other Info */}
                  <div className="flex gap-2 mb-6">
                    <button onClick={() => setOutputTab('graphs')} className={`px-6 py-2 rounded-full font-semibold transition-all ${outputTab==='graphs' ? 'bg-gradient-to-r from-cyan-400 to-purple-500 text-white shadow-lg' : 'bg-gray-800 text-cyan-300 hover:bg-cyan-900/40'}`}>Graphs</button>
                    <button onClick={() => setOutputTab('other')} className={`px-6 py-2 rounded-full font-semibold transition-all ${outputTab==='other' ? 'bg-gradient-to-r from-purple-500 to-cyan-400 text-white shadow-lg' : 'bg-gray-800 text-purple-300 hover:bg-purple-900/40'}`}>Other Info</button>
                  </div>
                  {/* Tab Content */}
                  {outputTab === 'graphs' && (
                    <div className="bg-white/10 rounded-2xl border border-cyan-400/40 shadow-lg p-6 glassmorphic animate-fade-in-up mb-6">
                      <div className="font-bold text-cyan-300 mb-2 text-lg">Routing Before Traffic</div>
                      <iframe
                        src="/vrp_shift_1_ibm_aer_simulator.html"
                        title="Routing Before Traffic"
                        className="w-full aspect-video rounded-xl border border-cyan-400/20 bg-white"
                      />
                    </div>
                  )}
                  {outputTab === 'other' && (
                    <div className="mt-2">
                      <div className="flex gap-2 mb-4">
                        <button
                          onClick={() => setInfoTab('driver')}
                          className={`px-4 py-1 rounded-full font-semibold transition-all ${infoTab==='driver' ? 'bg-cyan-500 text-white shadow' : 'bg-gray-800 text-cyan-300 hover:bg-cyan-900/40'}`}
                        >
                          Driver Shift
                        </button>
                        <button
                          onClick={() => setInfoTab('traffic')}
                          className={`px-4 py-1 rounded-full font-semibold transition-all ${infoTab==='traffic' ? 'bg-purple-500 text-white shadow' : 'bg-gray-800 text-purple-300 hover:bg-purple-900/40'}`}
                        >
                          Traffic
                        </button>
                      </div>
                      {infoTab === 'driver' && (
                        <div className="bg-white/10 rounded-2xl border border-cyan-400/40 shadow-lg p-6 glassmorphic animate-fade-in-up">
                          <div className="font-bold text-cyan-300 mb-2 text-lg">Driver Shift</div>
                          <div className="font-mono text-xs text-cyan-100 bg-black/30 rounded-xl p-4 max-h-60 overflow-auto">[Driver Shift Log Info Placeholder]</div>
                        </div>
                      )}
                      {infoTab === 'traffic' && (
                        <div className="bg-white/10 rounded-2xl border border-purple-400/40 shadow-lg p-6 glassmorphic animate-fade-in-up">
                          <div className="font-bold text-purple-300 mb-2 text-lg">Traffic</div>
                          <div className="font-mono text-xs text-purple-100 bg-black/30 rounded-xl p-4 max-h-60 overflow-auto">[Traffic Log Info Placeholder]</div>
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