#!/usr/bin/env python3
"""
Multi-Framework Quantum Simulator
A lightweight quantum circuit simulator that supports multiple quantum programming frameworks
including Qiskit and Cirq, designed to run locally on user devices.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import json
import time
from dataclasses import dataclass
from enum import Enum

class GateType(Enum):
    """Supported quantum gate types"""
    X = "X"
    Y = "Y" 
    Z = "Z"
    H = "H"
    S = "S"
    T = "T"
    RX = "RX"
    RY = "RY"
    RZ = "RZ"
    CNOT = "CNOT"
    CZ = "CZ"
    SWAP = "SWAP"
    TOFFOLI = "TOFFOLI"
    MEASURE = "MEASURE"

@dataclass
class QuantumGate:
    """Represents a quantum gate operation"""
    gate_type: GateType
    qubits: List[int]
    parameters: Optional[List[float]] = None
    classical_bits: Optional[List[int]] = None

class QuantumState:
    """Manages quantum state vector and operations"""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0  # Initialize to |00...0⟩
        self.classical_bits = [0] * num_qubits
    
    def get_probability(self, state_index: int) -> float:
        """Get probability of measuring a specific computational basis state"""
        return abs(self.state_vector[state_index]) ** 2
    
    def get_probabilities(self) -> np.ndarray:
        """Get probabilities for all computational basis states"""
        return np.abs(self.state_vector) ** 2
    
    def normalize(self):
        """Normalize the quantum state"""
        norm = np.linalg.norm(self.state_vector)
        if norm > 1e-10:
            self.state_vector /= norm
    
    def copy(self):
        """Create a copy of the quantum state"""
        new_state = QuantumState(self.num_qubits)
        new_state.state_vector = self.state_vector.copy()
        new_state.classical_bits = self.classical_bits.copy()
        return new_state

class GateLibrary:
    """Library of quantum gate matrices"""
    
    @staticmethod
    def get_single_qubit_gates() -> Dict[GateType, np.ndarray]:
        """Return dictionary of single-qubit gate matrices"""
        return {
            GateType.X: np.array([[0, 1], [1, 0]], dtype=complex),
            GateType.Y: np.array([[0, -1j], [1j, 0]], dtype=complex),
            GateType.Z: np.array([[1, 0], [0, -1]], dtype=complex),
            GateType.H: np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            GateType.S: np.array([[1, 0], [0, 1j]], dtype=complex),
            GateType.T: np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
        }
    
    @staticmethod
    def rx_gate(theta: float) -> np.ndarray:
        """Rotation around X-axis"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    
    @staticmethod
    def ry_gate(theta: float) -> np.ndarray:
        """Rotation around Y-axis"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    @staticmethod
    def rz_gate(theta: float) -> np.ndarray:
        """Rotation around Z-axis"""
        return np.array([[np.exp(-1j * theta / 2), 0], 
                        [0, np.exp(1j * theta / 2)]], dtype=complex)

class QuantumSimulator:
    """Core quantum circuit simulator"""
    
    def __init__(self, num_qubits: int, shots: int = 1024):
        self.num_qubits = num_qubits
        self.shots = shots
        self.state = QuantumState(num_qubits)
        self.gate_library = GateLibrary()
        self.single_qubit_gates = self.gate_library.get_single_qubit_gates()
        
    def reset(self):
        """Reset simulator to initial state"""
        self.state = QuantumState(self.num_qubits)
    
    def apply_single_qubit_gate(self, gate: QuantumGate):
        """Apply single-qubit gate to the quantum state"""
        if gate.gate_type in [GateType.RX, GateType.RY, GateType.RZ]:
            if not gate.parameters:
                raise ValueError(f"Rotation gate {gate.gate_type} requires parameter")
            
            if gate.gate_type == GateType.RX:
                gate_matrix = self.gate_library.rx_gate(gate.parameters[0])
            elif gate.gate_type == GateType.RY:
                gate_matrix = self.gate_library.ry_gate(gate.parameters[0])
            else:  # RZ
                gate_matrix = self.gate_library.rz_gate(gate.parameters[0])
        else:
            gate_matrix = self.single_qubit_gates[gate.gate_type]
        
        qubit = gate.qubits[0]
        self._apply_single_qubit_matrix(gate_matrix, qubit)
    
    def apply_cnot_gate(self, control: int, target: int):
        """Apply CNOT gate"""
        for i in range(self.state.num_states):
            # Check if control qubit is 1
            if (i >> control) & 1:
                # Flip target qubit
                j = i ^ (1 << target)
                self.state.state_vector[i], self.state.state_vector[j] = \
                    self.state.state_vector[j], self.state.state_vector[i]
    
    def apply_cz_gate(self, control: int, target: int):
        """Apply controlled-Z gate"""
        for i in range(self.state.num_states):
            # Apply phase flip if both control and target are 1
            if ((i >> control) & 1) and ((i >> target) & 1):
                self.state.state_vector[i] *= -1
    
    def apply_swap_gate(self, qubit1: int, qubit2: int):
        """Apply SWAP gate"""
        for i in range(self.state.num_states):
            bit1 = (i >> qubit1) & 1
            bit2 = (i >> qubit2) & 1
            if bit1 != bit2:
                # Swap the qubits
                j = i ^ (1 << qubit1) ^ (1 << qubit2)
                if i < j:  # Avoid double swapping
                    self.state.state_vector[i], self.state.state_vector[j] = \
                        self.state.state_vector[j], self.state.state_vector[i]
    
    def _apply_single_qubit_matrix(self, gate_matrix: np.ndarray, qubit: int):
        """Apply single-qubit gate matrix to specified qubit"""
        new_state_vector = np.zeros_like(self.state.state_vector)
        
        for i in range(self.state.num_states):
            bit_value = (i >> qubit) & 1
            for new_bit in range(2):
                j = i ^ ((bit_value ^ new_bit) << qubit)
                new_state_vector[j] += gate_matrix[new_bit, bit_value] * self.state.state_vector[i]
        
        self.state.state_vector = new_state_vector
    
    def measure_qubit(self, qubit: int) -> int:
        """Measure a single qubit"""
        # Calculate probability of measuring |1⟩
        prob_1 = 0.0
        for i in range(self.state.num_states):
            if (i >> qubit) & 1:
                prob_1 += self.state.get_probability(i)
        
        # Simulate measurement
        measurement_result = 1 if np.random.random() < prob_1 else 0
        
        # Collapse the state
        self._collapse_state(qubit, measurement_result)
        
        return measurement_result
    
    def _collapse_state(self, qubit: int, measurement_result: int):
        """Collapse state after measurement"""
        for i in range(self.state.num_states):
            if ((i >> qubit) & 1) != measurement_result:
                self.state.state_vector[i] = 0
        
        self.state.normalize()
    
    def apply_gate(self, gate: QuantumGate):
        """Apply a quantum gate to the state"""
        if gate.gate_type == GateType.MEASURE:
            result = self.measure_qubit(gate.qubits[0])
            if gate.classical_bits:
                self.state.classical_bits[gate.classical_bits[0]] = result
            return result
        
        elif gate.gate_type in self.single_qubit_gates or gate.gate_type in [GateType.RX, GateType.RY, GateType.RZ]:
            self.apply_single_qubit_gate(gate)
        
        elif gate.gate_type == GateType.CNOT:
            self.apply_cnot_gate(gate.qubits[0], gate.qubits[1])
        
        elif gate.gate_type == GateType.CZ:
            self.apply_cz_gate(gate.qubits[0], gate.qubits[1])
        
        elif gate.gate_type == GateType.SWAP:
            self.apply_swap_gate(gate.qubits[0], gate.qubits[1])
        
        else:
            raise NotImplementedError(f"Gate {gate.gate_type} not implemented")
    
    def run_circuit(self, gates: List[QuantumGate]) -> Dict[str, Any]:
        """Run a complete quantum circuit"""
        start_time = time.time()
        
        for gate in gates:
            self.apply_gate(gate)
        
        execution_time = time.time() - start_time
        
        return {
            'final_state': self.state.state_vector.copy(),
            'probabilities': self.state.get_probabilities(),
            'classical_bits': self.state.classical_bits.copy(),
            'execution_time': execution_time
        }
    
    def run_shots(self, gates: List[QuantumGate]) -> Dict[str, int]:
        """Run circuit multiple times and collect measurement statistics"""
        results = {}
        
        for shot in range(self.shots):
            self.reset()
            self.run_circuit(gates)
            
            # Convert classical bits to bit string
            bit_string = ''.join(map(str, self.state.classical_bits))
            results[bit_string] = results.get(bit_string, 0) + 1
        
        return results

# Framework Adapters
class FrameworkAdapter(ABC):
    """Abstract base class for framework adapters"""
    
    @abstractmethod
    def convert_circuit(self, circuit) -> List[QuantumGate]:
        """Convert framework-specific circuit to our gate representation"""
        pass

class QiskitAdapter(FrameworkAdapter):
    """Adapter for Qiskit circuits"""
    
    def __init__(self):
        self.gate_mapping = {
            'x': GateType.X,
            'y': GateType.Y,
            'z': GateType.Z,
            'h': GateType.H,
            's': GateType.S,
            't': GateType.T,
            'rx': GateType.RX,
            'ry': GateType.RY,
            'rz': GateType.RZ,
            'cx': GateType.CNOT,
            'cz': GateType.CZ,
            'swap': GateType.SWAP,
            'measure': GateType.MEASURE
        }
    
    def convert_circuit(self, circuit) -> List[QuantumGate]:
        """Convert Qiskit circuit to our gate representation"""
        gates = []
        
        try:
            # Try to import qiskit to check if it's available
            import qiskit
            
            for instruction in circuit.data:
                gate_name = instruction.operation.name.lower()
                qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
                
                if gate_name in self.gate_mapping:
                    gate_type = self.gate_mapping[gate_name]
                    parameters = None
                    classical_bits = None
                    
                    # Handle parameterized gates
                    if hasattr(instruction.operation, 'params') and instruction.operation.params:
                        parameters = [float(param) for param in instruction.operation.params]
                    
                    # Handle measurements
                    if gate_name == 'measure' and instruction.clbits:
                        classical_bits = [circuit.find_bit(clbit).index for clbit in instruction.clbits]
                    
                    gates.append(QuantumGate(
                        gate_type=gate_type,
                        qubits=qubits,
                        parameters=parameters,
                        classical_bits=classical_bits
                    ))
                else:
                    print(f"Warning: Gate {gate_name} not supported, skipping")
            
        except ImportError:
            raise ImportError("Qiskit not installed. Please install qiskit to use QiskitAdapter")
        
        return gates

class CirqAdapter(FrameworkAdapter):
    """Adapter for Cirq circuits"""
    
    def __init__(self):
        self.gate_mapping = {
            'X': GateType.X,
            'Y': GateType.Y,
            'Z': GateType.Z,
            'H': GateType.H,
            'S': GateType.S,
            'T': GateType.T,
            'Rx': GateType.RX,
            'Ry': GateType.RY,
            'Rz': GateType.RZ,
            'CNOT': GateType.CNOT,
            'CZ': GateType.CZ,
            'SWAP': GateType.SWAP
        }
    
    def convert_circuit(self, circuit) -> List[QuantumGate]:
        """Convert Cirq circuit to our gate representation"""
        gates = []
        
        try:
            import cirq
            
            # Create a mapping from cirq qubits to integers
            qubit_map = {}
            for i, qubit in enumerate(sorted(circuit.all_qubits())):
                qubit_map[qubit] = i
            
            for moment in circuit:
                for operation in moment:
                    gate_name = operation.gate.__class__.__name__
                    qubits = [qubit_map[qubit] for qubit in operation.qubits]
                    
                    if gate_name in self.gate_mapping:
                        gate_type = self.gate_mapping[gate_name]
                        parameters = None
                        
                        # Handle parameterized gates
                        if hasattr(operation.gate, 'rads'):
                            parameters = [float(operation.gate.rads)]
                        elif hasattr(operation.gate, 'exponent'):
                            parameters = [float(operation.gate.exponent * np.pi)]
                        
                        gates.append(QuantumGate(
                            gate_type=gate_type,
                            qubits=qubits,
                            parameters=parameters
                        ))
                    else:
                        print(f"Warning: Gate {gate_name} not supported, skipping")
            
        except ImportError:
            raise ImportError("Cirq not installed. Please install cirq to use CirqAdapter")
        
        return gates

class MultiFrameworkSimulator:
    """Main simulator interface supporting multiple quantum frameworks"""
    
    def __init__(self, shots: int = 1024):
        self.shots = shots
        self.adapters = {
            'qiskit': QiskitAdapter(),
            'cirq': CirqAdapter()
        }
    
    def simulate(self, circuit, framework: str = 'qiskit', shots: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulate a quantum circuit from any supported framework
        
        Args:
            circuit: Quantum circuit object from supported framework
            framework: Framework name ('qiskit' or 'cirq')
            shots: Number of shots to run (default: use simulator default)
        
        Returns:
            Dictionary containing simulation results
        """
        if framework not in self.adapters:
            raise ValueError(f"Framework '{framework}' not supported. Available: {list(self.adapters.keys())}")
        
        # Convert circuit to our internal representation
        gates = self.adapters[framework].convert_circuit(circuit)
        
        # Determine number of qubits
        if framework == 'qiskit':
            num_qubits = circuit.num_qubits
        elif framework == 'cirq':
            num_qubits = len(circuit.all_qubits())
        else:
            # Fallback: count from gates
            num_qubits = max([max(gate.qubits) for gate in gates if gate.qubits]) + 1
        
        # Create simulator and run circuit
        shots_to_use = shots if shots is not None else self.shots
        simulator = QuantumSimulator(num_qubits, shots_to_use)
        
        # Check if circuit has measurements
        has_measurements = any(gate.gate_type == GateType.MEASURE for gate in gates)
        
        if has_measurements:
            # Run multiple shots
            counts = simulator.run_shots(gates)
            result = simulator.run_circuit(gates)
            
            return {
                'counts': counts,
                'shots': shots_to_use,
                'execution_time': result['execution_time'],
                'framework': framework
            }
        else:
            # Single execution for statevector
            result = simulator.run_circuit(gates)
            
            return {
                'statevector': result['final_state'],
                'probabilities': result['probabilities'],
                'execution_time': result['execution_time'],
                'framework': framework
            }
    
    def get_supported_frameworks(self) -> List[str]:
        """Get list of supported quantum frameworks"""
        return list(self.adapters.keys())
    
    def add_adapter(self, framework_name: str, adapter: FrameworkAdapter):
        """Add custom framework adapter"""
        self.adapters[framework_name] = adapter

# Example usage and testing
def create_example_circuits():
    """Create example circuits for testing"""
    examples = {}
    
    # Direct gate specification (framework-agnostic)
    bell_gates = [
        QuantumGate(GateType.H, [0]),
        QuantumGate(GateType.CNOT, [0, 1]),
        QuantumGate(GateType.MEASURE, [0], classical_bits=[0]),
        QuantumGate(GateType.MEASURE, [1], classical_bits=[1])
    ]
    examples['bell_direct'] = bell_gates
    
    return examples

def run_examples():
    """Run example simulations"""
    print("Multi-Framework Quantum Simulator Examples\n")
    
    simulator = MultiFrameworkSimulator(shots=1000)
    
    print(f"Supported frameworks: {simulator.get_supported_frameworks()}\n")
    
    # Test with direct gate specification
    examples = create_example_circuits()
    
    print("=== Bell State Circuit (Direct Gates) ===")
    bell_simulator = QuantumSimulator(2, 1000)
    result = bell_simulator.run_shots(examples['bell_direct'])
    
    print("Measurement results:")
    for state, count in sorted(result.items()):
        print(f"  |{state}⟩: {count} ({count/1000:.1%})")
    
    print(f"\nSimulator initialized with {bell_simulator.shots} shots")
    print("Circuit successfully executed!")

if __name__ == "__main__":
    run_examples()