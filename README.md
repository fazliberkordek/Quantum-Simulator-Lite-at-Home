# 🧠 Multi-Framework Quantum Simulator

A lightweight quantum circuit simulator in Python that supports multiple quantum programming frameworks including **Qiskit** and **Cirq**. This simulator allows you to:

- Build and run circuits manually using native gate definitions
- Convert and simulate Qiskit or Cirq circuits using adapters
- Run quantum algorithms like Bell state, Grover’s algorithm, and more

---
# Running Custom Circuits

You can create and run your own quantum algorithms in two ways:


```python 
from quantum_simulator.core.gates import QuantumGate, GateType
from quantum_simulator.core.simulator import QuantumSimulator

circuit = [
    QuantumGate(GateType.H, [0]),
    QuantumGate(GateType.CNOT, [0, 1]),
    QuantumGate(GateType.MEASURE, [0], classical_bits=[0]),
    QuantumGate(GateType.MEASURE, [1], classical_bits=[1]),
]

sim = QuantumSimulator(2, shots=1000)
results = sim.run_shots(circuit)
print(results)
```


or use qiskit,cirq

```python
from qiskit import QuantumCircuit
from quantum_simulator.multi_framework import MultiFrameworkSimulator

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

sim = MultiFrameworkSimulator(shots=1000)
results = sim.simulate(qc, framework='qiskit')
print(results) 
```

