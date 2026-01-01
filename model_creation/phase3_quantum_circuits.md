# Phase 3: Quantum Circuit Design (Weeks 5-6)
## Quantum-Enhanced Cryptanalysis System

**Duration:** 2 weeks  
**Goal:** Design, implement, and validate parameterized quantum circuits for all 5 tasks

---

## OBJECTIVES

1. Design task-specific ansätze (6-10 qubits, 3-4 layers)
2. Implement circuits in Cirq
3. Verify gradient computation (parameter shift rule)
4. Barren plateau analysis
5. Simulation benchmarks

---

## INPUT SPECIFICATIONS

**Classical Baseline:** Performance metrics from Phase 2  
**Data:** Processed features from Phase 1

**Target Improvements:**
- Task 1: 86% → 88% (+2%)
- Task 2: 0.80 → 0.85 AUC (+0.05)
- Task 3: 85% → 87% (+2%)

---

## OUTPUT SPECIFICATIONS

### Quantum Circuits
```python
src/quantum/circuits.py:
- create_attack_classification_circuit()
- create_attack_success_circuit()
- create_algorithm_id_circuit()
- create_plaintext_recovery_circuit()  # QAOA
- create_key_properties_circuit()
```

### Reports
```
reports/quantum_circuit_report.pdf:
- Circuit diagrams (ASCII + visual)
- Parameter counts
- Gradient flow analysis
- Barren plateau assessment
- Simulation benchmarks
```

---

## WEEK 5: ANSATZ DESIGN & IMPLEMENTATION

### Day 1-2: Task 1 - Attack Classification Circuit

**Design Decisions:**
```
Circuit Specifications:
├── n_qubits: 8 (balance between expressivity and NISQ feasibility)
├── n_layers: 4 (avoid barren plateau)
├── Entanglement: Ring topology (hardware-efficient)
├── Data encoding: Angle encoding (first 8 features)
└── Measurements: ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩ (3 observables)
```

**Implementation:**
```python
import cirq
import sympy
import numpy as np

def create_attack_classification_circuit(n_qubits=8, n_layers=4, n_data_features=8):
    """
    Parameterized Quantum Circuit for Attack Classification
    
    Returns:
        circuit: Cirq circuit with data and trainable symbols
        data_symbols: Input feature symbols
        param_symbols: Trainable parameter symbols
    """
    
    # Qubits
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Data encoding symbols (x₀, x₁, ..., xₙ)
    data_symbols = [sympy.Symbol(f'x{i}') for i in range(n_data_features)]
    
    # Layer 1: Data Encoding (Angle Encoding)
    for i in range(n_data_features):
        # Encode feature xᵢ as rotation angle on qubit i
        circuit.append(cirq.ry(data_symbols[i])(qubits[i]))
    
    # Variational Layers
    param_symbols = []
    for layer in range(n_layers):
        # Single-qubit parameterized rotations
        layer_symbols = [sympy.Symbol(f'theta_{layer}_{i}') for i in range(n_qubits)]
        param_symbols.extend(layer_symbols)
        
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(layer_symbols[i])(qubit))
        
        # Entanglement: Ring pattern (CNOT chain)
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
        
        # Close the ring (optional, for stronger entanglement)
        circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
    
    return circuit, data_symbols, param_symbols

# Create circuit
circuit, data_syms, theta_syms = create_attack_classification_circuit()

# Visualize
print(circuit)
print(f"Total parameters: {len(theta_syms)}")  # Should be 4 layers × 8 qubits = 32
```

**Circuit Diagram (ASCII):**
```
0: ───Ry(x0)───Ry(θ₀₀)───@───────────────────Ry(θ₁₀)───@───────────────────(...)
                         │                              │
1: ───Ry(x1)───Ry(θ₀₁)───X───@───────────────Ry(θ₁₁)───X───@───────────────(...)
                             │                              │
2: ───Ry(x2)───Ry(θ₀₂)───────X───@───────────Ry(θ₁₂)───────X───@───────────(...)
                                 │                              │
⋮
7: ───Ry(x7)───Ry(θ₀₇)───────────────@───────Ry(θ₁₇)───────────────@───────(...)
                                     │                              │
0: ─────────────────────────────────X───────────────────────────────X───────(...)
```

**Measurement Operators:**
```python
# Define observables for measurement
observables = [
    cirq.Z(qubits[0]),  # ⟨Z₀⟩
    cirq.Z(qubits[1]),  # ⟨Z₁⟩
    cirq.Z(qubits[2])   # ⟨Z₂⟩
]

# These will give 3 expectation values in [-1, 1]
# Classical post-processing will map to attack categories
```

**Deliverable:** `src/quantum/circuits.py` (partial)

---

### Day 3: Task 2 - Attack Success Prediction Circuit

**Simplified Binary Circuit:**
```python
def create_attack_success_circuit(n_qubits=6, n_layers=3):
    """
    Binary classification circuit for attack success prediction.
    Simpler than Task 1 (fewer qubits, fewer layers).
    """
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Data encoding
    data_symbols = [sympy.Symbol(f'x{i}') for i in range(n_qubits)]
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(data_symbols[i])(qubit))
    
    # Variational layers (3 repetitions)
    param_symbols = []
    for layer in range(n_layers):
        layer_symbols = [sympy.Symbol(f'theta_{layer}_{i}') for i in range(n_qubits)]
        param_symbols.extend(layer_symbols)
        
        # Rotations
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(layer_symbols[i])(qubit))
        
        # Entanglement
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
        circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
    
    return circuit, data_symbols, param_symbols

# Single qubit measurement for binary output
observable = cirq.Z(qubits[0])  # ⟨Z₀⟩ ∈ [-1, 1] → rescale to [0, 1]
```

---

### Day 4: Task 3 - Algorithm Identification Circuit

**Amplitude Encoding for Statistical Features:**
```python
def create_algorithm_id_circuit(n_qubits=10, n_layers=4):
    """
    Multi-class classification for algorithm identification.
    Uses amplitude encoding for ciphertext statistics.
    """
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Amplitude encoding: Normalize feature vector to unit vector
    # Then encode in state amplitudes (exponential compression)
    # For n qubits, can encode 2ⁿ features
    
    data_symbols = [sympy.Symbol(f'x{i}') for i in range(n_qubits)]
    
    # Data encoding layer (amplitude encoding simulation via rotations)
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(data_symbols[i])(qubit))
    
    # Variational feature extraction layers
    param_symbols = []
    for layer in range(n_layers):
        layer_symbols = [sympy.Symbol(f'theta_{layer}_{i}') for i in range(n_qubits)]
        param_symbols.extend(layer_symbols)
        
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(layer_symbols[i])(qubit))
        
        # Entanglement (ring + some all-to-all for expressivity)
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
        circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
    
    return circuit, data_symbols, param_symbols

# Multiple measurements for multi-class output
observables = [
    cirq.Z(qubits[0]),
    cirq.Z(qubits[1]),
    cirq.Z(qubits[2]),
    cirq.Z(qubits[3])
]  # 4 measurements → 4D quantum feature vector
```

---

### Day 5: Task 4 - Plaintext Recovery (QAOA)

**QAOA Circuit:**
```python
def create_plaintext_recovery_qaoa(n_qubits=8, p=2):
    """
    QAOA circuit for combinatorial search over plaintext space.
    
    Args:
        n_qubits: Number of bits to recover
        p: QAOA depth (number of cost + mixer layers)
    
    Returns:
        circuit with gamma and beta parameters
    """
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Initial state: Equal superposition |+⟩⊗ⁿ
    circuit.append([cirq.H(q) for q in qubits])
    
    # QAOA layers
    gamma_symbols = [sympy.Symbol(f'gamma_{i}') for i in range(p)]
    beta_symbols = [sympy.Symbol(f'beta_{i}') for i in range(p)]
    
    for layer in range(p):
        # Cost Hamiltonian: exp(-i γ H_C)
        # H_C encodes plaintext-ciphertext relationship
        # Simplified example: Ising model
        for i in range(n_qubits - 1):
            circuit.append(cirq.ZZ(qubits[i], qubits[i+1])**(gamma_symbols[layer]))
        
        # Mixer Hamiltonian: exp(-i β H_M)
        # H_M = ∑ᵢ Xᵢ (drive transitions between bitstrings)
        for qubit in qubits:
            circuit.append(cirq.rx(2 * beta_symbols[layer])(qubit))
    
    return circuit, gamma_symbols, beta_symbols

# Measurement: Sample bitstrings (not expectation values)
# Output: Most probable bitstring is candidate plaintext
```

---

## WEEK 6: VALIDATION & BENCHMARKING

### Day 6-7: Gradient Verification

**Parameter Shift Rule Implementation:**
```python
def parameter_shift_gradient(circuit, param_symbol, param_value, observable, data_values):
    """
    Compute gradient ∂⟨ψ|O|ψ⟩/∂θ using parameter shift rule.
    
    ∂f/∂θ = [f(θ + π/2) - f(θ - π/2)] / 2
    """
    shift = np.pi / 2
    
    # Resolve data symbols
    resolver_plus = {param_symbol: param_value + shift}
    resolver_plus.update({str(sym): val for sym, val in zip(data_symbols, data_values)})
    
    resolver_minus = {param_symbol: param_value - shift}
    resolver_minus.update({str(sym): val for sym, val in zip(data_symbols, data_values)})
    
    # Simulate circuits
    simulator = cirq.Simulator()
    
    result_plus = simulator.simulate(circuit, resolver_plus)
    expectation_plus = result_plus.final_state_vector.conj() @ observable @ result_plus.final_state_vector
    
    result_minus = simulator.simulate(circuit, resolver_minus)
    expectation_minus = result_minus.final_state_vector.conj() @ observable @ result_minus.final_state_vector
    
    gradient = (expectation_plus - expectation_minus) / 2
    return gradient.real

# Verify against numerical gradient
def numerical_gradient(circuit, param_symbol, param_value, observable, epsilon=1e-5):
    """Finite difference approximation."""
    # ... (similar implementation)
    
# Test
param_shift_grad = parameter_shift_gradient(circuit, theta_syms[0], 0.5, observable, data_values)
numerical_grad = numerical_gradient(circuit, theta_syms[0], 0.5, observable)

error = abs(param_shift_grad - numerical_grad)
print(f"Gradient error: {error}")
assert error < 1e-6, "Gradient verification failed!"
```

**Deliverable:** `src/quantum/differentiators.py`

---

### Day 8: Barren Plateau Analysis

**Gradient Norm Monitoring:**
```python
def analyze_barren_plateau(circuit, param_symbols, n_random_inits=100):
    """
    Check if gradients vanish with random parameter initialization.
    """
    gradient_norms = []
    
    for _ in range(n_random_inits):
        # Random parameter initialization
        params = np.random.uniform(0, 2*np.pi, len(param_symbols))
        
        # Compute all gradients
        grads = []
        for i, sym in enumerate(param_symbols):
            grad = parameter_shift_gradient(circuit, sym, params[i], observable, data_values)
            grads.append(grad)
        
        # Gradient norm
        grad_norm = np.linalg.norm(grads)
        gradient_norms.append(grad_norm)
    
    # Plot distribution
    plt.hist(gradient_norms, bins=30)
    plt.xlabel('Gradient Norm')
    plt.ylabel('Frequency')
    plt.title('Barren Plateau Analysis')
    plt.axvline(1e-6, color='r', linestyle='--', label='Vanishing threshold')
    plt.legend()
    plt.savefig('results/plots/barren_plateau_analysis.png')
    
    # Check if gradients vanish
    avg_norm = np.mean(gradient_norms)
    print(f"Average gradient norm: {avg_norm:.2e}")
    
    if avg_norm < 1e-6:
        print("⚠️ WARNING: Barren plateau detected! Consider:")
        print("  - Reducing circuit depth (fewer layers)")
        print("  - Problem-inspired ansatz")
        print("  - Layer-wise training")
    else:
        print("✅ No barren plateau - gradients are healthy")
    
    return gradient_norms

# Run analysis
gradient_norms = analyze_barren_plateau(circuit, theta_syms)
```

---

### Day 9-10: Simulation Benchmarks

**Performance Metrics:**
```python
import time

def benchmark_circuit(circuit, n_samples=100, batch_size=32):
    """
    Benchmark circuit simulation performance.
    """
    simulator = cirq.Simulator()
    
    # Forward pass benchmark
    start = time.time()
    for _ in range(n_samples // batch_size):
        # Simulate batch
        batch_data = np.random.randn(batch_size, len(data_symbols))
        
        for sample in batch_data:
            resolver = {str(sym): val for sym, val in zip(data_symbols, sample)}
            result = simulator.simulate(circuit, resolver)
    
    forward_time = (time.time() - start) / n_samples * 1000  # ms per sample
    
    # Gradient computation benchmark
    start = time.time()
    for param in theta_syms:
        grad = parameter_shift_gradient(circuit, param, 0.5, observable, data_values)
    
    backward_time = (time.time() - start) / len(theta_syms) * 1000  # ms per parameter
    
    print(f"Forward pass: {forward_time:.2f} ms/sample")
    print(f"Gradient computation: {backward_time:.2f} ms/parameter")
    
    # Memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.2f} MB")
    
    return {
        'forward_time_ms': forward_time,
        'backward_time_ms': backward_time,
        'memory_mb': memory_mb
    }

# Run benchmarks for all circuits
benchmarks = {}
for task in ['task1', 'task2', 'task3', 'task5']:
    print(f"\n{task} benchmarks:")
    benchmarks[task] = benchmark_circuit(circuits[task])
```

---

## WEEK-END DELIVERABLES

### Code
- `src/quantum/circuits.py` - All 5 circuit definitions
- `src/quantum/differentiators.py` - Gradient computation
- `tests/test_quantum_circuits.py` - Unit tests

### Reports
- `reports/quantum_circuit_report.pdf`:
  - Circuit diagrams
  - Parameter counts
  - Gradient verification results
  - Barren plateau analysis
  - Simulation benchmarks

---

## SUCCESS CRITERIA

✓ All 5 circuits simulate without errors  
✓ Gradient verification error <1e-6  
✓ No barren plateaus (gradient norm >1e-6)  
✓ Forward pass <100ms per batch (100 samples)  
✓ Backward pass <50ms per parameter  
✓ Circuit diagrams documented  
✓ Unit tests pass (>80% coverage)

---

**Next Phase:** [Phase 4: Hybrid Integration](./phase4_hybrid_integration.md)
