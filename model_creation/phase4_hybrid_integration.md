# Phase 4: Hybrid Integration (Weeks 7-8)
## Quantum-Enhanced Cryptanalysis System

**Duration:** 2 weeks  
**Goal:** Integrate quantum circuits with TensorFlow/Keras for end-to-end training

---

## OBJECTIVES

1. Implement HybridModel classes (TFQ + Keras)
2. Verify end-to-end differentiability
3. Test on small dataset (100 samples)
4. Configure training loops with callbacks
5. Tune learning rates (quantum vs classical)

---

## INPUT SPECIFICATIONS

**Circuits:** From Phase 3  
**Data:** Processed datasets from Phase 1

---

## OUTPUT SPECIFICATIONS

### Hybrid Models
```
src/quantum/hybrid_model.py:
- HybridAttackClassifier (Task 1)
- HybridAttackSuccessPredictor (Task 2)
- HybridAlgorithmIdentifier (Task 3)
- QAOAPlaintextRecovery (Task 4)
- HybridKeyPropertiesPredictor (Task 5)
```

### Configuration
```yaml
configs/training_config.yaml:
task: attack_classification
model_type: hybrid

data:
  train_samples: 30000
  val_samples: 10000
  
training:
  epochs: 100
  batch_size: 32
  learning_rate_classical: 0.01
  learning_rate_quantum: 0.005
  early_stopping_patience: 10
  
architecture:
  n_qubits: 8
  n_layers: 4
  dense_layers: [32, 16]
  dropout_rate: 0.3
```

---

## WEEK 7: TFQ-KERAS INTEGRATION

### Day 1-2: Task 1 Hybrid Model

**Implementation:**
```python
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq

class HybridAttackClassifier(tf.keras.Model):
    """
    Hybrid Quantum-Classical Model for Attack Classification.
    
    Architecture:
    1. Input features → Quantum preprocessing (PQC)
    2. Quantum expectation values → Classical dense layers
    3. Final classification head
    """
    
    def __init__(self, circuit, qubits, n_classes=5, classical_dense=[32, 16]):
        super().__init__()
        
        self.circuit = circuit
        self.qubits = qubits
        
        # Define observables for measurement
        self.observables = [
            cirq.Z(qubits[0]),
            cirq.Z(qubits[1]),
            cirq.Z(qubits[2])
        ]
        
        # Quantum layer (Parameterized Quantum Circuit)
        self.pqc_layer = tfq.layers.PQC(
            circuit=circuit,
            operators=self.observables,
            differentiator=tfq.differentiators.ParameterShift(),
            initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi)
        )
        
        # Classical post-processing layers
        self.dense_layers = []
        for units in classical_dense:
            self.dense_layers.append(tf.keras.layers.Dense(units, activation='relu'))
            self.dense_layers.append(tf.keras.layers.Dropout(0.3))
            self.dense_layers.append(tf.keras.layers.BatchNormalization())
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Dictionary with:
                - 'features': Classical features (batch, n_features)
                - 'circuit_input': Prepared quantum circuit inputs
        """
        # Extract features
        features = inputs if isinstance(inputs, tf.Tensor) else inputs['features']
        
        # Prepare quantum circuit inputs (convert to circuit tensor)
        batch_size = tf.shape(features)[0]
        circuit_batch = tfq.convert_to_tensor([self.circuit] * batch_size)
        
        # Quantum preprocessing
        # Output shape: (batch_size, n_observables)
        quantum_output = self.pqc_layer([circuit_batch, features])
        
        # Classical post-processing
        x = quantum_output
        for layer in self.dense_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Final output
        output = self.output_layer(x)
        
        return output

# Instantiate model
from src.quantum.circuits import create_attack_classification_circuit

circuit, data_syms, param_syms = create_attack_classification_circuit(n_qubits=8, n_layers=4)
qubits = cirq.LineQubit.range(8)

model = HybridAttackClassifier(
    circuit=circuit,
    qubits=qubits,
    n_classes=5,
    classical_dense=[32, 16]
)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.F1Score(average='macro')]
)
```

**Deliverable:** `src/quantum/hybrid_model.py` (Task 1)

---

### Day 3: Test Training Loop

**Small Dataset Test:**
```python
# Create tiny dataset for testing
X_train_tiny = X_train[:100]  # 100 samples
y_train_tiny = y_train['attack_category'][:100]
X_val_tiny = X_val[:20]
y_val_tiny = y_val['attack_category'][:20]

# Test forward pass
try:
    predictions = model(X_train_tiny[:5])
    print(f"✅ Forward pass successful. Output shape: {predictions.shape}")
    assert predictions.shape == (5, 5), "Output shape mismatch!"
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    raise

# Test training for 5 epochs
history = model.fit(
    X_train_tiny, y_train_tiny,
    validation_data=(X_val_tiny, y_val_tiny),
    epochs=5,
    batch_size=16,
    verbose=2
)

# Verify loss decreases
assert history.history['loss'][0] > history.history['loss'][-1], "Loss not decreasing!"
print("✅ Loss decreased during training")

# Verify gradients flow
print("\n✅ All  tests passed! Model ready for full-scale training.")
```

---

### Day 4-5: Implement Remaining Tasks

**Task 2: Binary Classification**
```python
class HybridAttackSuccessPredictor(tf.keras.Model):
    def __init__(self, circuit, qubits):
        super().__init__()
        
        # Single observable for binary output
        self.observable = [cirq.Z(qubits[0])]
        
        self.pqc_layer = tfq.layers.PQC(
            circuit=circuit,
            operators=self.observable,
            differentiator=tfq.differentiators.ParameterShift()
        )
        
        self.rescale = tf.keras.layers.Lambda(lambda x: (x + 1) / 2)  # [-1,1] → [0,1]
        self.dense = tf.keras.layers.Dense(16, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        circuit_batch = tfq.convert_to_tensor([self.circuit] * batch_size)
        
        q_out = self.pqc_layer([circuit_batch, inputs])
        q_out = self.rescale(q_out)
        x = self.dense(q_out)
        return self.output_layer(x)

# Compile with binary crossentropy
model_task2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
```

**Task 5: Multi-Task Learning**
```python
class HybridKeyPropertiesPredictor(tf.keras.Model):
    """Predicts both key entropy (regression) and weak key probability (classification)."""
    
    def __init__(self, circuit, qubits):
        super().__init__()
        
        self.observables = [
            cirq.Z(qubits[0]),
            cirq.Z(qubits[1]),
            cirq.X(qubits[2]),
            cirq.Y(qubits[3])
        ]
        
        self.pqc_layer = tfq.layers.PQC(circuit, operators=self.observables)
        
        # Shared dense layers
        self.dense_shared = tf.keras.layers.Dense(32, activation='relu')
        
        # Task-specific heads
        self.entropy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear', name='entropy_output')
        ])
        
        self.weak_key_head = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='weak_key_output')
        ])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        circuit_batch = tfq.convert_to_tensor([self.circuit] * batch_size)
        
        q_out = self.pqc_layer([circuit_batch, inputs])
        shared = self.dense_shared(q_out)
        
        # Dual outputs
        entropy = self.entropy_head(shared)
        weak_key = self.weak_key_head(shared)
        
        return {'entropy': entropy, 'weak_key': weak_key}

# Compile with weighted loss
model_task5.compile(
    optimizer='adam',
    loss={
        'entropy': 'mse',
        'weak_key': 'binary_crossentropy'
    },
    loss_weights={'entropy': 0.5, 'weak_key': 0.5},
    metrics={
        'entropy': 'mae',
        'weak_key': 'accuracy'
    }
)
```

---

## WEEK 8: TRAINING PREPARATION

### Day 6-7: Callbacks & Logging

**Training Configuration:**
```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard,
    ReduceLROnPlateau, CSVLogger
)

# Callbacks
callbacks = [
    # Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        filepath='models/hybrid/task1_best.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    # TensorBoard logging
    TensorBoard(
        log_dir='results/tensorboard_logs/task1',
        histogram_freq=1,
        write_graph=True
    ),
    
    # Reduce learning rate on plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    
    # CSV logging
    CSVLogger('results/training_logs/task1_history.csv')
]
```

---

### Day 8-9: Learning Rate Strategy

**Separate LR for Quantum & Classical:**
```python
# Option 1: Different optimizers (not recommended)
# quantum_optimizer = tf.keras.optimizers.Adam(lr=0.005)
# classical_optimizer = tf.keras.optimizers.Adam(lr=0.01)

# Option 2: Custom training loop with separate LR (recommended)
class HybridTrainer:
    def __init__(self, model, lr_quantum=0.005, lr_classical=0.01):
        self.model = model
        self.quantum_vars = [v for v in model.trainable_variables if 'pqc' in v.name]
        self.classical_vars = [v for v in model.trainable_variables if 'pqc' not in v.name]
        
        self.quantum_optimizer = tf.keras.optimizers.Adam(lr_quantum)
        self.classical_optimizer = tf.keras.optimizers.Adam(lr_classical)
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
            loss = tf.reduce_mean(loss)
        
        # Separate gradients
        grads = tape.gradient(loss, self.quantum_vars + self.classical_vars)
        quantum_grads = grads[:len(self.quantum_vars)]
        classical_grads = grads[len(self.quantum_vars):]
        
        # Apply separate optimizers
        self.quantum_optimizer.apply_gradients(zip(quantum_grads, self.quantum_vars))
        self.classical_optimizer.apply_gradients(zip(classical_grads, self.classical_vars))
        
        return loss

# Usage
trainer = HybridTrainer(model, lr_quantum=0.005, lr_classical=0.01)

for epoch in range(num_epochs):
    for batch_x, batch_y in train_dataset:
        loss = trainer.train_step(batch_x, batch_y)
    # Validation...
```

---

### Day 10: Integration Tests

**Test Suite:**
```python
# tests/test_hybrid_integration.py

def test_model_compilation():
    """Test that model compiles without errors."""
    model = HybridAttackClassifier(circuit, qubits, n_classes=5)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    assert model is not None

def test_forward_pass():
    """Test forward pass with random data."""
    X_test = np.random.randn(10, 65).astype(np.float32)
    predictions = model(X_test)
    assert predictions.shape == (10, 5)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)

def test_gradient_flow():
    """Test that gradients flow through both quantum and classical layers."""
    X_test = np.random.randn(5, 65).astype(np.float32)
    y_test = np.random.randint(0, 5, 5)
    
    with tf.GradientTape() as tape:
        predictions = model(X_test)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_test, predictions)
    
    grads = tape.gradient(loss, model.trainable_variables)
    
    # Check that gradients are not None
    for i, grad in enumerate(grads):
        assert grad is not None, f"Gradient {i} is None!"
    
    # Check quantum gradients
    quantum_grads = [g for g in grads if 'pqc' in g.name]
    for grad in quantum_grads:
        assert tf.reduce_mean(tf.abs(grad)) > 1e-8, "Quantum gradients vanishing!"

def test_training_improves_loss():
    """Test that training reduces loss."""
    X_tiny = np.random.randn(50, 65).astype(np.float32)
    y_tiny = np.random.randint(0, 5, 50)
    
    initial_loss = model.evaluate(X_tiny, y_tiny, verbose=0)[0]
    
    model.fit(X_tiny, y_tiny, epochs=5, verbose=0)
    
    final_loss = model.evaluate(X_tiny, y_tiny, verbose=0)[0]
    
    assert final_loss < initial_loss, "Loss did not decrease!"

# Run tests
pytest tests/test_hybrid_integration.py -v
```

---

## WEEK-END DELIVERABLES

### Code
- `src/quantum/hybrid_model.py` - All 5 hybrid model classes
- `src/training/trainer.py` - Custom training loop
- `src/training/callbacks.py` - Custom callbacks
- `tests/test_hybrid_integration.py` - Integration tests

### Configuration
- `configs/training_config.yaml` - Full training config

---

## SUCCESS CRITERIA

✓ All 5 hybrid models compile without errors  
✓ Forward pass works (correct shapes)  
✓ Loss decreases on toy dataset  
✓ Gradients flow through quantum + classical layers  
✓ Integration tests pass (100%)  
✓ Training callbacks configured  
✓ Separate LR strategy implemented

---

**Next Phase:** [Phase 5: Full-Scale Training](./phase5_fullscale_training.md)
