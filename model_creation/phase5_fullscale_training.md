# Phase 5: Full-Scale Training (Weeks 9-10)
## Quantum-Enhanced Cryptanalysis System

**Duration:** 2 weeks  
**Goal:** Train all 5 tasks on full dataset and document quantum contribution

---

## OBJECTIVES

1. Train all 5 hybrid models (100 epochs each)
2. Monitor training progress (TensorBoard)
3. Ablation study (Hybrid vs Classical-only)
4. Quantum contribution analysis
5. Save best models and checkpoints

---

## TRAINING SCHEDULE

### Week 9: Tasks 1-3
- Day 1-2: Task 1 (Attack Classification)
- Day 3-4: Task 2 (Attack Success Prediction)
- Day 5: Task 3 (Algorithm Identification)

### Week 10: Tasks 4-5 + Analysis
- Day 6-7: Task 4 (QAOA Plaintext Recovery)
- Day 8: Task 5 (Key Properties)
- Day 9-10: Ablation studies & analysis

---

## TRAINING PROCEDURES

### Task 1: Attack Classification

**Training Script:**
```python
# Load data
X train, y_train = load_processed_data('train')
X_val, y_val = load_processed_data('val')

# Load hybrid model
model = HybridAttackClassifier(circuit, qubits, n_classes=5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.F1Score(average='macro')]
)

# Training
history = model.fit(
    X_train, y_train['attack_category'],
    validation_data=(X_val, y_val['attack_category']),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)

# Save
model.save('models/hybrid/task1_final.h5')
```

**Monitor:** Loss, accuracy, F1, quantum gradient norms

**Target:** ≥88% test accuracy

---

### Task 2: Attack Success Prediction

```python
# Handle class imbalance
class_weights = compute_class_weight('balanced', classes=[0,1], y=y_train['attack_success'])

model_task2.fit(
    X_train, y_train['attack_success'],
    validation_data=(X_val, y_val['attack_success']),
    class_weight={0: class_weights[0], 1: class_weights[1]},
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
```

**Monitor:** AUC, F1, precision@90recall

**Target:** ≥0.85 AUC

---

### Task 4: QAOA Plaintext Recovery

**Special Training Loop:**
```python
from scipy.optimize import minimize

def qaoa_cost_function(params, circuit, data):
    """Evaluate QAOA cost for parameter optimization."""
    gamma, beta = params[:p], params[p:]
    
    # Run circuit with current params
    resolver = {f'gamma_{i}': gamma[i] for i in range(p)}
    resolver.update({f'beta_{i}': beta[i] for i in range(p)})
    
    # Measure bitstrings
    sampler = cirq.Simulator()
    result = sampler.run(circuit, repetitions=1000, param_resolver=resolver)
    
    # Evaluate cost
    cost = evaluate_hamiltonian(result, data)
    return cost

# Optimize for each sample
for epoch in range(n_epochs):
    for batch_x, batch_y in train_dataset:
        for sample_x, sample_y in zip(batch_x, batch_y):
            # Initial params
            initial_params = np.random.randn(2*p)
            
            # Optimize
            result = minimize(
                qaoa_cost_function,
                initial_params,
                args=(circuit, sample_x),
                method='COBYLA'
            )
            
            # Evaluate success
            recovered = decode_bitstring(result.x)
            success = (recovered == sample_y).mean() > 0.9
```

**Target:** ≥60% bitstring match rate

---

## ABLATION STUDIES

### Hybrid vs Classical-Only Comparison

**Create Classical-Only Baseline:**
```python
class ClassicalOnlyModel(tf.keras.Model):
    """Same architecture but WITHOUT quantum layer."""
    def __init__(self, n_classes=5):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        return self.output_layer(x)

# Train classical-only
classical_model = ClassicalOnlyModel(n_classes=5)
classical_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classical_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

# Compare
hybrid_acc = model.evaluate(X_val, y_val)[1]
classical_acc = classical_model.evaluate(X_val, y_val)[1]

quantum_contribution = (hybrid_acc - classical_acc) * 100
print(f"Quantum Contribution: +{quantum_contribution:.2f}%")
```

**Analysis Table:**
```
Task 1 Results:
- Classical Baseline (XGBoost): 86.5%
- Classical-Only NN: 84.2%
- Hybrid Quantum-Classical: 88.3%
- Quantum Contribution: +1.8%
```

---

## MONITORING & CHECKPOINTS

### TensorBoard Visualization
```bash
tensorboard --logdir results/tensorboard_logs
```

**View:**
- Training/validation loss curves
- Accuracy metrics
- Quantum gradient histograms
- Learning rate schedules

### Checkpoint Strategy
```python
# Save every 5 epochs
checkpoint_callback = ModelCheckpoint(
    filepath='models/checkpoints/task1_epoch{epoch:02d}_val{val_accuracy:.4f}.h5',
    save_freq='epoch',
    period=5
)
```

---

## WEEK-END DELIVERABLES

### Trained Models
- `models/hybrid/task1_final.h5`
- `models/hybrid/task2_final.h5`
- `models/hybrid/task3_final.h5`
- `models/hybrid/task4_qaoa.pkl`
- `models/hybrid/task5_final.h5`

### Training Logs
- `results/training_logs/task{1-5}_history.csv`
- `results/tensorboard_logs/` (TensorBoard files)

### Reports
- `reports/training_results.md` (comprehensive analysis)

---

## SUCCESS CRITERIA

✓ Task 1: ≥88% test accuracy  
✓ Task 2: ≥0.85 test AUC  
✓ Task 3: ≥87% test accuracy  
✓ Task 4: ≥60% match rate  
✓ Task 5: ≤8-bit MAE  
✓ Quantum contribution documented (+1-3%)  
✓ All models saved

---

**Next Phase:** [Phase 6: Evaluation & Deployment](./phase6_evaluation_deployment.md)
