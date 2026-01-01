"""
TensorFlow Quantum Implementation Guide
Complete guide for training quantum hybrid models.

REQUIRES: tensorflow-quantum, cirq-google

Author: AIRAWAT Team
Date: 2026-01-01
"""

import numpy as np
import pandas as pd
import logging

# Check for TFQ availability
try:
    import tensorflow as tf
    import tensorflow_quantum as tfq
    import cirq
    import sympy
    TFQ_AVAILABLE = True
except ImportError:
    TFQ_AVAILABLE = False
    print("WARNING: TensorFlow Quantum not installed!")
    print("Install with: pip install tensorflow-quantum cirq-google")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumHybridModel:
    """Hybrid Quantum-Classical ML model for cryptanalysis."""
    
    def __init__(self, n_qubits=8, n_layers=4, n_classes=2):
        """
        Initialize hybrid model.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            n_classes: Number of output classes
        """
        if not TFQ_AVAILABLE:
            raise ImportError("TensorFlow Quantum not available!")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        # Create quantum circuit
        self.circuit, self.observables = self._create_circuit()
        
        # Build model
        self.model = self._build_hybrid_model()
        
        logger.info(f"Quantum Hybrid Model created:")
        logger.info(f"  Qubits: {n_qubits}")
        logger.info(f"  Layers: {n_layers}")
        logger.info(f"  Parameters: {n_qubits * n_layers}")
        logger.info(f"  Classes: {n_classes}")
    
    def _create_circuit(self):
        """Create parameterized quantum circuit."""
        qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()
        
        # Data encoding symbols
        data_symbols = sympy.symbols(f'x0:{self.n_qubits}')
        
        # Data encoding layer
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(data_symbols[i])(qubit))
        
        # Variational layers
        param_symbols = []
        for layer in range(self.n_layers):
            layer_symbols = sympy.symbols(f'theta_{layer}_0:{self.n_qubits}')
            param_symbols.extend(layer_symbols)
            
            # Rotations
            for i, qubit in enumerate(qubits):
                circuit.append(cirq.ry(layer_symbols[i])(qubit))
            
            # Entanglement (ring)
            for i in range(self.n_qubits - 1):
                circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
            circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
        
        # Observables
        observables = [cirq.Z(qubits[i]) for i in range(min(3, self.n_qubits))]
        
        return circuit, observables
    
    def _build_hybrid_model(self):
        """Build hybrid quantum-classical model."""
        # Input: Classical features
        inputs = tf.keras.Input(shape=(self.n_qubits,), dtype=tf.float32)
        
        # Quantum layer (PQC)
        quantum_layer = tfq.layers.PQC(
            self.circuit,
            self.observables,
            differentiator=tfq.differentiators.ParameterShift(),
            initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi)
        )
        
        # Pass through quantum circuit
        quantum_output = quantum_layer(inputs)
        
        # Classical post-processing
        dense1 = tf.keras.layers.Dense(32, activation='relu')(quantum_output)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(16, activation='relu')(dropout1)
        
        # Output layer
        if self.n_classes == 2:
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC()]
        else:
            outputs = tf.keras.layers.Dense(self.n_classes, activation='softmax')(dense2)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train hybrid model.
        
        Args:
            X_train: Training features (n_samples, n_qubits)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
        """
        logger.info(f"\nTraining hybrid model...")
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Val: {X_val.shape}")
        
        # Ensure correct dtype
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✓ Training complete")
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model."""
        X_test = X_test.astype(np.float32)
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"\nTest Results:")
        logger.info(f"  Loss: {results[0]:.4f}")
        logger.info(f"  Accuracy: {results[1]:.4f}")
        
        return results
    
    def save(self, filepath='models/hybrid/quantum_hybrid_model.h5'):
        """Save model."""
        self.model.save(filepath)
        logger.info(f"✓ Model saved to {filepath}")


def example_tfq_training():
    """Example: Train TFQ hybrid model."""
    if not TFQ_AVAILABLE:
        print("\nTensorFlow Quantum not available!")
        print("Install with: pip install tensorflow-quantum cirq-google")
        return
    
    print("\n" + "="*80)
    print("TENSORFLOW QUANTUM - HYBRID MODEL TRAINING")
    print("="*80)
    
    # Load data
    logger.info("Loading attack dataset...")
    df = pd.read_csv('../dataset_generation/attack_dataset.csv')
    
    # Prepare features (use first 8 for quantum circuit)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['attack_execution_id', 'encryption_row_id', 'attack_id']
    numeric_features = [col for col in numeric_features if col not in exclude]
    
    # Select first 8 features for quantum encoding
    X = df[numeric_features[:8]].fillna(0).values
    y = df['attack_success'].astype(int).values
    
    # Normalize to [0, 2π] for quantum encoding
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
    X_scaled = scaler.fit_transform(X)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data prepared:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val: {X_val.shape}")
    logger.info(f"  Test: {X_test.shape}")
    
    # Create and train model
    model = QuantumHybridModel(n_qubits=8, n_layers=4, n_classes=2)
    
    # Train (use small subset for demo)
    history = model.train(
        X_train[:1000], y_train[:1000],
        X_val[:200], y_val[:200],
        epochs=20,
        batch_size=32
    )
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    
    # Save
    model.save()
    
    print("\n✓ TFQ training complete!")


if __name__ == "__main__":
    example_tfq_training()
