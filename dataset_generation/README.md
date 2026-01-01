# Dataset Generation

This folder contains all the scripts and data for generating cryptographic datasets and attack simulations.

## Contents

### Python Scripts
- **`main.py`** - Main cryptographic dataset generator
- **`generate_attack_dataset.py`** - Attack simulation engine
- **`complete_pipeline.py`** - Automated pipeline orchestrator
- **`run_complete_pipeline.py`** - Complete pipeline runner with logging

### Source Code
- **`src/`** - Source code modules
  - `crypto_dataset_generator/` - Core cryptographic implementation modules

### Generated Data
- **`crypto_dataset.csv`** - Cryptographic algorithm dataset (510 samples)
- **`attack_dataset.csv`** - Attack simulation results (137,700 rows)
- **`attack_metadata.csv`** - Attack metadata
- **`Cryptographic_Algorithm_Summary.csv`** - Algorithm security summary
- **`attack_dataset.summary.json`** - Dataset statistics

### Output
- **`output/`** - Generated output files and reports

## Usage

Navigate to this directory and run:

```bash
# Generate crypto dataset
python main.py --samples 10

# Generate attack dataset
python generate_attack_dataset.py

# Run complete pipeline
python run_complete_pipeline.py
```

For more details, see the main [README](../README.md) and [USAGE_MANUAL](../USAGE_MANUAL.md).
