# PQC Installation Guide for Windows

## Overview

This guide explains how to enable Post-Quantum Cryptography (PQC) support on Windows by installing CMake, building liboqs, and linking it with liboqs-python.

## Prerequisites

- Windows 10/11
- Python 3.8+
- Visual Studio 2019/2022 with C++ Desktop Development workload
- CMake 3.18+ (download from https://cmake.org/download/)
- Git for Windows

## Step 1: Install CMake

### Option A: Installer (Recommended)
1. Download CMake from https://cmake.org/download/
2. Run installer: `cmake-3.x.x-windows-x86_64.msi`
3. During installation, select "Add CMake to system PATH for all users"
4. Verify: Open new Command Prompt and run `cmake --version`

### Option B: Chocolatey
```powershell
choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System'
```

## Step 2: Clone and Build liboqs

```powershell
# Create install directory
mkdir C:\Users\%USERNAME%\_oqs
cd C:\Users\%USERNAME%\_oqs

# Clone liboqs (version matching liboqs-python 0.14.1)
git clone --branch 0.11.0 --depth 1 https://github.com/open-quantum-safe/liboqs.git

# Configure build
cmake -S liboqs -B liboqs/build `
  -DBUILD_SHARED_LIBS=ON `
  -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE `
  -DCMAKE_INSTALL_PREFIX="C:\Users\%USERNAME%\_oqs" `
  -A x64

# Build (use --parallel to speed up, adjust number to CPU cores)
cmake --build liboqs/build --config Release --parallel 8

# Install
cmake --build liboqs/build --target install --config Release
```

## Step 3: Set Environment Variables

Add the DLL path to your system PATH:

### Option A: GUI
1. Open "Edit the system environment variables"
2. Click "Environment Variables"
3. Under "System variables", find "Path"
4. Click "Edit" → "New"
5. Add: `C:\Users\%USERNAME%\_oqs\bin`
6. Click OK on all dialogs

### Option B: PowerShell (Run as Administrator)
```powershell
[Environment]::SetEnvironmentVariable(
    "Path",
    $env:Path + ";C:\Users\$env:USERNAME\_oqs\bin",
    [EnvironmentVariableTarget]::Machine
)
```

## Step 4: Install liboqs-python

```powershell
# Install liboqs-python (already installed via requirements.txt)
pip install liboqs-python

# Verify installation
python -c "import oqs; print(f'✅ liboqs version: {oqs.oqs_version()}')"
```

## Step 5: Verify PQC Algorithms

```powershell
cd dataset_generation
python main.py --list-algorithms
```

You should now see 111+ algorithms including:
- ML-KEM-512, ML-KEM-768, ML-KEM-1024
- CRYSTALS-Kyber variants
- NTRU variants
- BIKE variants
- ML-DSA, Falcon, SPHINCS+ (signatures)

## Troubleshooting

### "No oqs shared libraries found"
- Ensure `C:\Users\%USERNAME%\_oqs\bin\oqs.dll` exists
- Verify DLL path is in system PATH
- Restart Command Prompt/PowerShell after setting PATH

### CMake not found
- Download and install CMake from https://cmake.org/download/
- Make sure to add to PATH during installation
- Restart terminal after installation

### Visual Studio Build Tools missing
- Download from: https://visualstudio.microsoft.com/downloads/
- Select "Desktop development with C++"
- Install and restart

### Build fails with "NMake Makefiles" error
- Use `-G "Visual Studio 17 2022"` or `-G "Visual Studio 16 2019"` in cmake command
- Or install Ninja build system: `choco install ninja`

## Verification Tests

```powershell
# Test 1: Library import
python -c "import oqs; print(oqs.oqs_version()); print(f'KEMs: {len(oqs.get_enabled_kem_mechanisms())}')"

# Test 2: ML-KEM encryption
python -c "from src.crypto_dataset_generator.crypto.cipher_factory import create_cipher; c = create_cipher('ML-KEM-768'); c.generate_key(); ct, _ = c.encrypt(b'test'); print('✅ ML-KEM works')"

# Test 3: Generate PQC dataset
python main.py --algorithms ML-KEM-768,NTRU-HPS-2048-509 --samples 3
```

## Alternative: Use WSL or Linux VM

If Windows native build is problematic:
1. Install WSL2 (Windows Subsystem for Linux)
2. Use Ubuntu in WSL2
3. Install liboqs with `apt-get install liboqs-dev` (if available) or build from source
4. Run dataset generation in WSL2 environment

## Support

For issues:
- liboqs: https://github.com/open-quantum-safe/liboqs/issues
- liboqs-python: https://github.com/open-quantum-safe/liboqs-python/issues
