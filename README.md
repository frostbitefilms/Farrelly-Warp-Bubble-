# Alcubierre Warp Simulation

This repository contains data and code for the study "Quantum Simulation of an Energy-Efficient Alcubierre Warp Bubble at Astrophysical Scales" by Jamie Farrelly, submitted to *Physical Review D* on August 8, 2025. The simulation achieves unprecedented energy efficiency for an Alcubierre warp bubble at \( R = 100,000,000 \, \text{m} \), with 100% stability, surpassing prior models by Alcubierre (1994), White (2021), Lentz (2021), and Fuchs et al. (2024).

## Contents

- `thinking_like_miguel.py`: Python script implementing a 6-qubit variational quantum circuit in PennyLane, encoding the Alcubierre metric tensor and a Casimir-like negative energy model.
- `enhanced_simulation_log_100000000m.txt`: Simulation results, including expectation values, energy density, total energy, and stability metrics.

## Simulation Overview

The simulation models an Alcubierre warp bubble at \( R = 100,000,000 \, \text{m} \), \( v_s = 0.1c \), achieving:
- Expectation value: \( -9.80 \times 10^{-6} \)
- Energy density: \( -1.24 \times 10^{-19} \, \text{J/m}^3 \)
- Total energy: \( -5.20 \times 10^5 \, \text{J} \)
- Stability: 100% (ratio = 0.0000)

Results are scalable from 5 m to 100,000,000 m and potentially align with UAP observations (e.g., Chester, NY, 2025).

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install pennylane numpy scipy
   ```

2. **Clone Repository**:
   ```bash
   git clone https://github.com/SmallChineseMan/AlcubierreWarpSimulation.git
   cd AlcubierreWarpSimulation
   ```

3. **Verify Files**:
   Ensure `thinking_like_miguel.py` and `enhanced_simulation_log_100000000m.txt` are present.

## Usage

Run the simulation to generate results:
```bash
python thinking_like_miguel.py
```

The script outputs results to the console and saves them to `enhanced_simulation_log_100000000m.txt` in the specified directory (e.g., `c:\Users\jamie\Downloads` on Windows). Expected output:
```
Enhanced Alcubierre Simulation (R=1.0e+08 m, v_s=0.10c)
t=0.00e+00 s: exp_val=-9.80e-06, ρ=-1.24e-19 J/m³, E=-5.20e+05 J
...
Stability Analysis: Ratio=0.0010, Stable=Yes
Results logged to c:\Users\jamie\Downloads\enhanced_simulation_log_100000000m.txt
```

## Citation

Please cite the manuscript if using this code or data:
> Farrelly, J. (2025). Quantum Simulation of an Energy-Efficient Alcubierre Warp Bubble at Astrophysical Scales. *Physical Review D* (under review).

## Contact

For questions or additional data, contact:
- Jamie Farrelly (contact@aaro.mil)
- X: @X_mayafrost_X

## License

This work is licensed under [MIT License](LICENSE). Data are timestamped with OriginStamp for intellectual property protection.
