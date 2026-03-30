# NIAMH-MPCD OpenMP Parallelization Plan

## Status: COMPLETE (all planned parallelizations implemented)

## Pre-existing State
- Makefile already has `-fopenmp` flags
- RNG already thread-safe (per-thread state arrays in rand.c)
- `stream_all()` and `acc_all()` already parallelized with `#pragma omp parallel for`

## Tasks

### 1. Grid Shift Loop (mpc.c:734) - PARTICLE LOOP
- `gridShift_all()` - parallelize the SRD particle loop `for(i=0;i<GPOP;i++)`
- Each particle is independent; `signedSHIFT[]` is read-only
- `#pragma omp parallel for schedule(static) private(j)`

### 2. localPROP Cell Loop (mpc.c:86) - CELL LOOP
- Triple-nested loop over all cells, each cell writes only to its own data
- Linked list traversal is read-only (no structural modification)
- `#pragma omp parallel for collapse(3) private(a,b,c,d,id,mass,pMPC,pMD,pSW,V,Q)`
- Also parallelize the moment of inertia loop (line 175) and order param loop (line 186)
- NOTE: order param loop allocates S matrix per-iteration, needs private copy

### 3. LC Collision Loop (mpc.c:4697) - CELL LOOP
- Each cell is independent; SP and in are read-only
- `zeroMFPot` is set per-iteration, make private
- `#pragma omp parallel for collapse(3) private(zeroMFPot)` on the outer i,j,k

### 4. Magnetic Torque Loop (lc.c:889) - PARTICLE LOOP
- `magTorque_all()` - independent per-particle operations
- `#pragma omp parallel for schedule(static)`

### 5. Orientation Shear Alignment Loop (mpc.c:4719) - CELL LOOP
- `jefferysTorque()` per cell, each cell independent
- `#pragma omp parallel for collapse(3)`

### 6. Velocity Collision Loop (mpc.c:4750) - CELL LOOP [BIGGEST WIN]
- Core computational kernel, called every timestep
- Issues:
  - `CLQ` array is shared - make `private(CLQ)`
  - `SP->ACT` mutation when `MFPLAYERH > 0` causes data race
- Solution: Split into two paths:
  - When MFPLAYERH == 0: `#pragma omp parallel for collapse(3) private(CLQ)`
  - When MFPLAYERH > 0: loop j serially (set SP->ACT), parallelize i,k with collapse(2)

### 7. Multiphase Collision Loop (mpc.c:4769) - CELL LOOP
- Each cell independent; needs CLQ set per-cell (currently stale from previous loop)
- `#pragma omp parallel for collapse(3) private(CLQ)`

### 8. Incompressibility Correction Loop (mpc.c:4773) - CELL LOOP
- Same as multiphase
- `#pragma omp parallel for collapse(3) private(CLQ)`

### 9. Orientation-Velocity Loops (mpc.c:4633, 4835) - PARTICLE LOOP
- Subtract/add orientation from velocity
- `#pragma omp parallel for schedule(static)`

### 10. TEMP Calculation (therm.c:214) - PARTICLE LOOP WITH REDUCTION
- `#pragma omp parallel for reduction(+:KBT) private(temp)`
- Only the GPOP loop (wall loop is small, leave serial)

### 11. Average Velocity Loop (therm.c:365) - CELL LOOP WITH REDUCTION
- Need reduction on AVVEL[d] array
- Use manual reduction or separate accumulation

## NOT Parallelized (intentionally)
- **Binning** (mpc.c:933, 960): Linked-list pointer manipulation has data races
- **Scramble** (mpc.c:4138): Random swaps between particles create data races
- **BC collision** (mpc.c:4885): Writes to shared WALL[].dV/dL from multiple particles
- **Ghost particles, swimmer dipole, swimmer integration**: Complex shared state
