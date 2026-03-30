# CLAUDE.md — AI Assistant Guide for 2D-VS-Solver

## Project Overview

**2D NS Solver** is a real-time 2D incompressible Navier-Stokes fluid simulator with interactive SFML visualization. It implements Jos Stam's "Stable Fluids" algorithm on a uniform Eulerian grid, supporting arbitrary obstacle shapes (circles, NACA airfoils), mouse-driven dye injection, and live pressure/velocity field rendering.

- **Language**: C++20
- **Build system**: CMake 3.20+
- **Key dependencies**: Eigen 3 (linear algebra / pressure solver), SFML 3 (rendering & windowing)
- **Version**: 0.1.0
- **Single author project** — all 8 commits by Christian Rojas (March 2026)

---

## Repository Structure

```
2D-VS-Solver/
├── CMakeLists.txt           # Build configuration — three targets defined here
├── README.md                # User-facing project overview and build instructions
├── Agent.md                 # Technical specification and implementation roadmap
├── include/
│   └── FluidField.hpp       # Public API — FluidField class declaration (62 lines)
├── src/
│   ├── FluidField.cpp       # Core physics implementation (396 lines)
│   └── main.cpp             # SFML application, visualization, user input (198 lines)
├── tests/
│   └── FluidFieldTests.cpp  # Unit tests with custom assertion framework (70 lines)
└── assets/
    └── demo-NS.gif          # Animated demo screenshot for README
```

---

## Build System

### CMake Targets

| Target | Type | Description |
|--------|------|-------------|
| `ns_solver_core` | Static library | FluidField physics (no SFML dependency) |
| `ns_solver_app` | Executable | Links `ns_solver_core` + SFML for the live demo |
| `ns_solver_tests` | Executable | Links `ns_solver_core`; registered with ctest |

### Building

```bash
cmake -B build
cmake --build build
```

Optional: enable warnings-as-errors:
```bash
cmake -B build -DNS_SOLVER_WARNINGS_AS_ERRORS=ON
cmake --build build
```

### Running

```bash
./build/ns_solver_app      # live visualization
ctest --test-dir build --output-on-failure   # run tests
```

---

## Architecture

### FluidField — Core Physics (include/FluidField.hpp, src/FluidField.cpp)

`FluidField` owns all simulation state as a flat collection of `std::vector<float>` grids (row-major, `nx * ny` elements):

| Member | Purpose |
|--------|---------|
| `density_` / `density0_` | Current / scratch dye scalar field |
| `vx_` / `vx0_` | Horizontal velocity (current / scratch) |
| `vy_` / `vy0_` | Vertical velocity (current / scratch) |
| `pressure_` / `pressure0_` | Pressure field (current / scratch) |
| `obstacle_` | `uint8_t` mask: 0 = fluid, 1 = solid |

**Simulation step order** (called once per frame):
1. Apply continuous left-edge inflow (fixed velocity `1.75`, density `1.0`)
2. Enforce obstacle no-slip (zero velocity/density inside solids)
3. `diffuse()` — Gauss-Seidel relaxation (12 iterations) for viscosity
4. `project()` — Poisson pressure solve via Eigen sparse solver for incompressibility
5. `advect()` — Semi-Lagrangian backtracking with bilinear interpolation
6. `project()` — Second projection after advection
7. Density dissipation (`× 0.997` per step)

**Hardcoded simulation constants** (in source — no config file):

| Constant | Value | Location |
|----------|-------|----------|
| Grid size | 128 × 128 | `main.cpp` |
| Window size | 768 × 768 px | `main.cpp` |
| Diffusion coefficient | `0.0001` | `FluidField.cpp` |
| Viscosity | `0.00001` | `FluidField.cpp` |
| Delta time (dt) | `0.1` s | `main.cpp` |
| Density dissipation | `0.997` | `FluidField.cpp` |
| Inflow velocity | `1.75` | `FluidField.cpp` |
| Solver iterations | `12` | `FluidField.cpp` |

**Public API surface:**

```cpp
FluidField(int nx, int ny);
void step(float dt);
void addDensity(int x, int y, float amount);
void addVelocity(int x, int y, float vx, float vy);
void setObstacleCircle(float cx, float cy, float r);
void setObstacleAirfoil(float cx, float cy, float chord, float thickness, float angleDeg);

// Accessors (return const ref to internal vectors)
const std::vector<float>& density() const;
const std::vector<float>& pressure() const;
const std::vector<float>& vx() const;
const std::vector<float>& vy() const;
const std::vector<uint8_t>& obstacle() const;
int nx() const;
int ny() const;
```

**`BoundaryMode` enum** — used internally by `setBoundary()`:
- `Scalar` — zero-gradient at walls
- `HorizontalVelocity` — negate at vertical walls
- `VerticalVelocity` — negate at horizontal walls

### main.cpp — Visualization & Input

- Creates a `128×128` `FluidField` and `768×768` SFML window.
- Runs a fixed 60 FPS event loop; calls `field.step(0.1f)` each frame.
- Renders an `sf::Image` pixel-by-pixel, one pixel per grid cell, upscaled to window.
- **Visualization modes** (keyboard `1`/`2`/`3`):
  1. Density — grayscale smoke
  2. Velocity magnitude — heat colormap
  3. Pressure — cool/warm colormap
- **Velocity glyphs** — arrow lines drawn every 6 cells, always overlaid.
- **Mouse left-click/drag** — injects density + upward velocity at cursor position.
- **`[` / `]` keys** — decrease/increase airfoil angle of attack (clamped ±20°); rebuilds obstacle mask each change.

---

## Testing

### Framework

No third-party test library — tests use a hand-rolled `expect()` macro:

```cpp
#define expect(cond, msg) \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; return false; }
```

Each test function returns `bool`; `main()` aggregates pass/fail and exits `1` on any failure.

### Test Cases (tests/FluidFieldTests.cpp)

| Function | What it validates |
|----------|-------------------|
| `testInflowSeedsVelocityAndDensity()` | Left-edge inflow sets non-zero vx & density after one step (32×32 grid) |
| `testObstacleRejectsInjectedState()` | Cells inside an obstacle remain zero after `addDensity`/`addVelocity` + `step()` (32×32) |
| `testStepKeepsFieldsFinite()` | No NaN/Inf appears in any field after 12 steps on a 48×48 grid |

### Running Tests

```bash
ctest --test-dir build --output-on-failure
# or directly:
./build/ns_solver_tests
```

Tests use smaller grids (32–48 cells) than the production app (128×128) for speed.

---

## Key Conventions

### Code Style

- C++20, no third-party headers beyond Eigen and SFML.
- No `.clang-format` file — match the surrounding style manually:
  - 4-space indentation (spaces, not tabs)
  - Snake_case for member variables with trailing underscore (`density_`, `vx0_`)
  - PascalCase for types (`FluidField`, `BoundaryMode`)
  - camelCase for methods (`addDensity`, `setBoundary`)
- Compiler warnings (`-Wall -Wextra -Wpedantic`) are enabled. New code must compile without warnings.
- **No external test framework** — extend `FluidFieldTests.cpp` using the existing `expect()` pattern.

### Physics / Algorithm Conventions

- Grid indexing: `idx(i, j) = i + j * nx` (x-major in memory, column-major layout).
- All boundary operations go through `setBoundary(BoundaryMode, std::vector<float>&)`.
- Obstacle enforcement is always applied **before** diffusion and **after** advection.
- Projection uses `Eigen::SparseMatrix<float>` with `Eigen::SimplicialLLT` solver — the matrix is rebuilt each call (acceptable for 128×128; revisit for larger grids).
- Semi-Lagrangian advection clamps backtracked positions to `[0.5, n-1.5]`.

### Adding New Obstacles

1. Add a `setObstacle*()` method to `FluidField.hpp` / `FluidField.cpp`.
2. Fill `obstacle_[idx]` to `1` for solid cells.
3. Call the new setter from `main.cpp` after constructing the `FluidField`.
4. Add a test in `FluidFieldTests.cpp` verifying no-slip enforcement.

### Adding New Visualization Modes

1. Add a new branch to the visualization `switch` in `main.cpp` (around the pixel coloring loop).
2. Bind to the next available number key.
3. Document in the README keyboard controls table.

---

## Development Workflow

There is no CI/CD pipeline. The workflow is entirely local:

```bash
# 1. Edit source
# 2. Rebuild
cmake --build build

# 3. Run tests
ctest --test-dir build --output-on-failure

# 4. Run app
./build/ns_solver_app
```

If CMakeLists.txt changes, re-run the configure step:
```bash
cmake -B build
cmake --build build
```

---

## Future Work (from Agent.md and README)

- More obstacle geometry (rectangles, polygons, STL import)
- Improved inflow/outflow boundary conditions
- Performance: OpenMP parallelism, SIMD, CUDA port
- Larger grid support (requires revisiting per-frame Eigen solve)
- Formal CI pipeline
- Configurable simulation parameters (currently all hardcoded)

---

## Dependency Notes

- **Eigen 3** must be installed system-wide (or via vcpkg/conan). CMake finds it with `find_package(Eigen3 REQUIRED)`.
- **SFML 3** (not SFML 2) is required. The API differs significantly between major versions. CMake finds components `Graphics` and `Window`.
- No package manager lockfile is present — install dependencies through your OS package manager or build from source.
