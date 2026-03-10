# 2D NS Solver

Early scaffold for a 2D incompressible Navier-Stokes solver focused on Formula 1 style airflow visualization.

## Current scope

- CMake-based C++20 project layout
- `FluidField` core type with an initial Stable Fluids style update loop
- SFML window that renders density or velocity magnitude as a color field
- Mouse injection to seed density and upward velocity into the domain
- Basic left-boundary inflow to mimic a wind tunnel feed

## Dependencies

- CMake 3.20+
- A C++20 compiler
- Eigen 3.4+
- SFML 2.6+

## Configure

```bash
cmake -S . -B build
cmake --build build
./build/ns_solver_app
```

Press `1` for density view and `2` for velocity magnitude.

## Next steps

- Add pressure visualization alongside the current density and velocity-magnitude views
- Introduce explicit inflow/outflow and obstacle masks
- Add tests and performance measurements for the `128x128` target
