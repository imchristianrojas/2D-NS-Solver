This document serves as the primary technical specification and roadmap for the project. The goal is to develop a high-performance 2D fluid dynamics solver in C++ to simulate airflow over Formula 1 aerodynamic components.

1. Project Overview

Objective: Create a grid-based (Eulerian) fluid solver using the Stable Fluids method to visualize 2D airflow.

Target hardware: Initial CPU implementation with a future path to GPU acceleration via CUDA (RTX 3070).

Core Physics: Incompressible Navier-Stokes equations.

2. Technical Stack

Language: C++17/20

Math Library: Eigen (for linear system solving)

Graphics: SFML (Simple and Fast Multimedia Library) for real-time visualization.

Build System: CMake

3. Requirements & Constraints

Performance: Must maintain at least 30 FPS on a 128×128 grid on the CPU.

Stability: Numerical integration must remain stable even with high-velocity "wind tunnel" inputs.

Boundary Conditions:

Inflow: Constant velocity from the left boundary.

Outflow: Zero-gradient/Open boundary on the right.

Obstacles: Support for arbitrary 2D shapes (F1 wing cross-sections) using no-slip conditions.

Visualization: Toggleable views for Velocity Fields (vectors), Pressure Maps (color gradients), and Smoke/Density injection.

4. Implementation Roadmap

Phase 1: The FluidField Class

Establish the data structures. We need two main buffers for velocity (u,v) and density to allow for "ping-pong" buffering during advection.

C++
class FluidField {
private:
    int size;
    float dt;
    float diffusion;
    float viscosity;

    std::vector<float> s;    // Density
    std::vector<float> density;

    std::vector<float> Vx;   // Velocity X
    std::vector<float> Vy;   // Velocity Y

    std::vector<float> Vx0;  // Previous Velocity X
    std::vector<float> Vy0;  // Previous Velocity Y

public:
    FluidField(int size, float diffusion, float viscosity, float dt);
    void step();             // The main simulation loop
    void addDensity(int x, int y, float amount);
    void addVelocity(int x, int y, float px, float py);
    // ... Solver methods (lin_solve, project, advect, diffuse)
};
Phase 2: The Solver Loop

Add Forces: Incorporate external inputs (user interaction or wind tunnel).

Diffuse: Spread the velocity/density using a Gauss-Seidel iterative solver.

Project (The Divergence-Free Step): Solve the Poisson equation for pressure to ensure mass conservation.

Advect: Move the fluid quantities along the velocity field.

Project (Again): Final correction to ensure the field remains incompressible.

Phase 3: Obstacle Integration

Implement a mask grid. If a cell is part of an "F1 Part," its velocity is forced to zero, and it acts as a hard boundary for the pressure solver.

5. Future Scope (The "Up-Level")

Multi-threading: Use OpenMP to parallelize the lin_solve loops.

CUDA Port: Move the massive grid arrays to the 3070 VRAM and write kernels for the projection and advection steps.

Importing Geometry: Ability to load .svg or simple coordinate files for specific F1 wing profiles (e.g., a 2026-spec front wing flap).