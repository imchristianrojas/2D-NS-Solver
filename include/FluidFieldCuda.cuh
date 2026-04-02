#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

enum class ViewMode : int {
    Density = 0,
    VelocityMagnitude = 1,
    Pressure = 2,
};

class FluidFieldCuda {
public:
    FluidFieldCuda(int size, float diffusion, float viscosity, float dt);
    ~FluidFieldCuda();

    FluidFieldCuda(const FluidFieldCuda&) = delete;
    FluidFieldCuda& operator=(const FluidFieldCuda&) = delete;

    void step();
    void addDensity(int x, int y, float amount);
    void addVelocity(int x, int y, float amountX, float amountY);
    void clearObstacles();
    void setObstacleCircle(int centerX, int centerY, float radius);
    void setObstacleAirfoil(int centerX, int centerY, float chord, float thickness, float angleDegrees = 0.0f);
    void setObstacleRectangle(int centerX, int centerY, float width, float height, float angleDegrees = 0.0f);

    // Render the current field state into an RGBA pixel buffer on the GPU,
    // then copy it to the provided host buffer. Returns gridSize*gridSize*4 bytes.
    void renderToPixels(std::uint8_t* hostPixels, ViewMode mode) const;

    // Single-cell accessors (device-to-host copy — use for tests, not rendering)
    [[nodiscard]] int size() const noexcept;
    [[nodiscard]] float densityAt(int x, int y) const;
    [[nodiscard]] float pressureAt(int x, int y) const;
    [[nodiscard]] float velocityXAt(int x, int y) const;
    [[nodiscard]] float velocityYAt(int x, int y) const;
    [[nodiscard]] bool isObstacleAt(int x, int y) const;

private:
    int m_size;
    int m_totalCells;
    float m_dt;
    float m_diffusion;
    float m_viscosity;
    float m_inflowVelocity;
    float m_densityDissipation;
    int m_jacobiIterations;

    // Device pointers (GPU memory)
    float* d_density;
    float* d_densityScratch;
    float* d_pressure;
    float* d_pressureScratch;
    float* d_divergence;
    float* d_velocityX;
    float* d_velocityY;
    float* d_velocityXScratch;
    float* d_velocityYScratch;
    std::uint8_t* d_obstacles;

    // Pixel rendering buffer (GPU)
    std::uint8_t* d_pixels;

    // Host-side obstacle buffer for building masks on CPU
    std::vector<std::uint8_t> h_obstacles;

    void uploadObstacles();
};
