#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class FluidField {
public:
    FluidField(int size, float diffusion, float viscosity, float dt);

    void step();
    void addDensity(int x, int y, float amount);
    void addVelocity(int x, int y, float amountX, float amountY);
    void clearObstacles();
    void setObstacleCircle(int centerX, int centerY, float radius);
    void setObstacleAirfoil(int leadingEdgeX, int centerY, float chord, float thickness);

    [[nodiscard]] int size() const noexcept;
    [[nodiscard]] float densityAt(int x, int y) const;
    [[nodiscard]] float pressureAt(int x, int y) const;
    [[nodiscard]] float velocityXAt(int x, int y) const;
    [[nodiscard]] float velocityYAt(int x, int y) const;
    [[nodiscard]] bool isObstacleAt(int x, int y) const;

private:
    enum class BoundaryMode {
        Scalar,
        HorizontalVelocity,
        VerticalVelocity,
    };

    [[nodiscard]] int index(int x, int y) const;
    void enforceObstacles();
    void applyInflow();
    void diffuse(std::vector<float>& current, const std::vector<float>& previous, float rate, BoundaryMode mode);
    void advect(
        std::vector<float>& current,
        const std::vector<float>& previous,
        const std::vector<float>& velocityX,
        const std::vector<float>& velocityY,
        BoundaryMode mode
    );
    void project(std::vector<float>& velocityX, std::vector<float>& velocityY);
    void setBoundary(std::vector<float>& field, BoundaryMode mode) const;

    int m_size;
    float m_dt;
    float m_diffusion;
    float m_viscosity;
    float m_inflowVelocity;
    float m_densityDissipation;

    std::vector<float> m_density;
    std::vector<float> m_densityScratch;
    std::vector<float> m_pressure;

    std::vector<float> m_velocityX;
    std::vector<float> m_velocityY;
    std::vector<float> m_velocityXScratch;
    std::vector<float> m_velocityYScratch;
    std::vector<std::uint8_t> m_obstacles;
};
