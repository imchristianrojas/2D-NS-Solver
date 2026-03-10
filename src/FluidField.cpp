#include "FluidField.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace {

constexpr int kJacobiIterations = 12;
constexpr float kDensityClamp = 255.0F;

}

FluidField::FluidField(int size, float diffusion, float viscosity, float dt)
    : m_size(size),
      m_dt(dt),
      m_diffusion(diffusion),
      m_viscosity(viscosity),
      m_inflowVelocity(1.75F),
      m_densityDissipation(0.997F),
      m_density(static_cast<std::size_t>(size * size), 0.0F),
      m_densityScratch(static_cast<std::size_t>(size * size), 0.0F),
      m_pressure(static_cast<std::size_t>(size * size), 0.0F),
      m_velocityX(static_cast<std::size_t>(size * size), 0.0F),
      m_velocityY(static_cast<std::size_t>(size * size), 0.0F),
      m_velocityXScratch(static_cast<std::size_t>(size * size), 0.0F),
      m_velocityYScratch(static_cast<std::size_t>(size * size), 0.0F),
      m_obstacles(static_cast<std::size_t>(size * size), 0U) {}

void FluidField::step() {
    applyInflow();
    enforceObstacles();

    diffuse(m_velocityXScratch, m_velocityX, m_viscosity, BoundaryMode::HorizontalVelocity);
    diffuse(m_velocityYScratch, m_velocityY, m_viscosity, BoundaryMode::VerticalVelocity);
    project(m_velocityXScratch, m_velocityYScratch);

    advect(
        m_velocityX,
        m_velocityXScratch,
        m_velocityXScratch,
        m_velocityYScratch,
        BoundaryMode::HorizontalVelocity
    );
    advect(
        m_velocityY,
        m_velocityYScratch,
        m_velocityXScratch,
        m_velocityYScratch,
        BoundaryMode::VerticalVelocity
    );
    project(m_velocityX, m_velocityY);

    diffuse(m_densityScratch, m_density, m_diffusion, BoundaryMode::Scalar);
    advect(m_density, m_densityScratch, m_velocityX, m_velocityY, BoundaryMode::Scalar);

    for (float& value : m_density) {
        value = std::clamp(value * m_densityDissipation, 0.0F, kDensityClamp);
    }

    enforceObstacles();
}

void FluidField::addDensity(int x, int y, float amount) {
    const auto cell = static_cast<std::size_t>(index(x, y));
    if (m_obstacles[cell] != 0U) {
        return;
    }
    m_density[cell] += amount;
}

void FluidField::addVelocity(int x, int y, float amountX, float amountY) {
    const auto cell = static_cast<std::size_t>(index(x, y));
    if (m_obstacles[cell] != 0U) {
        return;
    }
    m_velocityX[cell] += amountX;
    m_velocityY[cell] += amountY;
}

void FluidField::clearObstacles() {
    std::fill(m_obstacles.begin(), m_obstacles.end(), 0U);
}

void FluidField::setObstacleCircle(int centerX, int centerY, float radius) {
    const float radiusSquared = radius * radius;
    for (int y = 1; y < m_size - 1; ++y) {
        for (int x = 1; x < m_size - 1; ++x) {
            const float dx = static_cast<float>(x - centerX);
            const float dy = static_cast<float>(y - centerY);
            if ((dx * dx) + (dy * dy) <= radiusSquared) {
                m_obstacles[static_cast<std::size_t>(index(x, y))] = 1U;
            }
        }
    }
    enforceObstacles();
}

void FluidField::setObstacleAirfoil(int leadingEdgeX, int centerY, float chord, float thickness) {
    for (int y = 1; y < m_size - 1; ++y) {
        for (int x = 1; x < m_size - 1; ++x) {
            const float localX = (static_cast<float>(x) - static_cast<float>(leadingEdgeX)) / chord;
            if (localX < 0.0F || localX > 1.0F) {
                continue;
            }

            const float thicknessDistribution =
                5.0F * thickness *
                ((0.2969F * std::sqrt(localX)) -
                 (0.1260F * localX) -
                 (0.3516F * localX * localX) +
                 (0.2843F * localX * localX * localX) -
                 (0.1036F * localX * localX * localX * localX));

            const float halfThickness = std::max(0.5F, chord * thicknessDistribution);
            const float dy = std::abs(static_cast<float>(y - centerY));
            if (dy <= halfThickness) {
                m_obstacles[static_cast<std::size_t>(index(x, y))] = 1U;
            }
        }
    }

    enforceObstacles();
}

int FluidField::size() const noexcept {
    return m_size;
}

float FluidField::densityAt(int x, int y) const {
    return m_density[static_cast<std::size_t>(index(x, y))];
}

float FluidField::pressureAt(int x, int y) const {
    return m_pressure[static_cast<std::size_t>(index(x, y))];
}

float FluidField::velocityXAt(int x, int y) const {
    return m_velocityX[static_cast<std::size_t>(index(x, y))];
}

float FluidField::velocityYAt(int x, int y) const {
    return m_velocityY[static_cast<std::size_t>(index(x, y))];
}

bool FluidField::isObstacleAt(int x, int y) const {
    return m_obstacles[static_cast<std::size_t>(index(x, y))] != 0U;
}

int FluidField::index(int x, int y) const {
    const int clampedX = std::clamp(x, 0, m_size - 1);
    const int clampedY = std::clamp(y, 0, m_size - 1);
    return clampedX + (clampedY * m_size);
}

void FluidField::enforceObstacles() {
    for (std::size_t i = 0; i < m_obstacles.size(); ++i) {
        if (m_obstacles[i] == 0U) {
            continue;
        }

        m_density[i] = 0.0F;
        m_densityScratch[i] = 0.0F;
        m_pressure[i] = 0.0F;
        m_velocityX[i] = 0.0F;
        m_velocityY[i] = 0.0F;
        m_velocityXScratch[i] = 0.0F;
        m_velocityYScratch[i] = 0.0F;
    }
}

void FluidField::applyInflow() {
    for (int y = 1; y < m_size - 1; ++y) {
        const std::size_t inlet = static_cast<std::size_t>(index(1, y));
        if (m_obstacles[inlet] != 0U) {
            continue;
        }
        m_velocityX[inlet] = m_inflowVelocity;
        m_velocityY[inlet] = 0.0F;

        const int centerline = m_size / 2;
        const int upperTracer = centerline - (m_size / 10);
        const int lowerTracer = centerline + (m_size / 10);
        const bool mainStream = std::abs(y - centerline) < (m_size / 5);
        const bool tracerStream = std::abs(y - upperTracer) <= 1 || std::abs(y - lowerTracer) <= 1;
        if (mainStream) {
            m_density[inlet] = std::max(m_density[inlet], 18.0F);
        }
        if (tracerStream) {
            m_density[inlet] = std::max(m_density[inlet], 64.0F);
        }
    }

    setBoundary(m_velocityX, BoundaryMode::HorizontalVelocity);
    setBoundary(m_velocityY, BoundaryMode::VerticalVelocity);
    setBoundary(m_density, BoundaryMode::Scalar);
}

void FluidField::diffuse(
    std::vector<float>& current,
    const std::vector<float>& previous,
    float rate,
    BoundaryMode mode
) {
    current = previous;
    const float a = m_dt * rate * static_cast<float>((m_size - 2) * (m_size - 2));

    for (int iter = 0; iter < kJacobiIterations; ++iter) {
        for (int y = 1; y < m_size - 1; ++y) {
            for (int x = 1; x < m_size - 1; ++x) {
                const int cell = index(x, y);
                if (m_obstacles[static_cast<std::size_t>(cell)] != 0U) {
                    current[static_cast<std::size_t>(cell)] = 0.0F;
                    continue;
                }
                current[static_cast<std::size_t>(cell)] =
                    (previous[static_cast<std::size_t>(cell)] +
                     a * (current[static_cast<std::size_t>(index(x - 1, y))] +
                          current[static_cast<std::size_t>(index(x + 1, y))] +
                          current[static_cast<std::size_t>(index(x, y - 1))] +
                          current[static_cast<std::size_t>(index(x, y + 1))])) /
                    (1.0F + 4.0F * a);
            }
        }
        setBoundary(current, mode);
    }
}

void FluidField::advect(
    std::vector<float>& current,
    const std::vector<float>& previous,
    const std::vector<float>& velocityX,
    const std::vector<float>& velocityY,
    BoundaryMode mode
) {
    for (int y = 1; y < m_size - 1; ++y) {
        for (int x = 1; x < m_size - 1; ++x) {
            const int cell = index(x, y);
            if (m_obstacles[static_cast<std::size_t>(cell)] != 0U) {
                current[static_cast<std::size_t>(cell)] = 0.0F;
                continue;
            }
            const float backtraceX =
                static_cast<float>(x) - m_dt * static_cast<float>(m_size - 2) * velocityX[static_cast<std::size_t>(cell)];
            const float backtraceY =
                static_cast<float>(y) - m_dt * static_cast<float>(m_size - 2) * velocityY[static_cast<std::size_t>(cell)];

            const float clampedX = std::clamp(backtraceX, 0.5F, static_cast<float>(m_size) - 1.5F);
            const float clampedY = std::clamp(backtraceY, 0.5F, static_cast<float>(m_size) - 1.5F);

            const int x0 = static_cast<int>(clampedX);
            const int y0 = static_cast<int>(clampedY);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;

            const float sx = clampedX - static_cast<float>(x0);
            const float sy = clampedY - static_cast<float>(y0);

            const float sample =
                (1.0F - sx) *
                    ((1.0F - sy) * previous[static_cast<std::size_t>(index(x0, y0))] +
                     sy * previous[static_cast<std::size_t>(index(x0, y1))]) +
                sx *
                    ((1.0F - sy) * previous[static_cast<std::size_t>(index(x1, y0))] +
                     sy * previous[static_cast<std::size_t>(index(x1, y1))]);

            current[static_cast<std::size_t>(cell)] = sample;
        }
    }

    setBoundary(current, mode);
}

void FluidField::project(std::vector<float>& velocityX, std::vector<float>& velocityY) {
    Eigen::VectorXf pressure = Eigen::VectorXf::Zero(m_size * m_size);
    Eigen::VectorXf divergence = Eigen::VectorXf::Zero(m_size * m_size);

    for (int y = 1; y < m_size - 1; ++y) {
        for (int x = 1; x < m_size - 1; ++x) {
            const int cell = index(x, y);
            if (m_obstacles[static_cast<std::size_t>(cell)] != 0U) {
                divergence[cell] = 0.0F;
                continue;
            }
            divergence[cell] = -0.5F *
                               (velocityX[static_cast<std::size_t>(index(x + 1, y))] -
                                velocityX[static_cast<std::size_t>(index(x - 1, y))] +
                                velocityY[static_cast<std::size_t>(index(x, y + 1))] -
                                velocityY[static_cast<std::size_t>(index(x, y - 1))]) /
                               static_cast<float>(m_size);
        }
    }

    std::vector<float> pressureField(static_cast<std::size_t>(m_size * m_size), 0.0F);
    std::vector<float> divergenceField(static_cast<std::size_t>(m_size * m_size), 0.0F);
    for (int i = 0; i < pressure.size(); ++i) {
        divergenceField[static_cast<std::size_t>(i)] = divergence[i];
    }
    setBoundary(divergenceField, BoundaryMode::Scalar);

    for (int iter = 0; iter < kJacobiIterations; ++iter) {
        for (int y = 1; y < m_size - 1; ++y) {
            for (int x = 1; x < m_size - 1; ++x) {
                const int cell = index(x, y);
                if (m_obstacles[static_cast<std::size_t>(cell)] != 0U) {
                    pressure[cell] = 0.0F;
                    pressureField[static_cast<std::size_t>(cell)] = 0.0F;
                    continue;
                }
                pressure[cell] =
                    (divergenceField[static_cast<std::size_t>(cell)] +
                     pressure[index(x - 1, y)] +
                     pressure[index(x + 1, y)] +
                     pressure[index(x, y - 1)] +
                     pressure[index(x, y + 1)]) /
                    4.0F;
                pressureField[static_cast<std::size_t>(cell)] = pressure[cell];
            }
        }
        setBoundary(pressureField, BoundaryMode::Scalar);
        for (int i = 0; i < pressure.size(); ++i) {
            pressure[i] = pressureField[static_cast<std::size_t>(i)];
        }
    }

    for (int y = 1; y < m_size - 1; ++y) {
        for (int x = 1; x < m_size - 1; ++x) {
            const int cell = index(x, y);
            if (m_obstacles[static_cast<std::size_t>(cell)] != 0U) {
                velocityX[static_cast<std::size_t>(cell)] = 0.0F;
                velocityY[static_cast<std::size_t>(cell)] = 0.0F;
                continue;
            }
            velocityX[static_cast<std::size_t>(cell)] -=
                0.5F * static_cast<float>(m_size) * (pressure[index(x + 1, y)] - pressure[index(x - 1, y)]);
            velocityY[static_cast<std::size_t>(cell)] -=
                0.5F * static_cast<float>(m_size) * (pressure[index(x, y + 1)] - pressure[index(x, y - 1)]);
        }
    }

    for (int i = 0; i < pressure.size(); ++i) {
        m_pressure[static_cast<std::size_t>(i)] = pressure[i];
    }

    setBoundary(velocityX, BoundaryMode::HorizontalVelocity);
    setBoundary(velocityY, BoundaryMode::VerticalVelocity);
    enforceObstacles();
}

void FluidField::setBoundary(std::vector<float>& field, BoundaryMode mode) const {
    for (int i = 1; i < m_size - 1; ++i) {
        field[static_cast<std::size_t>(index(0, i))] =
            (mode == BoundaryMode::HorizontalVelocity) ? m_inflowVelocity : field[static_cast<std::size_t>(index(1, i))];
        field[static_cast<std::size_t>(index(m_size - 1, i))] = field[static_cast<std::size_t>(index(m_size - 2, i))];

        if (mode == BoundaryMode::VerticalVelocity) {
            field[static_cast<std::size_t>(index(i, 0))] = 0.0F;
            field[static_cast<std::size_t>(index(i, m_size - 1))] = 0.0F;
        } else {
            field[static_cast<std::size_t>(index(i, 0))] =
                (mode == BoundaryMode::HorizontalVelocity)
                    ? field[static_cast<std::size_t>(index(i, 1))]
                    : field[static_cast<std::size_t>(index(i, 1))];
            field[static_cast<std::size_t>(index(i, m_size - 1))] =
                (mode == BoundaryMode::HorizontalVelocity)
                    ? field[static_cast<std::size_t>(index(i, m_size - 2))]
                    : field[static_cast<std::size_t>(index(i, m_size - 2))];
        }
    }

    field[static_cast<std::size_t>(index(0, 0))] =
        0.5F * (field[static_cast<std::size_t>(index(1, 0))] + field[static_cast<std::size_t>(index(0, 1))]);
    field[static_cast<std::size_t>(index(0, m_size - 1))] =
        0.5F * (field[static_cast<std::size_t>(index(1, m_size - 1))] + field[static_cast<std::size_t>(index(0, m_size - 2))]);
    field[static_cast<std::size_t>(index(m_size - 1, 0))] =
        0.5F * (field[static_cast<std::size_t>(index(m_size - 2, 0))] + field[static_cast<std::size_t>(index(m_size - 1, 1))]);
    field[static_cast<std::size_t>(index(m_size - 1, m_size - 1))] =
        0.5F * (field[static_cast<std::size_t>(index(m_size - 2, m_size - 1))] +
                field[static_cast<std::size_t>(index(m_size - 1, m_size - 2))]);

    for (std::size_t i = 0; i < m_obstacles.size(); ++i) {
        if (m_obstacles[i] != 0U) {
            field[i] = 0.0F;
        }
    }
}
