#include "FluidFieldCuda.cuh"

#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdio>
#include <algorithm>

// ── Helpers ──────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err));                   \
        }                                                                      \
    } while (0)

static constexpr int kBlockSize = 16;

__device__ __forceinline__ int devIdx(int x, int y, int n) {
    return x + y * n;
}

static dim3 gridDim2D(int n) {
    return dim3((n + kBlockSize - 1) / kBlockSize,
                (n + kBlockSize - 1) / kBlockSize);
}

static dim3 blockDim2D() {
    return dim3(kBlockSize, kBlockSize);
}

// ── Kernels ──────────────────────────────────────────────────────────────────

__global__ void applyInflowKernel(float* vx, float* vy, float* density,
                                   const std::uint8_t* obstacle, int n,
                                   float inflowVelocity) {
    int y = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (y >= n - 1) return;

    int idx = devIdx(1, y, n);
    if (obstacle[idx] != 0) return;

    vx[idx] = inflowVelocity;
    vy[idx] = 0.0f;

    int centerline = n / 2;
    int upperTracer = centerline - (n / 10);
    int lowerTracer = centerline + (n / 10);
    bool mainStream = abs(y - centerline) < (n / 5);
    bool tracerStream = abs(y - upperTracer) <= 1 || abs(y - lowerTracer) <= 1;

    if (mainStream) {
        density[idx] = fmaxf(density[idx], 18.0f);
    }
    if (tracerStream) {
        density[idx] = fmaxf(density[idx], 64.0f);
    }
}

__global__ void enforceObstaclesKernel(float* density, float* densityScratch,
                                        float* pressure, float* vx, float* vy,
                                        float* vxScratch, float* vyScratch,
                                        const std::uint8_t* obstacle, int totalCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalCells) return;
    if (obstacle[i] == 0) return;

    density[i] = 0.0f;
    densityScratch[i] = 0.0f;
    pressure[i] = 0.0f;
    vx[i] = 0.0f;
    vy[i] = 0.0f;
    vxScratch[i] = 0.0f;
    vyScratch[i] = 0.0f;
}

// Jacobi diffusion: dst[cell] = (src[cell] + a * sum_neighbors(dst)) / (1 + 4a)
// We read from 'src' (previous iteration) and write to 'dst' (current iteration).
// Caller ping-pongs src/dst pointers between iterations.
__global__ void diffuseKernel(float* dst, const float* src, const float* original,
                               const std::uint8_t* obstacle, int n, float a) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= n - 1 || y >= n - 1) return;

    int idx = devIdx(x, y, n);
    if (obstacle[idx] != 0) {
        dst[idx] = 0.0f;
        return;
    }

    float neighborSum = src[devIdx(x - 1, y, n)] + src[devIdx(x + 1, y, n)] +
                        src[devIdx(x, y - 1, n)] + src[devIdx(x, y + 1, n)];
    dst[idx] = (original[idx] + a * neighborSum) / (1.0f + 4.0f * a);
}

__global__ void advectKernel(float* dst, const float* src,
                              const float* velX, const float* velY,
                              const std::uint8_t* obstacle, int n, float dtScale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= n - 1 || y >= n - 1) return;

    int idx = devIdx(x, y, n);
    if (obstacle[idx] != 0) {
        dst[idx] = 0.0f;
        return;
    }

    float bx = (float)x - dtScale * velX[idx];
    float by = (float)y - dtScale * velY[idx];

    bx = fmaxf(0.5f, fminf(bx, (float)n - 1.5f));
    by = fmaxf(0.5f, fminf(by, (float)n - 1.5f));

    int x0 = (int)bx;
    int y0 = (int)by;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float sx = bx - (float)x0;
    float sy = by - (float)y0;

    dst[idx] = (1.0f - sx) * ((1.0f - sy) * src[devIdx(x0, y0, n)] +
                               sy * src[devIdx(x0, y1, n)]) +
               sx * ((1.0f - sy) * src[devIdx(x1, y0, n)] +
                      sy * src[devIdx(x1, y1, n)]);
}

__global__ void projectDivergenceKernel(float* divergence, const float* vx,
                                         const float* vy,
                                         const std::uint8_t* obstacle, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= n - 1 || y >= n - 1) return;

    int idx = devIdx(x, y, n);
    if (obstacle[idx] != 0) {
        divergence[idx] = 0.0f;
        return;
    }

    divergence[idx] = -0.5f * (vx[devIdx(x + 1, y, n)] - vx[devIdx(x - 1, y, n)] +
                                vy[devIdx(x, y + 1, n)] - vy[devIdx(x, y - 1, n)]) /
                      (float)n;
}

// Jacobi iteration for pressure Poisson: p = (div + sum_neighbors(p_prev)) / 4
__global__ void projectJacobiKernel(float* pDst, const float* pSrc,
                                     const float* divergence,
                                     const std::uint8_t* obstacle, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= n - 1 || y >= n - 1) return;

    int idx = devIdx(x, y, n);
    if (obstacle[idx] != 0) {
        pDst[idx] = 0.0f;
        return;
    }

    pDst[idx] = (divergence[idx] + pSrc[devIdx(x - 1, y, n)] +
                 pSrc[devIdx(x + 1, y, n)] + pSrc[devIdx(x, y - 1, n)] +
                 pSrc[devIdx(x, y + 1, n)]) /
                4.0f;
}

__global__ void projectApplyKernel(float* vx, float* vy, const float* pressure,
                                    const std::uint8_t* obstacle, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= n - 1 || y >= n - 1) return;

    int idx = devIdx(x, y, n);
    if (obstacle[idx] != 0) {
        vx[idx] = 0.0f;
        vy[idx] = 0.0f;
        return;
    }

    vx[idx] -= 0.5f * (float)n * (pressure[devIdx(x + 1, y, n)] - pressure[devIdx(x - 1, y, n)]);
    vy[idx] -= 0.5f * (float)n * (pressure[devIdx(x, y + 1, n)] - pressure[devIdx(x, y - 1, n)]);
}

__global__ void dissipateKernel(float* density, const std::uint8_t* obstacle,
                                 int totalCells, float dissipation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalCells) return;
    if (obstacle[i] != 0) return;

    float val = density[i] * dissipation;
    density[i] = fminf(fmaxf(val, 0.0f), 255.0f);
}

// Boundary conditions — one kernel handles all edges
// mode: 0 = Scalar, 1 = HorizontalVelocity, 2 = VerticalVelocity
__global__ void setBoundaryKernel(float* field, const std::uint8_t* obstacle,
                                   int n, int mode, float inflowVelocity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i >= n - 1) return;

    // Left edge
    if (mode == 1) { // HorizontalVelocity
        field[devIdx(0, i, n)] = inflowVelocity;
    } else {
        field[devIdx(0, i, n)] = field[devIdx(1, i, n)];
    }

    // Right edge — zero-gradient outflow
    field[devIdx(n - 1, i, n)] = field[devIdx(n - 2, i, n)];

    // Top/bottom edges
    if (mode == 2) { // VerticalVelocity
        field[devIdx(i, 0, n)] = 0.0f;
        field[devIdx(i, n - 1, n)] = 0.0f;
    } else {
        field[devIdx(i, 0, n)] = field[devIdx(i, 1, n)];
        field[devIdx(i, n - 1, n)] = field[devIdx(i, n - 2, n)];
    }
}

__global__ void setBoundaryCornersKernel(float* field, int n) {
    // Only one thread needed
    if (threadIdx.x != 0) return;

    field[devIdx(0, 0, n)] =
        0.5f * (field[devIdx(1, 0, n)] + field[devIdx(0, 1, n)]);
    field[devIdx(0, n - 1, n)] =
        0.5f * (field[devIdx(1, n - 1, n)] + field[devIdx(0, n - 2, n)]);
    field[devIdx(n - 1, 0, n)] =
        0.5f * (field[devIdx(n - 2, 0, n)] + field[devIdx(n - 1, 1, n)]);
    field[devIdx(n - 1, n - 1, n)] =
        0.5f * (field[devIdx(n - 2, n - 1, n)] + field[devIdx(n - 1, n - 2, n)]);
}

__global__ void clearObstaclesInFieldKernel(float* field,
                                             const std::uint8_t* obstacle,
                                             int totalCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalCells) return;
    if (obstacle[i] != 0) {
        field[i] = 0.0f;
    }
}

// ── Render kernel ────────────────────────────────────────────────────────────

__device__ void densityToRGBA(float d, std::uint8_t* out) {
    float clamped = fminf(fmaxf(d, 0.0f), 255.0f) / 255.0f;
    float shaped = sqrtf(clamped);
    out[0] = (std::uint8_t)fminf(fmaxf(30.0f + 225.0f * shaped, 0.0f), 255.0f);
    out[1] = (std::uint8_t)fminf(fmaxf(10.0f + 180.0f * shaped * shaped, 0.0f), 255.0f);
    out[2] = (std::uint8_t)fminf(fmaxf(24.0f + 255.0f * (1.0f - shaped * 0.75f), 0.0f), 255.0f);
    out[3] = 255;
}

__device__ void velocityToRGBA(float vx, float vy, std::uint8_t* out) {
    float speed = sqrtf(vx * vx + vy * vy);
    float normalized = fminf(fmaxf(speed * 48.0f, 0.0f), 255.0f);
    std::uint8_t intensity = (std::uint8_t)normalized;
    out[0] = 20;
    out[1] = intensity;
    out[2] = (std::uint8_t)(255 - intensity / 2);
    out[3] = 255;
}

__device__ void pressureToRGBA(float p, std::uint8_t* out) {
    float clamped = fminf(fmaxf(p * 900.0f, -255.0f), 255.0f);
    if (clamped >= 0.0f) {
        std::uint8_t warm = (std::uint8_t)clamped;
        out[0] = 255;
        out[1] = (std::uint8_t)(255 - warm / 2);
        out[2] = (std::uint8_t)(255 - warm);
    } else {
        std::uint8_t cool = (std::uint8_t)(-clamped);
        out[0] = (std::uint8_t)(255 - cool);
        out[1] = (std::uint8_t)(255 - cool / 3);
        out[2] = 255;
    }
    out[3] = 255;
}

__global__ void renderKernel(std::uint8_t* pixels,
                              const float* density, const float* pressure,
                              const float* vx, const float* vy,
                              const std::uint8_t* obstacle,
                              int n, int mode) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    int idx = devIdx(x, y, n);
    int pixIdx = (y * n + x) * 4;

    if (obstacle[idx] != 0) {
        pixels[pixIdx + 0] = 244;
        pixels[pixIdx + 1] = 236;
        pixels[pixIdx + 2] = 210;
        pixels[pixIdx + 3] = 255;
        return;
    }

    if (mode == 1) {
        velocityToRGBA(vx[idx], vy[idx], &pixels[pixIdx]);
    } else if (mode == 2) {
        pressureToRGBA(pressure[idx], &pixels[pixIdx]);
    } else {
        densityToRGBA(density[idx], &pixels[pixIdx]);
    }
}

// ── Host-side helpers ────────────────────────────────────────────────────────

static void launchBoundary(float* field, const std::uint8_t* obstacle, int n,
                           int mode, float inflowVelocity) {
    int edgeBlocks = (n - 2 + 255) / 256;
    setBoundaryKernel<<<edgeBlocks, 256>>>(field, obstacle, n, mode, inflowVelocity);
    setBoundaryCornersKernel<<<1, 1>>>(field, n);

    int totalBlocks = (n * n + 255) / 256;
    clearObstaclesInFieldKernel<<<totalBlocks, 256>>>(field, obstacle, n * n);
}

// ── FluidFieldCuda implementation ────────────────────────────────────────────

FluidFieldCuda::FluidFieldCuda(int size, float diffusion, float viscosity, float dt)
    : m_size(size),
      m_totalCells(size * size),
      m_dt(dt),
      m_diffusion(diffusion),
      m_viscosity(viscosity),
      m_inflowVelocity(1.75f),
      m_densityDissipation(0.997f),
      m_jacobiIterations(50),
      h_obstacles(static_cast<std::size_t>(size * size), 0U) {

    std::size_t floatBytes = static_cast<std::size_t>(m_totalCells) * sizeof(float);
    std::size_t byteBytes = static_cast<std::size_t>(m_totalCells) * sizeof(std::uint8_t);
    std::size_t pixelBytes = static_cast<std::size_t>(m_totalCells) * 4;

    CUDA_CHECK(cudaMalloc(&d_density, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_densityScratch, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_pressure, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_pressureScratch, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_divergence, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_velocityX, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_velocityY, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_velocityXScratch, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_velocityYScratch, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_obstacles, byteBytes));
    CUDA_CHECK(cudaMalloc(&d_pixels, pixelBytes));

    CUDA_CHECK(cudaMemset(d_density, 0, floatBytes));
    CUDA_CHECK(cudaMemset(d_densityScratch, 0, floatBytes));
    CUDA_CHECK(cudaMemset(d_pressure, 0, floatBytes));
    CUDA_CHECK(cudaMemset(d_pressureScratch, 0, floatBytes));
    CUDA_CHECK(cudaMemset(d_divergence, 0, floatBytes));
    CUDA_CHECK(cudaMemset(d_velocityX, 0, floatBytes));
    CUDA_CHECK(cudaMemset(d_velocityY, 0, floatBytes));
    CUDA_CHECK(cudaMemset(d_velocityXScratch, 0, floatBytes));
    CUDA_CHECK(cudaMemset(d_velocityYScratch, 0, floatBytes));
    CUDA_CHECK(cudaMemset(d_obstacles, 0, byteBytes));
    CUDA_CHECK(cudaMemset(d_pixels, 0, pixelBytes));
}

FluidFieldCuda::~FluidFieldCuda() {
    cudaFree(d_density);
    cudaFree(d_densityScratch);
    cudaFree(d_pressure);
    cudaFree(d_pressureScratch);
    cudaFree(d_divergence);
    cudaFree(d_velocityX);
    cudaFree(d_velocityY);
    cudaFree(d_velocityXScratch);
    cudaFree(d_velocityYScratch);
    cudaFree(d_obstacles);
    cudaFree(d_pixels);
}

void FluidFieldCuda::step() {
    int n = m_size;
    dim3 grid2 = gridDim2D(n - 2);
    dim3 block2 = blockDim2D();
    int linearBlocks = (m_totalCells + 255) / 256;
    int rowBlocks = (n - 2 + 255) / 256;

    // 1. Apply inflow
    applyInflowKernel<<<rowBlocks, 256>>>(d_velocityX, d_velocityY, d_density,
                                           d_obstacles, n, m_inflowVelocity);
    launchBoundary(d_velocityX, d_obstacles, n, 1, m_inflowVelocity);
    launchBoundary(d_velocityY, d_obstacles, n, 2, m_inflowVelocity);
    launchBoundary(d_density, d_obstacles, n, 0, m_inflowVelocity);

    // 2. Enforce obstacles
    enforceObstaclesKernel<<<linearBlocks, 256>>>(
        d_density, d_densityScratch, d_pressure, d_velocityX, d_velocityY,
        d_velocityXScratch, d_velocityYScratch, d_obstacles, m_totalCells);

    // 3. Diffuse velocity (viscosity)
    float a_visc = m_dt * m_viscosity * static_cast<float>((n - 2) * (n - 2));

    // Copy current velocity to scratch as starting point
    CUDA_CHECK(cudaMemcpy(d_velocityXScratch, d_velocityX,
                          m_totalCells * sizeof(float), cudaMemcpyDeviceToDevice));
    for (int iter = 0; iter < m_jacobiIterations; ++iter) {
        diffuseKernel<<<grid2, block2>>>(d_velocityXScratch, d_velocityXScratch,
                                          d_velocityX, d_obstacles, n, a_visc);
        launchBoundary(d_velocityXScratch, d_obstacles, n, 1, m_inflowVelocity);
    }

    CUDA_CHECK(cudaMemcpy(d_velocityYScratch, d_velocityY,
                          m_totalCells * sizeof(float), cudaMemcpyDeviceToDevice));
    for (int iter = 0; iter < m_jacobiIterations; ++iter) {
        diffuseKernel<<<grid2, block2>>>(d_velocityYScratch, d_velocityYScratch,
                                          d_velocityY, d_obstacles, n, a_visc);
        launchBoundary(d_velocityYScratch, d_obstacles, n, 2, m_inflowVelocity);
    }

    // 4. Project (pressure solve after diffusion)
    projectDivergenceKernel<<<grid2, block2>>>(d_divergence, d_velocityXScratch,
                                                d_velocityYScratch, d_obstacles, n);
    launchBoundary(d_divergence, d_obstacles, n, 0, m_inflowVelocity);

    CUDA_CHECK(cudaMemset(d_pressure, 0, m_totalCells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressureScratch, 0, m_totalCells * sizeof(float)));
    for (int iter = 0; iter < m_jacobiIterations; ++iter) {
        if (iter % 2 == 0) {
            projectJacobiKernel<<<grid2, block2>>>(d_pressureScratch, d_pressure,
                                                    d_divergence, d_obstacles, n);
            launchBoundary(d_pressureScratch, d_obstacles, n, 0, m_inflowVelocity);
        } else {
            projectJacobiKernel<<<grid2, block2>>>(d_pressure, d_pressureScratch,
                                                    d_divergence, d_obstacles, n);
            launchBoundary(d_pressure, d_obstacles, n, 0, m_inflowVelocity);
        }
    }
    // Ensure d_pressure has the final result
    if (m_jacobiIterations % 2 != 0) {
        CUDA_CHECK(cudaMemcpy(d_pressure, d_pressureScratch,
                              m_totalCells * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    projectApplyKernel<<<grid2, block2>>>(d_velocityXScratch, d_velocityYScratch,
                                           d_pressure, d_obstacles, n);
    launchBoundary(d_velocityXScratch, d_obstacles, n, 1, m_inflowVelocity);
    launchBoundary(d_velocityYScratch, d_obstacles, n, 2, m_inflowVelocity);

    // 5. Advect velocity
    advectKernel<<<grid2, block2>>>(d_velocityX, d_velocityXScratch,
                                    d_velocityXScratch, d_velocityYScratch,
                                    d_obstacles, n,
                                    m_dt * static_cast<float>(n - 2));
    launchBoundary(d_velocityX, d_obstacles, n, 1, m_inflowVelocity);

    advectKernel<<<grid2, block2>>>(d_velocityY, d_velocityYScratch,
                                    d_velocityXScratch, d_velocityYScratch,
                                    d_obstacles, n,
                                    m_dt * static_cast<float>(n - 2));
    launchBoundary(d_velocityY, d_obstacles, n, 2, m_inflowVelocity);

    // 6. Second projection after advection
    projectDivergenceKernel<<<grid2, block2>>>(d_divergence, d_velocityX,
                                                d_velocityY, d_obstacles, n);
    launchBoundary(d_divergence, d_obstacles, n, 0, m_inflowVelocity);

    CUDA_CHECK(cudaMemset(d_pressure, 0, m_totalCells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressureScratch, 0, m_totalCells * sizeof(float)));
    for (int iter = 0; iter < m_jacobiIterations; ++iter) {
        if (iter % 2 == 0) {
            projectJacobiKernel<<<grid2, block2>>>(d_pressureScratch, d_pressure,
                                                    d_divergence, d_obstacles, n);
            launchBoundary(d_pressureScratch, d_obstacles, n, 0, m_inflowVelocity);
        } else {
            projectJacobiKernel<<<grid2, block2>>>(d_pressure, d_pressureScratch,
                                                    d_divergence, d_obstacles, n);
            launchBoundary(d_pressure, d_obstacles, n, 0, m_inflowVelocity);
        }
    }
    if (m_jacobiIterations % 2 != 0) {
        CUDA_CHECK(cudaMemcpy(d_pressure, d_pressureScratch,
                              m_totalCells * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    projectApplyKernel<<<grid2, block2>>>(d_velocityX, d_velocityY,
                                           d_pressure, d_obstacles, n);
    launchBoundary(d_velocityX, d_obstacles, n, 1, m_inflowVelocity);
    launchBoundary(d_velocityY, d_obstacles, n, 2, m_inflowVelocity);

    // 7. Diffuse density
    float a_diff = m_dt * m_diffusion * static_cast<float>((n - 2) * (n - 2));
    CUDA_CHECK(cudaMemcpy(d_densityScratch, d_density,
                          m_totalCells * sizeof(float), cudaMemcpyDeviceToDevice));
    for (int iter = 0; iter < m_jacobiIterations; ++iter) {
        diffuseKernel<<<grid2, block2>>>(d_densityScratch, d_densityScratch,
                                          d_density, d_obstacles, n, a_diff);
        launchBoundary(d_densityScratch, d_obstacles, n, 0, m_inflowVelocity);
    }

    // 8. Advect density
    advectKernel<<<grid2, block2>>>(d_density, d_densityScratch,
                                    d_velocityX, d_velocityY,
                                    d_obstacles, n,
                                    m_dt * static_cast<float>(n - 2));
    launchBoundary(d_density, d_obstacles, n, 0, m_inflowVelocity);

    // 9. Dissipation
    dissipateKernel<<<linearBlocks, 256>>>(d_density, d_obstacles, m_totalCells,
                                            m_densityDissipation);

    // 10. Final obstacle enforcement
    enforceObstaclesKernel<<<linearBlocks, 256>>>(
        d_density, d_densityScratch, d_pressure, d_velocityX, d_velocityY,
        d_velocityXScratch, d_velocityYScratch, d_obstacles, m_totalCells);
}

void FluidFieldCuda::addDensity(int x, int y, float amount) {
    int idx = x + y * m_size;
    if (idx < 0 || idx >= m_totalCells) return;
    if (h_obstacles[static_cast<std::size_t>(idx)] != 0) return;

    float current = 0.0f;
    CUDA_CHECK(cudaMemcpy(&current, d_density + idx, sizeof(float), cudaMemcpyDeviceToHost));
    current += amount;
    CUDA_CHECK(cudaMemcpy(d_density + idx, &current, sizeof(float), cudaMemcpyHostToDevice));
}

void FluidFieldCuda::addVelocity(int x, int y, float amountX, float amountY) {
    int idx = x + y * m_size;
    if (idx < 0 || idx >= m_totalCells) return;
    if (h_obstacles[static_cast<std::size_t>(idx)] != 0) return;

    float vx = 0.0f, vy = 0.0f;
    CUDA_CHECK(cudaMemcpy(&vx, d_velocityX + idx, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&vy, d_velocityY + idx, sizeof(float), cudaMemcpyDeviceToHost));
    vx += amountX;
    vy += amountY;
    CUDA_CHECK(cudaMemcpy(d_velocityX + idx, &vx, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velocityY + idx, &vy, sizeof(float), cudaMemcpyHostToDevice));
}

void FluidFieldCuda::clearObstacles() {
    std::fill(h_obstacles.begin(), h_obstacles.end(), static_cast<std::uint8_t>(0));
    uploadObstacles();
}

void FluidFieldCuda::setObstacleCircle(int centerX, int centerY, float radius) {
    float r2 = radius * radius;
    for (int y = 1; y < m_size - 1; ++y) {
        for (int x = 1; x < m_size - 1; ++x) {
            float dx = static_cast<float>(x - centerX);
            float dy = static_cast<float>(y - centerY);
            if (dx * dx + dy * dy <= r2) {
                h_obstacles[static_cast<std::size_t>(x + y * m_size)] = 1;
            }
        }
    }
    uploadObstacles();
}

void FluidFieldCuda::setObstacleAirfoil(int centerX, int centerY, float chord,
                                          float thickness, float angleDegrees) {
    float angleRad = angleDegrees * static_cast<float>(M_PI) / 180.0f;
    float cosA = std::cos(angleRad);
    float sinA = std::sin(angleRad);

    for (int y = 1; y < m_size - 1; ++y) {
        for (int x = 1; x < m_size - 1; ++x) {
            float dx = static_cast<float>(x - centerX);
            float dy = static_cast<float>(y - centerY);

            float rotX = dx * cosA + dy * sinA + 0.5f * chord;
            float rotY = -dx * sinA + dy * cosA;
            float localX = rotX / chord;
            if (localX < 0.0f || localX > 1.0f) continue;

            float t = 5.0f * thickness *
                      (0.2969f * std::sqrt(localX) -
                       0.1260f * localX -
                       0.3516f * localX * localX +
                       0.2843f * localX * localX * localX -
                       0.1036f * localX * localX * localX * localX);

            float halfT = std::max(0.5f, chord * t);
            if (std::abs(rotY) <= halfT) {
                h_obstacles[static_cast<std::size_t>(x + y * m_size)] = 1;
            }
        }
    }
    uploadObstacles();
}

void FluidFieldCuda::setObstacleRectangle(int centerX, int centerY, float width,
                                            float height, float angleDegrees) {
    float angleRad = angleDegrees * static_cast<float>(M_PI) / 180.0f;
    float cosA = std::cos(angleRad);
    float sinA = std::sin(angleRad);
    float halfW = width * 0.5f;
    float halfH = height * 0.5f;

    for (int y = 1; y < m_size - 1; ++y) {
        for (int x = 1; x < m_size - 1; ++x) {
            float dx = static_cast<float>(x - centerX);
            float dy = static_cast<float>(y - centerY);
            float rotX = dx * cosA + dy * sinA;
            float rotY = -dx * sinA + dy * cosA;
            if (std::abs(rotX) <= halfW && std::abs(rotY) <= halfH) {
                h_obstacles[static_cast<std::size_t>(x + y * m_size)] = 1;
            }
        }
    }
    uploadObstacles();
}

void FluidFieldCuda::renderToPixels(std::uint8_t* hostPixels, ViewMode mode) const {
    dim3 grid = gridDim2D(m_size);
    dim3 block = blockDim2D();
    renderKernel<<<grid, block>>>(d_pixels, d_density, d_pressure,
                                   d_velocityX, d_velocityY, d_obstacles,
                                   m_size, static_cast<int>(mode));
    CUDA_CHECK(cudaMemcpy(hostPixels, d_pixels,
                          static_cast<std::size_t>(m_totalCells) * 4,
                          cudaMemcpyDeviceToHost));
}

int FluidFieldCuda::size() const noexcept {
    return m_size;
}

float FluidFieldCuda::densityAt(int x, int y) const {
    float val = 0.0f;
    int idx = x + y * m_size;
    CUDA_CHECK(cudaMemcpy(&val, d_density + idx, sizeof(float), cudaMemcpyDeviceToHost));
    return val;
}

float FluidFieldCuda::pressureAt(int x, int y) const {
    float val = 0.0f;
    int idx = x + y * m_size;
    CUDA_CHECK(cudaMemcpy(&val, d_pressure + idx, sizeof(float), cudaMemcpyDeviceToHost));
    return val;
}

float FluidFieldCuda::velocityXAt(int x, int y) const {
    float val = 0.0f;
    int idx = x + y * m_size;
    CUDA_CHECK(cudaMemcpy(&val, d_velocityX + idx, sizeof(float), cudaMemcpyDeviceToHost));
    return val;
}

float FluidFieldCuda::velocityYAt(int x, int y) const {
    float val = 0.0f;
    int idx = x + y * m_size;
    CUDA_CHECK(cudaMemcpy(&val, d_velocityY + idx, sizeof(float), cudaMemcpyDeviceToHost));
    return val;
}

bool FluidFieldCuda::isObstacleAt(int x, int y) const {
    return h_obstacles[static_cast<std::size_t>(x + y * m_size)] != 0;
}

void FluidFieldCuda::uploadObstacles() {
    CUDA_CHECK(cudaMemcpy(d_obstacles, h_obstacles.data(),
                          static_cast<std::size_t>(m_totalCells) * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice));
}
