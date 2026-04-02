#include "FluidFieldCuda.cuh"

#include <SFML/Graphics.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace {

std::string windowTitle(ViewMode mode) {
    if (mode == ViewMode::Pressure) {
        return "2D NS Solver [CUDA] - Pressure [3], Airfoil Angle [[] / ]";
    }
    if (mode == ViewMode::VelocityMagnitude) {
        return "2D NS Solver [CUDA] - Velocity Magnitude [2], Airfoil Angle [[] / ]";
    }
    return "2D NS Solver [CUDA] - Density [1], Airfoil Angle [[] / ]";
}

void rebuildAirfoil(FluidFieldCuda& field, int gridSize, float angleDegrees) {
    field.clearObstacles();
    field.setObstacleAirfoil(
        static_cast<int>(gridSize * 0.53f),
        gridSize / 2,
        static_cast<float>(gridSize) * 0.26f,
        0.14f,
        angleDegrees
    );
}

sf::VertexArray buildVelocityGlyphs(const FluidFieldCuda& field, float scale, int gridSize) {
    int sampleStride = std::max(6, gridSize / 24);
    constexpr float glyphScale = 10.0f;
    sf::VertexArray glyphs(sf::PrimitiveType::Lines);

    for (int y = 2; y < gridSize - 2; y += sampleStride) {
        for (int x = 2; x < gridSize - 2; x += sampleStride) {
            if (field.isObstacleAt(x, y)) continue;

            float vx = field.velocityXAt(x, y);
            float vy = field.velocityYAt(x, y);
            float speed = std::sqrt(vx * vx + vy * vy);
            if (speed < 0.05f) continue;

            sf::Vector2f origin((static_cast<float>(x) + 0.5f) * scale,
                                (static_cast<float>(y) + 0.5f) * scale);
            sf::Vector2f tip(origin.x + vx * glyphScale * scale,
                             origin.y + vy * glyphScale * scale);
            auto alpha = static_cast<std::uint8_t>(std::clamp(speed * 120.0f, 40.0f, 180.0f));
            sf::Color color(248, 248, 248, alpha);
            glyphs.append(sf::Vertex(origin, color));
            glyphs.append(sf::Vertex(tip, color));
        }
    }
    return glyphs;
}

}

int main() {
    constexpr int gridSize = 512;
    constexpr int windowSize = 1024;
    constexpr float deltaTime = 0.1f;

    FluidFieldCuda field(gridSize, 0.0001f, 0.00001f, deltaTime);
    float airfoilAngleDegrees = 6.0f;
    rebuildAirfoil(field, gridSize, airfoilAngleDegrees);

    sf::RenderWindow window(sf::VideoMode({windowSize, windowSize}), "2D NS Solver [CUDA]");
    window.setFramerateLimit(0); // uncapped — let the GPU rip

    // RGBA pixel buffer for GPU rendering
    std::vector<std::uint8_t> pixelBuffer(static_cast<std::size_t>(gridSize) * gridSize * 4);

    sf::Image fieldImage;
    fieldImage.resize(sf::Vector2u(gridSize, gridSize), sf::Color::Black);

    sf::Texture fieldTexture;
    if (!fieldTexture.resize(sf::Vector2u(gridSize, gridSize))) {
        return 1;
    }

    sf::Sprite fieldSprite(fieldTexture);
    float scale = static_cast<float>(windowSize) / static_cast<float>(gridSize);
    fieldSprite.setScale(sf::Vector2f(scale, scale));

    ViewMode currentView = ViewMode::Density;
    window.setTitle(windowTitle(currentView));

    sf::Clock fpsClock;
    int frameCount = 0;

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->code == sf::Keyboard::Key::Num1) {
                    currentView = ViewMode::Density;
                    window.setTitle(windowTitle(currentView));
                }
                if (keyPressed->code == sf::Keyboard::Key::Num2) {
                    currentView = ViewMode::VelocityMagnitude;
                    window.setTitle(windowTitle(currentView));
                }
                if (keyPressed->code == sf::Keyboard::Key::Num3) {
                    currentView = ViewMode::Pressure;
                    window.setTitle(windowTitle(currentView));
                }
                if (keyPressed->code == sf::Keyboard::Key::LBracket) {
                    airfoilAngleDegrees = std::clamp(airfoilAngleDegrees - 2.0f, -20.0f, 20.0f);
                    rebuildAirfoil(field, gridSize, airfoilAngleDegrees);
                }
                if (keyPressed->code == sf::Keyboard::Key::RBracket) {
                    airfoilAngleDegrees = std::clamp(airfoilAngleDegrees + 2.0f, -20.0f, 20.0f);
                    rebuildAirfoil(field, gridSize, airfoilAngleDegrees);
                }
            }
        }

        // Mouse injection
        const sf::Vector2i mousePixel = sf::Mouse::getPosition(window);
        const bool inject = sf::Mouse::isButtonPressed(sf::Mouse::Button::Left);
        if (inject && mousePixel.x >= 0 && mousePixel.y >= 0 &&
            mousePixel.x < windowSize && mousePixel.y < windowSize) {
            int cellX = std::clamp(static_cast<int>(mousePixel.x / scale), 1, gridSize - 2);
            int cellY = std::clamp(static_cast<int>(mousePixel.y / scale), 1, gridSize - 2);
            field.addDensity(cellX, cellY, 80.0f);
            field.addVelocity(cellX, cellY, 0.0f, -2.0f);
        }

        // Simulation step (all on GPU)
        field.step();

        // GPU renders pixels, single cudaMemcpy to host
        field.renderToPixels(pixelBuffer.data(), currentView);

        // Copy RGBA buffer into SFML image
        for (int y = 0; y < gridSize; ++y) {
            for (int x = 0; x < gridSize; ++x) {
                std::size_t pi = static_cast<std::size_t>((y * gridSize + x) * 4);
                fieldImage.setPixel(
                    sf::Vector2u(static_cast<unsigned>(x), static_cast<unsigned>(y)),
                    sf::Color(pixelBuffer[pi], pixelBuffer[pi + 1],
                              pixelBuffer[pi + 2], pixelBuffer[pi + 3]));
            }
        }

        fieldTexture.update(fieldImage);

        // Velocity glyphs (still CPU-side — sparse, cheap)
        sf::VertexArray velocityGlyphs = buildVelocityGlyphs(field, scale, gridSize);

        window.clear(sf::Color(10, 12, 18));
        window.draw(fieldSprite);
        if (currentView != ViewMode::VelocityMagnitude) {
            window.draw(velocityGlyphs);
        }
        window.display();

        // FPS counter in title
        frameCount++;
        if (fpsClock.getElapsedTime().asSeconds() >= 1.0f) {
            float fps = static_cast<float>(frameCount) / fpsClock.getElapsedTime().asSeconds();
            std::string title = windowTitle(currentView) +
                                " | " + std::to_string(static_cast<int>(fps)) + " FPS";
            window.setTitle(title);
            frameCount = 0;
            fpsClock.restart();
        }
    }

    return 0;
}
