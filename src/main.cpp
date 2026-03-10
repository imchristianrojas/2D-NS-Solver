#include "FluidField.hpp"

#include <SFML/Graphics.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

namespace {

enum class ViewMode {
    Density,
    VelocityMagnitude,
};

sf::Color densityColor(float density) {
    const float clamped = std::clamp(density, 0.0F, 255.0F);
    const auto shade = static_cast<std::uint8_t>(clamped);
    return sf::Color(shade, static_cast<std::uint8_t>(shade / 2U), static_cast<std::uint8_t>(255U - shade));
}

sf::Color velocityColor(float vx, float vy) {
    const float speed = std::sqrt((vx * vx) + (vy * vy));
    const float normalized = std::clamp(speed * 48.0F, 0.0F, 255.0F);
    const auto intensity = static_cast<std::uint8_t>(normalized);
    return sf::Color(20U, intensity, static_cast<std::uint8_t>(255U - intensity / 2U));
}

std::string windowTitle(ViewMode mode) {
    if (mode == ViewMode::VelocityMagnitude) {
        return "2D NS Solver - Velocity Magnitude [2]";
    }
    return "2D NS Solver - Density [1]";
}

}

int main() {
    constexpr int gridSize = 128;
    constexpr int windowSize = 768;
    constexpr float deltaTime = 0.1F;

    FluidField field(gridSize, 0.0001F, 0.00001F, deltaTime);
    sf::RenderWindow window(sf::VideoMode({windowSize, windowSize}), "2D NS Solver");
    window.setFramerateLimit(60);

    sf::Image densityImage;
    densityImage.resize(sf::Vector2u(gridSize, gridSize), sf::Color::Black);

    sf::Texture densityTexture;
    if (!densityTexture.resize(sf::Vector2u(gridSize, gridSize))) {
        return 1;
    }

    sf::Sprite densitySprite(densityTexture);
    const float scale = static_cast<float>(windowSize) / static_cast<float>(gridSize);
    densitySprite.setScale(sf::Vector2f(scale, scale));
    ViewMode currentView = ViewMode::Density;
    window.setTitle(windowTitle(currentView));

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
            }
        }

        const sf::Vector2i mousePixel = sf::Mouse::getPosition(window);
        const bool inject = sf::Mouse::isButtonPressed(sf::Mouse::Button::Left);

        if (inject && mousePixel.x >= 0 && mousePixel.y >= 0 &&
            mousePixel.x < windowSize && mousePixel.y < windowSize) {
            const int cellX = std::clamp(static_cast<int>(mousePixel.x / scale), 1, gridSize - 2);
            const int cellY = std::clamp(static_cast<int>(mousePixel.y / scale), 1, gridSize - 2);
            field.addDensity(cellX, cellY, 80.0F);
            field.addVelocity(cellX, cellY, 0.0F, -2.0F);
        }

        field.step();

        for (int y = 0; y < gridSize; ++y) {
            for (int x = 0; x < gridSize; ++x) {
                sf::Color pixel = sf::Color::Black;
                if (currentView == ViewMode::VelocityMagnitude) {
                    pixel = velocityColor(field.velocityXAt(x, y), field.velocityYAt(x, y));
                } else {
                    pixel = densityColor(field.densityAt(x, y));
                }
                densityImage.setPixel(sf::Vector2u(static_cast<unsigned int>(x), static_cast<unsigned int>(y)), pixel);
            }
        }

        densityTexture.update(densityImage);

        window.clear(sf::Color(10, 12, 18));
        window.draw(densitySprite);
        window.display();
    }

    return 0;
}
