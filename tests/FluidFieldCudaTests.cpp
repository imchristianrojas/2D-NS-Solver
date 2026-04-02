#include "FluidFieldCuda.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failure: " << message << '\n';
        std::exit(1);
    }
}

void testInflowSeedsVelocityAndDensity() {
    FluidFieldCuda field(32, 0.0001f, 0.00001f, 0.1f);
    field.step();

    expect(field.velocityXAt(0, 16) > 0.5f, "left boundary should have positive inflow velocity");
    expect(field.densityAt(1, 16) > 0.0f, "inflow should seed density into the tunnel");
}

void testObstacleRejectsInjectedState() {
    FluidFieldCuda field(32, 0.0001f, 0.00001f, 0.1f);
    field.setObstacleCircle(16, 16, 3.5f);

    expect(field.isObstacleAt(16, 16), "expected obstacle cell at test center");

    field.addDensity(16, 16, 100.0f);
    field.addVelocity(16, 16, 5.0f, -2.0f);
    field.step();

    expect(field.densityAt(16, 16) == 0.0f, "obstacle cells must not retain density");
    expect(field.velocityXAt(16, 16) == 0.0f, "obstacle cells must zero horizontal velocity");
    expect(field.velocityYAt(16, 16) == 0.0f, "obstacle cells must zero vertical velocity");
    expect(field.pressureAt(16, 16) == 0.0f, "obstacle cells must zero pressure");
}

void testStepKeepsFieldsFinite() {
    FluidFieldCuda field(48, 0.0001f, 0.00001f, 0.1f);
    field.setObstacleAirfoil(24, 24, 14.0f, 0.14f, 8.0f);
    field.addDensity(8, 24, 180.0f);
    field.addVelocity(8, 24, 4.0f, 0.5f);

    for (int step = 0; step < 12; ++step) {
        field.step();
    }

    for (int y = 0; y < field.size(); ++y) {
        for (int x = 0; x < field.size(); ++x) {
            expect(std::isfinite(field.densityAt(x, y)), "density must remain finite");
            expect(std::isfinite(field.pressureAt(x, y)), "pressure must remain finite");
            expect(std::isfinite(field.velocityXAt(x, y)), "horizontal velocity must remain finite");
            expect(std::isfinite(field.velocityYAt(x, y)), "vertical velocity must remain finite");
        }
    }
}

void testRectangleObstacle() {
    FluidFieldCuda field(32, 0.0001f, 0.00001f, 0.1f);
    field.setObstacleRectangle(16, 16, 6.0f, 4.0f, 0.0f);

    expect(field.isObstacleAt(16, 16), "center of rectangle should be obstacle");
    expect(!field.isObstacleAt(0, 0), "corner should not be obstacle");

    field.step();
    expect(field.densityAt(16, 16) == 0.0f, "obstacle cells must not retain density");
}

}

int main() {
    testInflowSeedsVelocityAndDensity();
    testObstacleRejectsInjectedState();
    testStepKeepsFieldsFinite();
    testRectangleObstacle();

    std::cout << "All FluidFieldCuda tests passed\n";
    return 0;
}
