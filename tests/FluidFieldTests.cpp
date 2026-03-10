#include "FluidField.hpp"

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
    FluidField field(32, 0.0001F, 0.00001F, 0.1F);
    field.step();

    expect(field.velocityXAt(0, 16) > 0.5F, "left boundary should have positive inflow velocity");
    expect(field.densityAt(1, 16) > 0.0F, "inflow should seed density into the tunnel");
}

void testObstacleRejectsInjectedState() {
    FluidField field(32, 0.0001F, 0.00001F, 0.1F);
    field.setObstacleCircle(16, 16, 3.5F);

    expect(field.isObstacleAt(16, 16), "expected obstacle cell at test center");

    field.addDensity(16, 16, 100.0F);
    field.addVelocity(16, 16, 5.0F, -2.0F);
    field.step();

    expect(field.densityAt(16, 16) == 0.0F, "obstacle cells must not retain density");
    expect(field.velocityXAt(16, 16) == 0.0F, "obstacle cells must zero horizontal velocity");
    expect(field.velocityYAt(16, 16) == 0.0F, "obstacle cells must zero vertical velocity");
    expect(field.pressureAt(16, 16) == 0.0F, "obstacle cells must zero pressure");
}

void testStepKeepsFieldsFinite() {
    FluidField field(48, 0.0001F, 0.00001F, 0.1F);
    field.setObstacleAirfoil(24, 24, 14.0F, 0.14F, 8.0F);
    field.addDensity(8, 24, 180.0F);
    field.addVelocity(8, 24, 4.0F, 0.5F);

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

}

int main() {
    testInflowSeedsVelocityAndDensity();
    testObstacleRejectsInjectedState();
    testStepKeepsFieldsFinite();

    std::cout << "All FluidField tests passed\n";
    return 0;
}
