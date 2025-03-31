#include <iostream>
#include "../include_cpp/Parameters.h"
#include "../include_cpp/Firm.h"


int main() {

    std::cout << "Hello World!" << std::endl;

    Parameters param = Parameters();

    Firm firm1 = Firm(param, 0);

    double revenue = 0;

    for (int t = 0; t < 10; ++t) {
        firm1.setExpectations(t);
        revenue += firm1.production(t, 1.0);
        firm1.abatement(t, 1.0);
    }

    std::cout << "REVENUE: " << revenue << std::endl;
    return 0;
}