#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "Parameters.h"
#include "Sector.h"

#include <vector>

class Environment {

    private:

        double tax_limit = 3.0;
        double last_action = 0.0;

    public: 
        Parameters params;
        Sector sector;

        int t;
        bool done;

        Environment();

        void reset();
        std::tuple<std::vector<double>, double, bool> step(double action);
        std::vector<double> observe();
        double calculateReward(const std::vector<double>& obs, double action, double last_action);
}



#endif //ENVIRONMENT_H