#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "Parameters.h"
#include "Firm.h"
#include "Sector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <vector>

typedef std::tuple<std::vector<double>, double, bool> MDP;

class Environment {

    private:

        double tax_limit = 3.0;
        double last_action = 0.0;

        std::vector<double> tax_actions;

        const std::array<double, 6> max_vals = {1.0, 1.0, 10.0, 1.0, 1.0, 25.0};

        std::ofstream emissionsVsTax;

    public: 
        Parameters params;
        Sector sector;

        MDP Markov;

        int t;
        bool done;

        Environment();

        MDP step(const double action);
        std::vector<double> observe();
        double calculateReward(const std::vector<double>& obs, const double action, const double last_action);

        void outputTxt();

        std::vector<double> reset();
};



#endif //ENVIRONMENT_H