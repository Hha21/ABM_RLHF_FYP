#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "Parameters.h"
#include "Firm.h"
#include "Sector.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

// MDP is {OBSERVATIONS, REWARD1, REWARD2, DONE}
typedef std::tuple<std::vector<double>, double, double, bool> MDP;

class Environment {

    private:

        Parameters params;
        Sector sector; 
        
        int t;                                                                      ///< Simulation Clock
        bool done;                                                                  ///< Termination Criteria
        MDP Markov;                                                                 ///< Current MDP

        // FOR PARAMS
        const std::string techno_mode;
        const int seed;

        const double chi;

        const double tax_limit = 5.0;
        double last_action = 0.0;
        double new_action = 0.0;

        double init_emissions = 0.0;                                                ///< For Reward Signal Calibration
        double init_CC0 = 0.0;

        std::vector<double> tax_actions;                                            ///< All Actions Taken
        std::vector<double> CC0;                                                    ///< Mean Price of Goods
        std::vector<double> CC;                                                     ///< Consumer Impact (Assume Tax Revenue Recycled)

        // FOR REWARD FUNCTION
        const double emissions_target;
        
        const int observation_dim = (this->params.N) * (4) + (3);

        // ACTION LOOK-UP
        const std::array<double, 10> action_table = {-0.2, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.2, 0.3};

        std::ofstream emissionsVsTax;

        struct ActionSpace {
            double low;
            double high;
        };

        struct ObservationSpace {
            std::vector<double> low;
            std::vector<double> high;
        };

        std::vector<double> observe();

        std::array<double, 2> calculateReward(const std::vector<double>& obs);

    public: 

        Environment(std::string TECH_MODE = "AVERAGE", int seed_ = -1, double target_ = 0.2, double chi_ = 0.5);

        // DISCRETE ACTION
        MDP step(const int action_idx);

        // CONTINUOUS ACTION
        MDP step(const double action_cont);

        void outputTxt();

        std::vector<double> reset();

        // GETTERS
        int getTime() {
            return this->t;
        }

        bool getDone() {
            return this->done;
        }

        // ActionSpace getActionSpace() {
        //     return {0.0, this->tax_limit};
        // }

        // ObservationSpace getObservationSpace() {
        //     return {
        //         std::vector<double>(6, 0.0),
        //         std::vector<double>(this->max_vals.begin(), this->max_vals.end())
        //     };
        // }
};



#endif //ENVIRONMENT_H