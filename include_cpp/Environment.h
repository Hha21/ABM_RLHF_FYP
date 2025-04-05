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

// MDP is {OBSERVATIONS, REWARD, DONE}
typedef std::tuple<std::vector<double>, double, bool> MDP;

class Environment {

    private:

        Parameters params;
        Sector sector; 
        
        int t;                                                                      ///< Simulation Clock
        bool done;                                                                  ///< Termination Criteria
        MDP Markov;                                                                 ///< Current MDP

        double tax_limit = 3.0;
        double last_action = 0.0;

        double init_emissions = 0.0;                                                ///< For Reward Signal Calibration
        double init_CC0 = 0.0;

        std::vector<double> tax_actions;                                            ///< All Actions Taken
        std::vector<double> CC0;                                                    ///< Mean Price of Goods
        std::vector<double> CC;                                                     ///< Consumer Impact (Assume Tax Revenue Recycled)

        const std::array<double, 6> max_vals = {1.0, 1.0, 10.0, 1.0, 1.0, 25.0};

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
        double calculateReward(const std::vector<double>& obs, const double action, const double last_action);

    public: 

        Environment();

        MDP step(const double action);

        void outputTxt();

        std::vector<double> reset();

        // GETTERS
        int getTime() {
            return this->t;
        }

        bool getDone() {
            return this->done;
        }

        ActionSpace getActionSpace() {
            return {0.0, this->tax_limit};
        }

        ObservationSpace getObservationSpace() {
            return {
                std::vector<double>(6, 0.0),
                std::vector<double>(this->max_vals.begin(), this->max_vals.end())
            };
        }
};



#endif //ENVIRONMENT_H