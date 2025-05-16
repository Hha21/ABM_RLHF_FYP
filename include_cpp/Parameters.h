#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// BASELINE RANGE
struct param_range {
    double bounds[16][2] = {{30, 50},           ///< 00 - N - Number of firms
                            {0.1, 0.5},         ///< 01 - γ - Price sensitivity of demand
                            {0.0, 0.4},         ///< 02 - ΔA_0,ΔB_0 - Heterogeneity of production factors
                            {0.1, 0.2},         ///< 03 - e^* - Emission target
                            {0.2, 0.4},         ///< 04 - m0 - Initial mark-up
                            {0.04, 0.2},        ///< 05 - ϑ - Mark-up adaption rate
                            {0.025, 0.15},      ///< 06 - χ - Market-share adaption rate
                            {0.2, 5.0},         ///< 07 - ω_1/ω_2 - Market-share weight difference
                            {0.05, 0.3},        ///< 08 - δ - Permit price adaption rate
                            {0.0, 0.4},         ///< 09 - Δδ - Heterogeneity of above
                            {0.17, 0.87},       ///< 10 - α_{pot} - Abatement potential
                            {1.0, 10.0},        ///< 11 - α_{costs} - Abatement cost factor
                            {0.0, 0.4},         ///< 12 - Δα_{costs} - Heterogeneity of above
                            {0.0, 0.4},         ///< 13 - η - Investment profitability target
                            {0.0, 0.4},         ///< 14 - Δη - Heterogeneity of above
                            {0.0, 3.0}};        ///< 15 - μ_1,μ_2,μ_3 - Expectation Rule
};

enum class TechScenario { AVERAGE, OPTIMISTIC, PESSIMISTIC };

class Parameters {

    private:

        param_range range;
        TechScenario scenario;

        void applyTechScenario(TechScenario Scenario);

        double spread = 0.01;  ///< Fractional spread, e.g. 0.1 = +/- 10% of midpoint

    public:
    
        int NP = 31;            // Num Periods
        int t_start = 1;       
        int t_period = 10;      //Length of one regulation period
        int t_impl = 10;        //Num. implementation periods

        double D0 = 1.0;
        double A0 = 1.0;
        double B0 = 1.0;

        int lamb_n = 20;
        double I_d = 0.1;

        int T = this->NP * this->t_period;

        int N;

        std::string mode = "Tax";
        bool emission_tax = true;

        double exp_x_trend[2] = {0.5, 1.0};
        double exp_x_adaptive[2] = {0.25, 0.75};

        double calibration_threshold = 10e-3;
        int calibration_max_runs = 20;

        double tax = 100.0;

        // FIRM RANDOM PARAMS:
        std::vector<double> a;
        std::vector<double> b;
        std::vector<double> c;
        std::vector<double> d;
        std::vector<double> e;

        std::vector<std::vector<std::array<double, 2>>> lamb;


        // VARIABLE PARAMETERS:
        double gamma;
        double delAB;
        double E_max;
        std::vector<double> m0;
        double theta;
        double chi;
        std::array<double, 2> omg;
        double delta;
        double delDelta;
        double lamb_max;
        double alpha;
        double delAlpha;
        double eta;
        double delEta;
        std::vector<std::string> exp_mode;
        std::vector<double> exp_x;


        // RNG
        std::mt19937 rng;

        Parameters(const std::string tech_scenario = "AVERAGE", const int fixed_seed = -1);
        void generateRandomPar();

        inline double random_val(int idx);
};


#endif //PARAMETERS_H