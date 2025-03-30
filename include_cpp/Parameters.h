#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct param_range {
    int num_vars = 17;

    std::string names[20] = {"N", "gamma", "delA_0",
                        "delB_0", "e^*", "m_0",
                        "theta", "chi", "omg_1/omg_2"
                        "delta", "delDelta", "alpha_pot",
                        "alpha_costs", "delAlpha_costs",
                        "eta", "delEta", "psi", "mu_1",
                        "mu_2", "mu_3"};

    double bounds[17][2] = {{30, 50},           ///< 00 - N - Number of firms
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
                            {0.0, 2.0},         ///< 15 - ψ - Auction mechanism
                            {0.0, 3.0}};        ///< 16 - μ_1,μ_2,μ_3 - Expectation Rule
};

class Parameters {

    private:

        param_range range;

    public:
    
        const int TP = 100;
        const int t_start = 10;
        const int t_period = 10;
        const int t_impl = 30;
        const int D0 = 1;
        const int A0 = 1;
        const int B0 = 1;
        const int lamb_n = 20;
        const double I_d = 0.1;

        const std::string mode = "Tax";
        const bool emission_tax = true;

        const double exp_x_trend[2] = {0.5, 1.0};
        const double exp_x_adaptive[2] = {0.25, 0.75};

        const double calibration_threshold = 10e-3;
        const int calibration_max_runs = 20;

        const double tax = 100.0;

        // FIRM RANDOM PARAMS:
        std::vector<double> a;
        std::vector<double> b;
        std::vector<double> c;
        std::vector<double> d;
        std::vector<double> e;

        std::vector<std::vector<std::array<double, 2>>> lamb;


        // VARIABLE PARAMETERS:

        int N;
        double gamma;
        double delAB;
        double E_max;
        std::vector<double> m0;
        double theta;
        double chi;
        std::vector<double> omg;
        double delta;
        double delDelta;
        double lamb_max;
        double alpha;
        double delAlpha;
        double eta;
        double delEta;
        std::string ex_mode;
        std::vector<std::string> exp_mode;
        double exp_x;


        // RNG
        std::mt19937 rng;

        Parameters();
        void generateRandomPar();

        inline double random_val(int idx);
};


#endif //PARAMETERS_H