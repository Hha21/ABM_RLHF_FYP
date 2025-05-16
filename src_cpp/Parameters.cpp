#include "../include_cpp/Parameters.h"

inline double Parameters::random_val(int idx) {

    // Tech-Optimism params
    if (idx == 10 || idx == 11 || idx == 12) {
        std::uniform_real_distribution<> dist(this->range.bounds[idx][0], this->range.bounds[idx][1]);
        return dist(this->rng);
    }

    // For all other params, use spread around midpoint:
    const double min = this->range.bounds[idx][0];
    const double max = this->range.bounds[idx][1];
    double midpoint = 0.5 * (min + max);
    double half_spread = 0.5 * (max - min) * this->spread;

    double lower = midpoint - half_spread;
    double upper = midpoint + half_spread;

    std::uniform_real_distribution<> dist(lower, upper);
    return dist(this->rng);
}

void Parameters::applyTechScenario(TechScenario scenario) {

    // OPTIMISTIC, PESSIMISTIC, AVERAGE TECH SCENARIOS.

    switch (scenario) {
        case TechScenario::OPTIMISTIC: {
            this->range.bounds[10][0] = 0.8; this->range.bounds[10][1] = 0.87;      ///< α_{pot} higher potential
            this->range.bounds[11][0] = 1.0; this->range.bounds[11][1] = 2.0;       ///< α_{costs} lower cost
            this->range.bounds[12][0] = 0.0; this->range.bounds[12][1] = 0.05;       ///< Δα_{costs} less heterogeneity
            break;
        }
        case TechScenario::PESSIMISTIC: {
            this->range.bounds[10][0] = 0.17; this->range.bounds[10][1] = 0.20;     ///< α_{pot} lower potential
            this->range.bounds[11][0] = 9.0; this->range.bounds[11][1] = 10.0;      ///< α_{costs} higher cost
            this->range.bounds[12][0] = 0.35; this->range.bounds[12][1] = 0.4;       ///< Δα_{costs} more heterogeneity
        }
        // ELSE KEEP PARAM RANGE AS IS
        case TechScenario::AVERAGE: 
        default:
            break;
    }

    std::cout << "Tech scenario set: ";
    if (scenario == TechScenario::OPTIMISTIC) std::cout << "OPTIMISTIC\n";
    else if (scenario == TechScenario::PESSIMISTIC) std::cout << "PESSIMISTIC\n";
    else std::cout << "AVERAGE\n";
}

Parameters::Parameters(const std::string tech_scenario, const int fixed_seed) {

    // SET TECH SCENARIO:
    if (tech_scenario == "OPTIMISTIC" || tech_scenario == "optimistic") {
        this->scenario = TechScenario::OPTIMISTIC;
    } else if (tech_scenario == "PESSIMISTIC" || tech_scenario == "pessimistic") {
        this->scenario = TechScenario::PESSIMISTIC;
    } else {
        this->scenario = TechScenario::AVERAGE;
    }
    this->applyTechScenario(this->scenario);

    // SET RNG:

    if (fixed_seed == -1) {
        std::random_device rd;
        this->rng = std::mt19937(rd());
    } else {
        this->rng = std::mt19937(fixed_seed);
        std::cout << "Fixed seed: " << fixed_seed << std::endl;
    }

    // GENERATE RANDOM VAL IN BOUNDS
    //this->N = static_cast<int>(std::round(Parameters::random_val(0)));
    this->N = 50;
    this->gamma = Parameters::random_val(1);
    this->delAB = Parameters::random_val(2);
    this->E_max = Parameters::random_val(3);

    double m0_base = Parameters::random_val(4);
    this->m0 = std::vector<double>(N, m0_base);

    this->theta = Parameters::random_val(5);
    this->chi = Parameters::random_val(6);

    double dOmg = Parameters::random_val(7);
    this->omg = {dOmg / (dOmg + 1.0), 1.0 / (dOmg + 1.0)};

    this->delta = Parameters::random_val(8);
    this->delDelta = Parameters::random_val(9);
    this->lamb_max = Parameters::random_val(10);
    this->alpha = Parameters::random_val(11);
    this->delAlpha = Parameters::random_val(12);
    this->eta = Parameters::random_val(13);
    this->delEta = Parameters::random_val(14);

    this->exp_mode.resize(this->N);
    this->exp_x.resize(this->N);
    double exp_mode_val; 

    for (int i = 0; i < this->N; ++i) {
        exp_mode_val = Parameters::random_val(15);
        if (exp_mode_val < 1.0) {
            this->exp_mode[i] = "trend";
            this->exp_x[i] = this->exp_x_trend[0] + (this->exp_x_trend[1] - this->exp_x_trend[0]) * (exp_mode_val); // - 1.0 TYPO!
        } else if (exp_mode_val < 2.0) {
            this->exp_mode[i] = "myopic";
            this->exp_x[i] = 0.0;
        } else {
            this->exp_mode[i] = "adaptive";
            this->exp_x[i] = this->exp_x_adaptive[0] + (this->exp_x_adaptive[1] - this->exp_x_adaptive[0]) * (exp_mode_val - 2.0);
        }
    }

    Parameters::generateRandomPar();

    //std::cout << "INIT WITH " << this->N << " FIRMS!" << std::endl;
}

void Parameters::generateRandomPar() {

    std::uniform_real_distribution<> uniform_dist(-0.5, 0.5);

    this->a.resize(N);
    this->b.resize(N);
    this->c.resize(N);
    this->d.resize(N);
    this->e.resize(N);
    this->lamb.resize(N);

    double a_step;

    for (int i = 0; i < this->N; ++i) {
        double randDelta = uniform_dist(rng);
        double randAB = uniform_dist(rng);
        double randAlpha = uniform_dist(rng);
        double randEta = uniform_dist(rng);

        double bVal = A0 * (1.0 + delAB * randAB);
        double dVal = alpha * (1.0 + delAlpha * randAlpha);

        this->a[i] = delta * (1.0 + delDelta * randDelta);
        this->b[i] = bVal;
        this->c[i] = B0 * (1.0 + delAB * randAB);
        this->d[i] = dVal;
        this->e[i] = eta * (1.0 + delEta * randEta);

        this->lamb[i].clear();
        this->lamb[i].reserve(this->lamb_n);

        a_step = (bVal * this->lamb_max) / (this->lamb_n);

        for (int j = 0; j < this->lamb_n; ++j) {
            double MAC = a_step * dVal * (j + 1);
            double b_j = a_step * MAC;
            this->lamb[i].push_back({a_step, b_j}); 
        }
    }
}