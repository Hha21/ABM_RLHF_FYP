#include "../include_cpp/Parameters.h"

inline double Parameters::random_val(int idx) {
    std::uniform_real_distribution<> dist(this->range.bounds[idx][0], this->range.bounds[idx][1]);
    return dist(rng);
}

Parameters::Parameters() {

    std::random_device rd;
    rng = std::mt19937(rd());
    
    // GENERATE RANDOM VAL IN BOUNDS
    this->N = static_cast<int>(std::round(Parameters::random_val(0)));
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

    double ex_mode_val = Parameters::random_val(15);
    double exp_mode_val = Parameters::random_val(16);

    // ex_mode HANDLING

    if (ex_mode_val <= 1.0) {
        this->ex_mode = "uniform";
    } else {
        this->ex_mode = "discriminate";
    }

    if (exp_mode_val < 1.0) {
        this->exp_mode = std::vector<std::string>(N, "trend");
        this->exp_x = this->exp_x_trend[0] + (this->exp_x_trend[1] - this->exp_x_trend[0]) * (exp_mode_val - 1.0);
    } else if (exp_mode_val < 2.0) {
        this->exp_mode = std::vector<std::string>(N, "myopic");
        this->exp_x = 0.0;
    } else {
        this->exp_mode = std::vector<std::string>(N, "adaptive");
        this->exp_x = this->exp_x_adaptive[0] + (this->exp_x_adaptive[1] - this->exp_x_adaptive[0]) * (exp_mode_val - 2.0);
    }

    Parameters::generateRandomPar();

    for (int i = 0; i < this->lamb_n; ++i) {
        int firm = 0;
        std::cout << "ABATEMENT FOR FIRM " << firm << " : [a,b] = [" << this->lamb[0][i][0] << " ," << this->lamb[0][i][1] << "] " << std::endl;
    }

    std::cout << "INIT WITH " << this->N << " FIRMS!" << std::endl;
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