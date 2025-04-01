#include "../include_cpp/Sector.h"

Sector::Sector(Parameters& p) {

    this->params = &p;

    // INIT FIRMS
    for (int j = 0; j < this->params->N; ++j) {
        this->firms.emplace_back(*this->params, j);
    }

    const int T_plus = this->params->T + 2;
    this->D.resize(T_plus, 0.0);
    this->E.resize(T_plus, 0.0);
    this->Q.resize(T_plus, 0.0);
    this->u_t.resize(T_plus, 0.0);

    const double initial_share =  1.0 / (this->params->N);
    double price_weighted_sum = 0.0;

    for (Firm& firm : this->firms) {
        firm.s[0] = initial_share;
        price_weighted_sum += initial_share * firm.pg[0];
    }

    this->D[0] = this->params->D0 * std::exp(-price_weighted_sum * this->params->gamma);
    //std::cout << "INITIAL DEMAND: " << this->D[0] << std::endl;
    //ALLOCATE INITIAL DEMAND
    for (Firm& firm : this->firms) {
        firm.D[0] = firm.qg_d[0] = firm.qg[0] = firm.qg_s[0] = firm.s[0] * this->D[0];
    }
}

void Sector::applyExpectations(const int t) {
    for (Firm& firm : this->firms) {
        firm.setExpectations(t);
    }
}

void Sector::applyAbatement(const int t, const double pe) {
    for (Firm& firm : this->firms) {
        firm.abatement(t, pe);
    }
}

void Sector::applyProduction(const int t, const double pe) {
    this->E[t] = 0.0;
    this->Q[t] = 0.0;

    double tax_revenue = 0.0;
    for (Firm& firm : this->firms) {
        tax_revenue += firm.production(t, pe);
        this->E[t] += firm.e[t];
        this->Q[t] += firm.qg[t];
    }

    for (Firm& firm : this->firms) {
        firm.sq[t] = (this->Q[t] > 0.0) ? (firm.qg[t] / this->Q[t]) : (1.0 / params->N);
    }

    this->tax_income = tax_revenue;
}

void Sector::tradeCommodities(const int t) {

    // Step 1 : Calculate Fitness for each Firm, and calculate Mean Fitness
    double f_mean = 0.0;
    for (Firm& firm : this->firms) {
        double fitness_j = - this->params->omg[0] * firm.pg[t] - this->params->omg[1] * firm.Dl[t - 1];
        firm.f[t] = fitness_j;
        f_mean += fitness_j * firm.s[t - 1];
    }

    // Step 2 : Update Market Shares based on fitness and mean fitness, and find Mean Price
    double p_mean = 0.0;
    for (Firm& firm : this->firms) {
        double marketshare_j = std::max(0.0, firm.s[t - 1] * (1.0 - this->params->chi * (firm.f[t] - f_mean) / f_mean));
        firm.s[t] =  marketshare_j;
        p_mean += marketshare_j * firm.pg[t];
    }

    // Step 3 : Calculate Total Demand
    this->D[t] = this->params->D0 * std::exp(-p_mean * this->params->gamma);

    // Step 4 : Allocate Demand and Update Inventories
    for (Firm& firm : this->firms) {
        firm.D[t] = firm.s[t] * this->D[t];
        firm.qg_s[t] = std::min(firm.D[t], firm.qg_I[t]);
        firm.qg_I[t] -= firm.qg_s[t];
        firm.Dl[t] = firm.D[t] - firm.qg_s[t];              // Unfilled Demand
    }

    // Ensure Market Shares sum to 1:
    double share_sum = 0.0;
    for (Firm& firm : this->firms) {
        share_sum += firm.s[t];
    }
    double err = 1.0 - share_sum;
    if (std::abs(err) > 1e-10) {
        //std::cout << "MARKET SHARE SUM: " << share_sum << " ERROR!" << std::endl;
        double correction_factor = 1.0 / (1.0 - err);
        for (Firm& firm : this->firms) {
            firm.s[t] *= correction_factor;
        }
    }
}