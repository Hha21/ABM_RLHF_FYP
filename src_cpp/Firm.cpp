#include "../include_cpp/Firm.h"

Firm::Firm(const Parameters& p, int firmIndex) :
    j(firmIndex),
    I_d(p.I_d),
    theta(p.theta),
    alpha(p.d[firmIndex]),
    delta(p.a[firmIndex]),
    eta(p.e[firmIndex]),
    exp_mode(p.exp_mode[firmIndex]),
    x(p.exp_x),
    lamb(p.lamb[firmIndex])    {
    
    int T_plus = p.TP + 2;

    // SIZE VECTORS
    this->s.resize(T_plus);
    this->sq.resize(T_plus);
    this->f.resize(T_plus);
    this->e.resize(T_plus);
    this->qg.resize(T_plus);
    this->qg_s.resize(T_plus);
    this->qg_d.resize(T_plus);
    this->qg_I.resize(T_plus);
    this->D.resize(T_plus);
    this->Dl.resize(T_plus);
    this->pg.resize(T_plus);
    this->m.resize(T_plus);
    this->A.resize(T_plus);
    this->B.resize(T_plus);
    this->qp_d.resize(T_plus);
    this->u_i.resize(T_plus);
    this->u_t.resize(T_plus);
    this->cu_t.resize(T_plus);
    this->c_e.resize(T_plus);
    this->c_pr.resize(T_plus);

    this->m[0] = p.m0[firmIndex];                          // initial mark-up rate
    this->pg[0] = p.B0 * (1.0 + p.m0[firmIndex]);          // initial sales price
    this->A[0] = this->A[1] = p.b[firmIndex];              // initial emissions intensity
    this->B[0] = this->B[1] = p.c[firmIndex];              // initial production costs

    std::cout << "FIRM " << this->j << " INIT!" << std::endl;
}

void Firm::setExpectations(const int t) {
    // Set desired output based on expectation mode
    if (this->exp_mode == "trend") {
        if (t > 1) {
            this->qg_d[t] = this->D[t - 1] + this->x * (this->D[t - 1] - this->D[t - 2]);
        } else {
            this->qg_d[t] = this->D[t - 1];
        }
    }
    else if (this->exp_mode == "adaptive") {
        this->qg_d[t] = this->x * this->D[t - 1] + (1.0 - this->x) * this->qg_d[t - 1];
    }
    else { // default to myopic
        this->qg_d[t] = this->D[t - 1];
    }

    // Adjust for inventory
    this->qg_d[t] = std::max(0.0, this->qg_d[t] * (1.0 + this->I_d) - this->qg_I[t - 1]);

    // Set desired mark-up based on share change
    if (t != 1 && this->s[t - 2] > 0.01) {
        this->m[t] = this->m[t - 1] * (1.0 + this->theta * (this->s[t - 1] - this->s[t - 2]) / this->s[t - 2]);
    } else {
        this->m[t] = this->m[t - 1];
    }
    std::cout << "EXPECTATION SET FIRM " << this->j << std::endl;
}

double Firm::production(const int t, const double pe) {
    // Step 1: Produce as planned
    this->qg[t] = this->qg_d[t];

    // Step 2: Calculate emissions
    this->e[t] = this->qg[t] * this->A[t];

    // Step 3: Update inventory
    this->qg_I[t] = this->qg_I[t - 1] + this->qg[t];

    // Step 4: Calculate product price
    this->pg[t] = std::max(0.0, (this->A[t] * pe + this->B[t]) * (1.0 + this->m[t]));

    // Step 5: Return tax amount = emissions × permit price
    return this->e[t] * pe;
}

void Firm::abatement(const int t, const double pe) {

    double o = 0.0;

    // CHECK IF ABATEMENT OPTIONS AVAILABLE
    if (this->lamb.size() > 0) {
        const std::array<double,2> ab = this->lamb[0];          // BEST OPTION
        const double MAC = ab[1] / ab[0];

        if (MAC * (1 + this->eta) <= pe && this->s[t-1] > 0.01) {
            o = 1.0;
            this->lamb.erase(this->lamb.begin());               //REMOVE USED OPTION
        }
        this->A[t+1] = this->A[t] - o * ab[0];
        this->B[t+1] = this->B[t] + o * ab[1];
    }
}