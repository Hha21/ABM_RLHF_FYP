#include "../include_cpp/Environment.h"

// FROM Prosumer Preferences Regarding the
// Adoption of Micro-Generation Technologies:
// Empirical Evidence for German Homeowners -- TABLE 5

static inline double interpolate_model_1_1(double rel_C02_reduction, double rel_price_increase) {
    // LINEAR INTERPOLATION

    //C02 reduction interpolation
    double C02_SCORE;
    if (rel_C02_reduction <= 0.5) {                                                 //0->0.5
        C02_SCORE = 0.41 * (rel_C02_reduction / 0.5);                           
    } else if (rel_C02_reduction <= 1.0) {                                          //0.5->1.0
        C02_SCORE = 0.41 + (0.56 - 0.41) * ((rel_C02_reduction - 0.5) / 0.5);
    } else if (rel_C02_reduction <= 2.0) {                                          //1.0->2.0
        C02_SCORE = 0.56 + (0.60 - 0.56) * ((rel_C02_reduction - 1.0)); // (/1.0)
    } else {
        if (rel_C02_reduction > 0.0) {
            C02_SCORE = 0.60;
        } else {
            C02_SCORE = 0.0;
        }
    }

    //Price increase interpolation
    double PRICE_SCORE;
    if (rel_price_increase <= 0.6) {                                                //0->0.6
        PRICE_SCORE = 1.09 * (rel_price_increase / 0.6);
    } else if (rel_price_increase <= 0.8) {                                         //0.6->0.8
        PRICE_SCORE = 1.09 + (0.58 - 1.09) * ((rel_price_increase - 0.6) / 0.2);
    } else {
        if (rel_price_increase > 0.0) {
            PRICE_SCORE = -0.88;
        } else {
            PRICE_SCORE = 0.0;
        }
    }

    return C02_SCORE + PRICE_SCORE;
}

std::tuple<std::vector<double>, double, bool> Environment::step(const double action) {

    // double new_action = std::clamp(this->last_action + action, 0.0, this->tax_limit);

    double new_action;

    // ENSURE WITHIN LIMITS
    if (this->last_action + action > this->tax_limit) {
        new_action = this->tax_limit;
    } else if (this->last_action + action < 0.0) {
        new_action = 0.0;
    } else {
        new_action = this->last_action + action;
    }

    int period = this->params.t_period;

    // FOR ONE REGULATION PERIOD
    for (int i = 0; i < period && this->t <= this->params.T; ++i, ++this->t) {
        this->tax_actions.push_back(new_action);

        // MAIN DYNAMICS
        this->sector.applyExpectations(t);
        this->sector.applyAbatement(t, new_action);
        this->sector.applyProduction(t, new_action);
        this->sector.tradeCommodities(t);

        // MEASURES
        double price_goods = 0.0;
        for (const Firm& firm : this->sector.firms) {
                price_goods += firm.s[this->t] * firm.pg[this->t];
        }

        this->CC0.push_back(price_goods);

        double total_sold = this->sector.Q_s[t];
        double total_revenue = this->sector.R[t];

        double consumer_impact = price_goods - (total_sold > 0.0 ? total_revenue / total_sold : 0.0);
        this->CC.push_back(consumer_impact);

        // FOR INITIALISING VALUES
        if (this->t <= 10) {
            this->init_emissions += this->sector.E[this->t];
            this->init_CC0 += price_goods;
        }
        // MEAN
        if (this->t == 10) {
            std::cout << "MODEL CALIBRATION t=" << this->t << std::endl;
            this->init_emissions /= this->params.t_period;
            this->init_CC0 /= this->params.t_period;
            std::cout << "INIT EMISSIONS, CC0 = [" << this->init_emissions << " ," << this->init_CC0 << "] " << std::endl;
        }
    }


    // GET OBSERVATIONS
    std::vector<double> next_obs = Environment::observe();

    // GET REWARD
    double reward = Environment::calculateReward(next_obs, new_action, last_action);
    last_action = new_action;
    
    done = (t > this->params.T);

    // if (done) {
    //     Environment::outputTxt();
    // }

    // RETURN MDP
    this->Markov = {next_obs, reward, done};
    return this->Markov;
}

std::vector<double> Environment::observe() {
    // int period = this->params.t_period;
    // int start = this->t - period;

    // double LE = this->sector.E[this->t - 1];
    // double LT = this->last_action;

    // // TOTAL EMISSIONS IN PERIOD
    // double E_period = 0.0;

    // // HHI MARKET CONCENTRATION
    // double HHI = 0.0;

    // // Avg. PROFIT RATE (PL), Avg. SALES PRICE (CC0)
    // double numerator_PL = 0.0, denominator_PL = 0.0, CC0 = 0.0;

    // for (int i = start; i < this->t; ++i) {
    //     E_period += this->sector.E[i];
    //     for (const Firm& firm : this->sector.firms) {
    //         HHI += firm.s[i] * firm.s[i];
    //         double unit_cost = this->last_action * firm.A[i] + firm.B[i];
    //         numerator_PL += firm.qg_s[i] * (firm.pg[i] - unit_cost);
    //         denominator_PL += firm.qg[i] * unit_cost;
    //         CC0 += firm.s[i] * firm.pg[i];
    //     }
    // }
    // double PL = (denominator_PL > 0) ? numerator_PL / denominator_PL : 0.0;

    // std::vector<double> obs = {LE, LT, E_period, HHI, PL, CC0};

    // for (int i = 0; i < 6; ++i) {
    //     obs[i] /= this->max_vals[i];
    // }

    int period = this->params.t_period;
    int start = this->t - period;

    double E_period = 0.0;
    double price_goods = 0.0;
    double goods_sold = 0.0;
    double tax_revenue = 0.0;

    for (int i = start; i < this->t; ++i) {
        E_period += this->sector.E[i];
        goods_sold += this->sector.Q_s[i];
        tax_revenue += this->sector.R[i];

        for (const Firm& firm : this->sector.firms) {
            price_goods += firm.s[i] * firm.pg[i];
        }
    }

    E_period /= period;
    goods_sold /= period;
    tax_revenue /= period;
    price_goods /= period;

    double consumer_impact = price_goods - (goods_sold > 0.0 ? tax_revenue / goods_sold : 0.0);
    std::vector<double> obs = {E_period, consumer_impact};
    
    return obs;
}

double Environment::calculateReward(const std::vector<double>& obs, const double action, const double last_action) {
    // double emissions = obs[2];
    // double target = 0.2;
    // double deviation = emissions - target;

    // double emissions_reward = std::exp(10 * std::abs(deviation)) + 1;
    // double smoothness_penalty = -0.2 * (action - last_action) * (action - last_action);

    // return emissions_reward + smoothness_penalty;

    static const double E0 = this->init_emissions;
    static const double CC0_0 = this->init_CC0;

    double emissions_decrease = (E0 - obs[0]) / (E0);
    double price_increase = (obs[1] / CC0_0) - 1.0;
    std::cout << "EMISSIONS PERIOD, CC = " << obs[0] << ", " << obs[1] << std::endl;
    std::cout << "EMISSIONS DEC, PRICE INC = " << emissions_decrease << ", " << price_increase << std::endl;

    double agreeableness_util = interpolate_model_1_1(emissions_decrease, price_increase);
    std::cout << "AGREEABLENESS RAW: "<< agreeableness_util << std::endl;
    double agreeableness = 1.0 / (1.0 + std::exp(-agreeableness_util));
    std::cout << "AGREEABLENESS [0,1]: "<< agreeableness << std::endl;
    std::cout << "\n" << std::endl;

    return agreeableness;
}

std::vector<double> Environment::reset() {
    this->params = Parameters();
    this->sector = Sector(this->params);

    this->t = this->params.t_start;
    this->done = false;
    this->last_action = 0.0;

    auto MDP_init = Environment::step(0.0);
    std::vector<double> obs_init = Environment::observe();

    return obs_init;
}

Environment::Environment() : params(), sector(params) {
    std::cout << "C++ ENVIRONMENT INTIALISED WITH " << this->params.N << " FIRMS..." << std::endl;
    std::vector<double> init_obs = Environment::reset();   
}

void Environment::outputTxt() {
    const std::string fileName = "EmissionsVsTaxData.txt";
    this->emissionsVsTax.open(fileName);
    this->emissionsVsTax << "EMISSIONS TAX PG\n";

    int start = this->params.t_start;
    int end = this->params.T;

    for (int i = start; i < end; ++i) {
        this->emissionsVsTax << this->sector.E[i] << " " 
                                << this->tax_actions[i] << " "
                                << this->CC[i] << "\n";
    }
    this->emissionsVsTax.close();
    std::cout << "DATA WRITTEN TO " << fileName << std::endl;
}