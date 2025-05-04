#include "../include_cpp/Environment.h"

// FROM Prosumer Preferences Regarding the
// Adoption of Micro-Generation Technologies:
// Empirical Evidence for German Homeowners -- TABLE 5

static inline double interpolate_model_1_1(double rel_C02_reduction, double rel_price_increase) {
    // LINEAR INTERPOLATION
    // C02 UTILITY (BASED ON REDUCTIONS 0%, 50%, 100%, 200%)
    double C02_SCORE;

    // CASE 1: 0->0.5, beta = 0.41
    if (rel_C02_reduction <= 0.5) {
        C02_SCORE = (0.41 * 2.0) * rel_C02_reduction;                               // (0.41/0.5)                           
    } 
    // CASE 2 : 0.5->1.0, beta = 0.56
    else if (rel_C02_reduction <= 1.0) {                                         
        C02_SCORE = 0.41 + (0.56 - 0.41) * ((rel_C02_reduction - 0.5) * 2.0);      // (rel_C02_reduction - 0.5) / 0.5
    } 
    // CASE 3 : 1.0->2.0, beta = 0.60
    else if (rel_C02_reduction <= 2.0) {
        C02_SCORE = 0.56 + (0.60 - 0.56) * ((rel_C02_reduction - 1.0)); // (/1.0)
    } 
    // IF BEYOND SET TO REMAIN AT 0.60
    else {
        C02_SCORE = 0.60;
    }

    // PRICE UTILITY (BASED ON 60%, 80%, 120%)
    double PRICE_SCORE;

    // CASE 1: 0->0.6, beta = 1.09
    if (rel_price_increase <= 0.6) {
        PRICE_SCORE = (1.09 / 0.6) * rel_price_increase;
    } 
    // CASE 2: 0.6->0.8, beta = 0.58
    else if (rel_price_increase <= 0.8) {                                        
        PRICE_SCORE = 1.09 + (0.58 - 1.09) * ((rel_price_increase - 0.6) * 5.0); // ( / 0.2)
    } 
    // IF BEYOND, LINEARLY EXTEND
    else {
        static const double slope = (-0.88 - 0.58) * 2.5; //(.../0.4) = -3.65
        PRICE_SCORE = -0.88 * slope * (rel_price_increase - 1.2);
    }

    // RETURN UTILITY V
    return C02_SCORE + PRICE_SCORE;
}

std::tuple<std::vector<double>, double, double, bool> Environment::step(const int action_idx) {
    
    // ENSURE WITHIN LIMITS
    assert(action_idx >= 0 && action_idx < 10);
    double action = this->action_table[action_idx];

    if (this->last_action + action > this->tax_limit) {
        this->new_action = this->tax_limit;
    } else if (this->last_action + action < 0.0) {
        this->new_action = 0.0;
    } else {
        this->new_action = this->last_action + action;
    }

    int period = this->params.t_period;

    // FOR ONE REGULATION PERIOD
    for (int i = 0; i < period && this->t <= this->params.T; ++i, ++this->t) {
        this->tax_actions[this->t] = this->new_action;

        // MAIN DYNAMICS
        this->sector.applyExpectations(this->t);
        this->sector.applyAbatement(this->t, this->new_action);
        this->sector.applyProduction(this->t, this->new_action);
        this->sector.tradeCommodities(this->t);

        // MEASURES
        double price_goods = 0.0;
        for (const Firm& firm : this->sector.firms) {
                price_goods += firm.s[this->t] * firm.pg[this->t];
        }

        this->CC0[this->t] = price_goods;

        double total_sold = this->sector.Q_s[this->t];
        double total_revenue = this->sector.R[this->t];

        double consumer_impact = price_goods - (total_sold > 0.0 ? total_revenue / total_sold : 0.0);
        this->CC[this->t] = consumer_impact;

        // FOR INITIALISING VALUES
        if (this->t <= 10) {
            this->init_emissions += this->sector.E[this->t];
            this->init_CC0 += price_goods;
        }
        // MEAN
        if (this->t == 10) {
            this->init_emissions /= this->params.t_period;
            this->init_CC0 /= this->params.t_period;
        }
    }


    // GET OBSERVATIONS
    std::vector<double> next_obs = Environment::observe();

    // GET REWARD(s)
    std::array<double, 2> reward = Environment::calculateReward(next_obs);

    this->last_action = this->new_action;
    done = (this->t > this->params.T);

    // RETURN MDP
    this->Markov = {next_obs, reward[0], reward[1], done};
    return this->Markov;
}

std::vector<double> Environment::observe() {
    
    // DEFINE OBSERVATIONS:
    // FOR EACH FIRM : {MARKET SHARE, EMISSIONS INTENSITY, COST OF PRODUCTION, NO. REMAINING ABATEMENT OPTIONS}
    // FOR WHOLE SECTOR : {CURRENT EMISSIONS, LAST TAX LEVEL, CURRENT CONSUMER IMPACT}

    static const double invNumOptions = 1.0 / (static_cast<double>(this->params.lamb_n));

    std::vector<double> current_obs(this->observation_dim);
    const int t_curr = this->t - 1;

    // GET FIRM OBS
    for (int i = 0; i < this->params.N; ++i) {
        const Firm& firm = this->sector.firms[i];
        int baseIdx = i * 4;
        current_obs[baseIdx + 0] = firm.s[t_curr];
        current_obs[baseIdx + 1] = firm.A[t_curr];
        current_obs[baseIdx + 2] = firm.B[t_curr];
        current_obs[baseIdx + 3] = firm.lamb.size() * invNumOptions;
    }

    double E_recent = this->sector.E[t_curr];
    double CC_recent = this->CC[t_curr];

    int endIdx = this->params.N * 4;
    
    current_obs[endIdx + 0] = E_recent;
    current_obs[endIdx + 1] = this->new_action;
    current_obs[endIdx + 2] = CC_recent;

    return current_obs;
}

std::array<double, 2> Environment::calculateReward(const std::vector<double>& obs) {

    // INIT EMISSIONS, CONSUMER IMPACT, TARGET FOR EMISSIONS REDUCTION, 1.0 / (INIT - TARGET)
    static const double E0 = this->init_emissions;
    static const double CC0_0 = this->init_CC0;
    static const double alpha = 5.0;

    // SIGMOID TEMPERATURE
    static const double temp_exp = 3.0;

    // static const double E_TARGET = E0 * this->emissions_target;
    // static const double inv_denominator = 1.0 / (E0 - E_TARGET);

    static const double TARGET_RATIO = 1.0 - this->emissions_target;

    const int endIdx = obs.size() - 1;
    double consumer_impact = obs[endIdx];
    double emissions_current = obs[endIdx - 2];

    double emissions_decrease = (E0 - emissions_current) / (E0);
    double price_increase = (consumer_impact / CC0_0) - 1.0;

    // AGREEABLENESS REWARD
    double agreeableness_util = interpolate_model_1_1(emissions_decrease, price_increase);
    double agreeableness_reward = 1.0 / (1.0 + std::exp(- temp_exp * agreeableness_util));
    
    // EMISSIONS REWARD
    const double exponent = -alpha * (emissions_decrease - TARGET_RATIO) * (emissions_decrease - TARGET_RATIO);
    double emissions_reward = std::exp(exponent);

    std::array<double, 2> reward = {emissions_reward, agreeableness_reward};

    return reward;
}

std::vector<double> Environment::reset() {
    this->params = Parameters(this->tech_mode, this->seed);
    this->sector = Sector(this->params);

    this->t = this->params.t_start;
    this->done = false;
    this->last_action = 0.0;
    this->new_action = 0.0;
    this->init_emissions = 0.0;
    this->init_CC0 = 0.0;

    const int T_plus = this->params.T + 2;

    this->tax_actions.clear();
    this->CC0.clear();
    this->CC.clear();

    this->tax_actions.resize(T_plus, 0.0);
    this->CC0.resize(T_plus, 0.0);
    this->CC.resize(T_plus, 0.0);

    auto MDP_init = Environment::step(4);                   //CALIBRATE WITH NO PRICE OF EMISSIONS
    std::vector<double> obs_init = Environment::observe();

    return obs_init;
}

Environment::Environment(std::string TECH_MODE, int seed_, double target_, double chi_) : 
                            params(TECH_MODE, seed), sector(params), 
                            emissions_target(target_),
                            chi(chi_),
                            tech_mode(TECH_MODE),
                            seed(seed_)  {

    std::cout << "C++ ENVIRONMENT INTIALISED WITH " << this->params.N << " FIRMS..." << std::endl;
    std::vector<double> init_obs = Environment::reset();   

}

void Environment::outputTxt() {
    std::string fileName = "EmissionsVsTaxData_Chi";
    std::string chiString = std::to_string(this->chi);

    fileName += chiString.substr(0, 4);

    if (this->seed != -1) {
        std::string seedString = "_SEED" + std::to_string(this->seed);
        fileName += seedString;
    }

    if (this->tech_mode != "AVERAGE" || this->tech_mode != "average") {
        fileName += "_MODE" + this->tech_mode;
    }

    fileName += ".txt";

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