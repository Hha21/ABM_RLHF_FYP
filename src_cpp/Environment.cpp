#include "Environment.h"

std::tuple<std::vector<double>, double, bool> Environment::step(double action) {

    double new_action = std::clamp(this->last_action + action, 0.0, this->tax_limit);

    int period = this->params.t_period;

    for (int i = 0; i < period && t <= this->params.TP; ++i, ++t) {
        this->sector.applyExpectations(t);
        this->sector.applyAbatement(t, new_action);
        this->sector.applyProduction(t, new_action);
        this->sector.tradeCommodities(t);
    }

    std::vector<double> next_obs = Environment::observe();
    double reward = Environment::calculateReward(next_obs, new_action, last_action);
    last_action = new_action;

    done = (t > this->params.TP);

    return {next_obs, reward, done};
}