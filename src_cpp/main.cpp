#include <iostream>
#include <fstream>
#include "../include_cpp/Environment.h"

bool emissionsZeroes(const std::vector<double>& emissions, double threshold = 0.01) {
    for (double e : emissions) {
        if (e < threshold) {
            return true;
        }
    }
    return false;
}

void runEmissionsTests() {
    std::ofstream logfile("test_results.txt");

    Environment env = Environment();
    
    const std::vector<double> actions = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30};
    const int num_trials = 2000;
    int trial_id = 0;

    for (double action : actions) {
        int failures = 0;

        Environment env;

        for (int trial = 0; trial < num_trials; ++trial) {
            if (trial == 0) {
                std::cout << "Trial " << trial_id << ": action=" << action << std::endl;
            }
            
            std::vector<double> emissions_log;

            while (!env.getDone()) {
                auto [obs, reward, done] = env.step(action);
                //std::cout << "EMISSIONS: " << obs[0] << std::endl;
                emissions_log.push_back(obs[0]); 
            }

            if (emissionsZeroes(emissions_log)) {
                ++failures;
            }

            env.reset();

            ++trial_id;
        }

        logfile << "Action: " << action << " -> Crashes: " << failures << " / " << num_trials << std::endl;
        std::cout << "Action: " << action << " -> Crashes: " << failures << " / " << num_trials << std::endl;
    }

    logfile.close();
    std::cout << "All trials complete.\n";
}

int main() {
    
    Environment env = Environment();

    while (!env.getDone()) {
        env.step(0.0);
    }

    return 0;
}