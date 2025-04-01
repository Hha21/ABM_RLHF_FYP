#include <iostream>
#include "../include_cpp/Environment.h"


int main() {

    Environment env = Environment();

    int t_max = env.params.T;
    
    MDP Markov;
    
    for (int episode = 0; episode < 400; ++episode) {
        for (int t = 1; t < t_max; ++t) {
        Markov = env.step(0.0);
        }
        env.reset();
    }

    std::cout << "RUNNING TERMINATED!" << std::endl;
    return 0;
}