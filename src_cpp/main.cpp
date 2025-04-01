#include <iostream>
#include "../include_cpp/Environment.h"


int main() {

    Environment env = Environment();

    int t0 = env.params.t_start;
    int t_max = env.params.T;
    double action = 0.1;
    MDP Markov;
    

    while (!env.done) {
        Markov = env.step(action);
    }
    env.outputTxt();
    

    std::cout << "RUNNING TERMINATED!" << std::endl;
    return 0;
}