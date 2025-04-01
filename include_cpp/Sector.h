#ifndef SECTOR_H
#define SECTOR_H

#include "Firm.h"
#include "Parameters.h"

#include <cmath>
#include <vector>

class Sector {

    private:

        Parameters* params;

    public:

        Sector(Parameters& p);

        void applyExpectations(const int t);
        void applyAbatement(const int t, const double pe);
        void applyProduction(const int t, const double pe);
        void tradeCommodities(const int t);

        std::vector<Firm> firms;
        std::vector<double> D;
        std::vector<double> E;
        std::vector<double> Q;
        std::vector<double> u_t;

        double tax_income;
};


#endif //SECTOR_H