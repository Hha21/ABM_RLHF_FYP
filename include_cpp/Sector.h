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
        std::vector<double> D;                              ///< Sector Level Demand
        std::vector<double> E;                              ///< Sector Emissions
        std::vector<double> Q;                              ///< Sector Quantity of Goods Produced
        std::vector<double> Q_s;                            ///< Sector Quantity of Goods Sold
        std::vector<double> u_t;
        std::vector<double> R;                              ///< Sector Tax Revenue
};


#endif //SECTOR_H