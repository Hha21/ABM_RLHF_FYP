#ifndef FIRM_H
#define FIRM_H

#include "Parameters.h"

#include <array>
#include <string>
#include <vector>

class Firm {

    public:

        // FIRM ATTRIBUTES

        const int j;                        ///< Firm Index

        const double I_d;                   ///< Desired Inventory Share
        const double theta;                 ///< Mark-up Adaption Rate

        const double alpha;                 ///< Abatement Cost Factor
        const double delta;                 ///< Permit Price Adaption Rate
        const double eta;                   ///< Profitability Target for Investments
        const std::string exp_mode;         ///< Expectation Rule
        const double x;                     ///< Expectation Factor

        //ABATEMENT CURVE for j
        std::vector<std::array<double, 2>> lamb;

        //DYNAMIC VARIABLES
        std::vector<double> s;      // Firm market share
        std::vector<double> sq;     // Squared market share (HHI calculation)
        std::vector<double> f;      // Fixed (investment) costs per period
        std::vector<double> e;      // Emissions per period
        std::vector<double> qg;     // Actual goods produced per period
        std::vector<double> qg_s;   // Actual quantity of goods sold per period
        std::vector<double> qg_d;   // Desired production quantity (output) per period
        std::vector<double> qg_I;   // Inventory of goods per period
        std::vector<double> D;      // Actual demand faced per period
        std::vector<double> Dl;     // Latent demand per period 
        std::vector<double> pg;     // Product price per period
        std::vector<double> m;      // Mark-up rate over production costs
        std::vector<double> A;      // Emission intensity (emissions per unit produced)
        std::vector<double> B;      // Unit production costs (excluding abatement costs)
        std::vector<double> qp_d;   // Desired permit quantity
        std::vector<double> u_i;    // Investment utilization rate
        std::vector<double> u_t;    // Total capacity utilization rate
        std::vector<double> cu_t;   // Cumulative utilization rate (historical average)
        std::vector<double> c_pr;   // Total production costs (including abatement)

        Firm(const Parameters& p, int firmIndex);

        // FIRM METHODS
        void setExpectations(const int t);
        double production(const int t, const double pe);     //Returns tax-revenue from firm j
        void abatement(const int t, const double pe);

};

#endif //FIRM_H