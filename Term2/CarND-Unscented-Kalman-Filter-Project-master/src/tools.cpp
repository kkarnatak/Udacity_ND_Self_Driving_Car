#include <iostream>
#include "tools.h"

///////////////////////////////////////////////////////////////////////////////

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

///////////////////////////////////////////////////////////////////////////////

Tools::Tools() {}

///////////////////////////////////////////////////////////////////////////////

Tools::~Tools() {}

///////////////////////////////////////////////////////////////////////////////

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    // Intialize rmse matrix
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // Verify the data
    if(estimations.size() == 0 || estimations.size() != ground_truth.size()){
        cout << "Invalid data" << endl;
        return rmse;
    }

    // As discussed in lecture and implemented in the first project
    for(unsigned int i=0; i<estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array()* residual.array();
        rmse += residual;
    }

    // Calculate the mean
    rmse /= estimations.size();

    // Compute the square root
    rmse = rmse.array().sqrt();

    return rmse;
}

///////////////////////////////////////////////////////////////////////////////