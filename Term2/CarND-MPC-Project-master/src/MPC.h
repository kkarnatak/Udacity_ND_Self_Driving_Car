#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

///////////////////////////////////////////////////////////////////////////////

using namespace std;

class MPC {
 public:
  const double Lf = 2.67;
  vector<double> v_cte;
  vector<double> v_vel;
  vector<double> v_delta;
  MPC();

  virtual ~MPC();

  // For plotting the data
  inline vector<vector<double>> GetParameters()
  {
    vector<vector<double>> tmp;
    tmp.push_back(v_vel);
    tmp.push_back(v_cte);
    tmp.push_back(v_delta);

    return tmp;
  }
  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
};

///////////////////////////////////////////////////////////////////////////////

#endif /* MPC_H */
