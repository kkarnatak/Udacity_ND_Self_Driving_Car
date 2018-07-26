#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

///////////////////////////////////////////////////////////////////////////////

KalmanFilter::KalmanFilter() {}

///////////////////////////////////////////////////////////////////////////////

KalmanFilter::~KalmanFilter() {}

///////////////////////////////////////////////////////////////////////////////

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {

  // Initialize the matrices

  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

///////////////////////////////////////////////////////////////////////////////

void KalmanFilter::Predict() {
  /**
  Completed:
    * predict the state
  */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

///////////////////////////////////////////////////////////////////////////////

void KalmanFilter::Update(const VectorXd &z) {
  /**
  Completed:
    * update the state by using Kalman Filter equations
  */
  MatrixXd Ht = H_.transpose();

  VectorXd y = z - (H_ * x_);
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

///////////////////////////////////////////////////////////////////////////////

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  Completed:
    * update the state by using Extended Kalman Filter equations
  */

  // Taken from the lecture notes and exercise
  // Local variables
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];

  // Calculation related to the radar sensor
  float rho = sqrt(pow(px,2)+pow(py,2));

  // The use of atan2 return values between -pi and pi instead of -pi/2 and pi/2 for atan
  float phi = atan2(py,px);
  float rhodot;

  //Check the division by 0
  if (fabs(rho) < SMALL_VAL) {
    rhodot = 0;
  } else {
    rhodot = (vx*px + vy*py)/rho;
  }

  // Create a function that contains our predictions
  VectorXd z_pred (3);
  z_pred << rho, phi, rhodot;

  // Compute the error> current values - estimations
  VectorXd y = z - z_pred;

  //Normalizing the angles here; such that y[1] stays between -Pi and +Pi
  while (y[1] > M_PI){
     y[1] -= (2 * M_PI);
  }
  while (y[1] < - M_PI){
     y[1] += (2 * M_PI);
  }

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

///////////////////////////////////////////////////////////////////////////////