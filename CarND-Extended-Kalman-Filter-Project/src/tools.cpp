#include "tools.h"
#include <iostream>
#include "Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  // check the validity of the inputs:
  if(estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    cout << "estimations vector has size zero or is not same size as ground truth" << endl;
    return rmse;
    }
  
  for(unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd Residual = estimations[i] - ground_truth[i];
    Residual = Residual.array() * Residual.array();
    rmse += Residual;
    
  }
  
  rmse = rmse / estimations.size();
  
  rmse = rmse.array().sqrt();
  
  return rmse;
    
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
  MatrixXd Hj(3,4);
  
  // Get state parameter values (position and velocity in x,y)
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  //Pre compute to avoid repeat calculations
  float c1 = px*px + py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);
  
  // Check for division by zero
  if (fabs(c1) < 0.00001) { 
    cout << "CalculationJacobian() error - division by zero" << endl;
    
    return Hj;
  }
  
  // compute the Jacobian matrix
  Hj << (px / c2), (py / c2), 0, 0,
      -(py / c1), (px / c1), 0, 0,
      py*(vx*py - vy*px) / c3, px*(px*vy - py*vx) / c3, px / c2, py / c2;

  return Hj;
}
