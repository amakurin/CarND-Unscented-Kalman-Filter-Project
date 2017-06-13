#ifndef TOOLS_H_
#define TOOLS_H_
#include <iostream>
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using std::vector;

namespace Tools
{
  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * A helper function to normalize angle.
  */
  double NormAngle (double angleRad); 
}


#endif /* TOOLS_H_ */