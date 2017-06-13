#include "tools.h"
#include <iostream>
namespace Tools
{
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                         const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    int estimationsSize = estimations.size();

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if(estimationsSize != ground_truth.size()
          || estimationsSize == 0){
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }
    //accumulate squared residuals
    for(int i=0; i < estimationsSize; ++i){
        VectorXd diff = estimations[i]-ground_truth[i];
        VectorXd residual = diff.array()*diff.array();
        rmse += residual;
    }
    rmse = rmse/estimationsSize;
    rmse = rmse.array().sqrt();
    return rmse;
  }

  double NormAngle (double angleRad) {
    if (fabs(angleRad) > M_PI) {
        double pi2 = (M_PI * 2.);
        double correction = (floor((angleRad + M_PI) / pi2) * pi2);
        angleRad -= correction;
    }
    return angleRad;
  }

}