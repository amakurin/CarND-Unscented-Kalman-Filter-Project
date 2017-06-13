#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  is_initialized_ = false;
  
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  n_p_ = 2 * n_aug_ + 1;

  Xsig_pred_ = MatrixXd(n_x_, n_p_);

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.25;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  epsilon_ = 0.0001;
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 5);
  
  R_laser_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;

  //measurement covariance matrix - radar
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0, std_radrd_*std_radrd_;

  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  weights_ = VectorXd(n_p_);
  weights_.fill(0.5/(n_aug_+lambda_));
  weights_(0) = lambda_/(lambda_+n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  bool is_radar = (meas_package.sensor_type_ == MeasurementPackage::RADAR);
  bool is_laser = (meas_package.sensor_type_ == MeasurementPackage::LASER);
  /*****************************************************************************
  *  Initialization
  ****************************************************************************/
  if (!is_initialized_) {
    cout << "UKF...";
    time_us_ = meas_package.timestamp_;

    //state covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 10, 0, 0,
          0, 0, 0, 10, 0,
          0, 0, 0, 0, 10;
    
    if (is_radar) {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double px = (rho*cos(phi));
      double py = (rho*sin(phi));
      x_ << px, py, 0,0,0;
    }
    else if (is_laser) {
      x_ << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1),0,0,0;
    }
    // done initializing, no need to predict or update
    is_initialized_ = true;
    cout << "initialized" << endl;
    return;
  }

  if ((is_laser && !use_laser_)|| (is_laser && !use_laser_)){
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0; 
  time_us_ = meas_package.timestamp_;

  Prediction(dt);  
  
  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (is_radar) {
    UpdateRadar(meas_package);
  } else if (is_laser){
    UpdateLidar(meas_package);
  }

  // print the output
  //cout << "x_ = " << x_ << endl;
  //cout << "P_ = " << P_ << endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  Xsig_pred_ = predictSigmaPoints(delta_t);

  //predicted state mean
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);
  for (int i = 0; i < n_p_; i++) { 
    x = x + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);
  for (int i = 0; i < n_p_; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    //angle normalization
    x_diff(3) = Tools::NormAngle(x_diff(3));
    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }

  x_ = x;
  P_ = P;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  VectorXd y = meas_package.raw_measurements_ - H_laser_ * x_;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;

  //new state
  x_ = x_ + (K * y);
  x_(3) = Tools::NormAngle(x_(3));
  P_ = (MatrixXd::Identity(n_x_, n_x_) - K * H_laser_) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  //return;
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, n_p_);

  //transform sigma points into measurement space
  for (int i = 0; i < n_p_; i++) {
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    if (fabs(Zsig(0,i)) > epsilon_){ 
      Zsig(2,i) = (p_x*v1 + p_y*v2 ) / Zsig(0,i);               //r_dot
    } else {
      Zsig(2,i) = 0;
    }  
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < n_p_; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < n_p_; i++) { 
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    z_diff(1) = Tools::NormAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S += R_radar_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_p_; i++) {  //iterate over sigma points
    // differences
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    x_diff(3) = Tools::NormAngle(x_diff(3));
    z_diff(1) = Tools::NormAngle(z_diff(1));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose() ;
  }
  MatrixXd Si = S.inverse();
  
  //calculate Kalman gain K;
  MatrixXd K = Tc * Si;
  
  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  z_diff(1) = Tools::NormAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  double NIS = z_diff.transpose() * Si * z_diff;
  cout<< "NIS Radar: "<< NIS << endl;
}

MatrixXd UKF::generateSigmaPoints(){

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_p_);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_+1) = 0;
  
  //create augmented covariance matrix
  P_aug.setZero();
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_ * std_yawdd_;
  
  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  A *= sqrt(lambda_ + n_aug_);
  //create augmented sigma points

  Xsig_aug.col(0) = x_aug;
  for (int i=0; i<n_aug_; ++i){
      Xsig_aug.col(i+1) = x_aug + A.col(i);
      Xsig_aug.col(i+n_aug_+1) = x_aug - A.col(i);
  }
  return Xsig_aug;
}

MatrixXd UKF::predictSigmaPoints(double dt){
  MatrixXd Xsig_aug = generateSigmaPoints();
  MatrixXd Xsig_pred = MatrixXd(n_x_, n_p_);
  double dt2 = dt * dt;
  //predict sigma points
  for (int i=0; i < n_p_; ++i){
    VectorXd point = Xsig_aug.col(i);
    double px = point(0);
    double py = point(1);
    double v = point(2);
    double psi = point(3);
    double psid = point(4);
    double ua = point(5);
    double upsidd = point(6);
    
    VectorXd noise = VectorXd(5);
    noise(0) = 0.5*dt2*cos(psi)*ua;
    noise(1) = 0.5*dt2*sin(psi)*ua;
    noise(2) = dt*ua;
    noise(3) = 0.5*dt2*upsidd;
    noise(4) = dt*upsidd;
    VectorXd fx = VectorXd(5);
    //avoid division by zero
    if (fabs(psid) > 0.0001){
      double d = v/psid;
      double psi_delta = psid*dt;
      fx(0) = d * (sin(psi + psi_delta) - sin(psi));
      fx(1) = d * (-cos(psi + psi_delta) + cos(psi));
      fx(2) = 0;
      fx(3) = psi_delta;
      fx(4) = 0;
    } else {
      fx(0) = v * cos(psi) * dt;
      fx(1) = v * sin(psi) * dt;
      fx(2) = 0;
      fx(3) = 0;
      fx(4) = 0;
    }
    //write predicted sigma points into right column
    Xsig_pred.col(i) = point.head(n_x_) + fx + noise;      
  }
  return Xsig_pred;
}
