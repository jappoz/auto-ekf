#include <iostream>
#include "ekf.h"
#include "matplotlibcpp.h"
#include <vector>
#include <fstream>


int main(int argc, char** argv) {

    namespace plt = matplotlibcpp;
    matplotlibcpp::backend("Gtk3Agg");

    std::string base_path = EXAMPLES_DATA_PATH;
    std::ifstream file(base_path + "measurements.txt");

    // getting radar measurements and ground thruth
    // meas_rho meas_phi meas_rho_dot timestamp gt_px gt_py gt_vx gt_vy
    using radar_measurement = std::array<double, 8>;
    std::vector<radar_measurement> radar_measurements;
    
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        
        char sensor;
        iss >> sensor;
        if(sensor == 'R') {
            radar_measurement tmp;
            for(size_t i = 0; i < 8; ++i)
                iss >> tmp[i];
            radar_measurements.emplace_back(tmp);
        }
    }

    // prepare for plotting and kalman filter execution
    std::vector<double> meas_rho, meas_phi, meas_rho_dot;
    std::vector<double> gt_x, gt_y, gt_vx, gt_vy;
    std::vector<double> timestamps;

    for(auto&& i : radar_measurements) {
        meas_rho.emplace_back(i[0]);
        meas_phi.emplace_back(i[1]);
        meas_rho_dot.emplace_back(i[2]);

        timestamps.push_back(i[3]);

        gt_x.push_back(i[4]);
        gt_y.push_back(i[5]);
        gt_vx.push_back(i[6]);
        gt_vy.push_back(i[7]);
    }

    // state = (px, py, vx, vy)
    // obs = (rho, phi, rho_dot)
    // no control
    const auto STATE_DIM = 4;
    const auto OBS_DIM = 3;

    ExtendedKalmanFilter<double, STATE_DIM, OBS_DIM, 0> radar_ekf;

    // measurement error covariance
    Eigen::Matrix<double, OBS_DIM, OBS_DIM> R;
    R << 0.09, 0, 0,
         0, 0.0009, 0,
         0, 0, 0.09;

    // prediction error covariance
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> P;
    P << 1,0,0,0,
         0,1,0,0,
         0,0,50,0,
         0,0,0,50;

    // process noise covariance ( changes over time )
    auto Q = Eigen::Matrix<double, STATE_DIM, STATE_DIM>::Zero();

    radar_ekf.SetP(P);
    radar_ekf.SetQ(Q);
    radar_ekf.SetR(R);

    // setting process model in ekf
    const auto dt = 100000 / 1.e6;

    // process model
    auto g = [&](const autodiff::VectorXdual& x, const autodiff::VectorXdual&) {

      autodiff::VectorXdual ret(4);
      ret << x(0) + dt * x(2),  x(1) + dt * x(3), x(2), x(3);
      return ret;

    };

    radar_ekf.SetProcessModel(g);

    // observation model
    auto h = [](const autodiff::VectorXdual& x) {

        Eigen::Matrix<autodiff::dual, OBS_DIM, 1> ret;
        ret << sqrt(pow(x(0), 2) + pow(x(1), 2)),
               atan2(x(1), x(0)),
               ((x(0) * x(2)) + (x(1) * x(3))) / sqrt(pow(x(0), 2) + pow(x(1), 2));
        return ret;

    };

    radar_ekf.SetMeasurementModel(h);

    // start
    // acceleration covariances
    const float ax = 9.0;
    const float ay = 9.0;

    // initialize state vector
    autodiff::VectorXdual x(4);

    // measurements need to be in cartesian coordinates

    auto polar2cart = [](const radar_measurement& m) -> auto {

        autodiff::VectorXdual x(4);
        x << m[0] * cos(m[1]), m[0] * sin(m[1]), m[2] * cos(m[1]), m[2] * sin(m[1]);
        return x;

    };

    x = polar2cart(radar_measurements[0]);
    radar_ekf.SetInitialState(x);
    std::vector<double> est_x, est_y, est_vx, est_vy;

    // remark: we need to clamp the value of phi in y = z - h(x)
    auto clamp_phi = [](const autodiff::VectorXdual& y) -> auto {
        
        auto phi = y(1);
        while (phi > M_PI) phi -= 2 * M_PI;

        while (phi <= -M_PI) phi += 2 * M_PI;
    
        auto ret = y;
        ret(1) = phi;
        return ret;
  
    };

    for(auto&& m : radar_measurements) {

        radar_ekf.Predict();

        Eigen::Matrix<double, OBS_DIM, 1> measurement;
        measurement << m[0], m[1], m[2];

        radar_ekf.Update(measurement, clamp_phi);

        auto e = radar_ekf.GetPredictedState();

        est_x.emplace_back(e[0]);
        est_y.emplace_back(e[1]);
        est_vx.emplace_back(e[2]);
        est_vy.emplace_back(e[3]);

        auto dt2 = dt * dt;
        auto dt3 = dt * dt2;
        auto dt4 = dt * dt3;

        auto r11 = dt4 * ax / 4;
        auto r13 = dt3 * ax / 2;
        auto r22 = dt4 * ay / 4;
        auto r24 = dt3 * ay / 2;
        auto r31 = dt3 * ax / 2;
        auto r33 = dt2 * ax;
        auto r42 = dt3 * ay / 2;
        auto r44 = dt2 * ay;

        Eigen::Matrix<double, STATE_DIM, STATE_DIM> new_Q;
        new_Q << r11, 0.0, r13, 0.0,
             0.0, r22, 0.0, r24,
             r31, 0.0, r33, 0.0,
             0.0, r42, 0.0, r44;
        radar_ekf.SetQ(new_Q);

    }

    plt::named_plot("actual motion", gt_x, gt_y);
    plt::named_plot("estimated motion", est_x, est_y);
    plt::legend();
    plt::show();


}

