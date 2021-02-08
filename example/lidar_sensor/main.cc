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

    // getting Lidar measurements and ground thruth
    using lidar_measurement = std::array<double, 7>;
    std::vector<lidar_measurement> lidar_measurements;

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        
        char sensor;
        iss >> sensor;
        if(sensor == 'L') {
            lidar_measurement tmp;
            for(size_t i = 0; i < 7; ++i)
                iss >> tmp[i];
            lidar_measurements.emplace_back(tmp);
        }
    }

    // prepare for plotting and kalman filter execution
    std::vector<double> px, py;
    std::vector<double> gt_x, gt_y, gt_vx, gt_vy;
    std::vector<double> timestamps;

    for(auto&& i : lidar_measurements) {
        px.push_back(i[0]);
        py.push_back(i[1]);

        timestamps.push_back(i[2]);

        gt_x.push_back(i[3]);
        gt_y.push_back(i[4]);
        gt_vx.push_back(i[5]);
        gt_vy.push_back(i[6]);
    }

    // state = (px, py, vx, vy)
    // obs = (px, py) (using Lidar)
    // no control
    const auto STATE_DIM = 4;
    const auto OBS_DIM = 2;

    ExtendedKalmanFilter<double, STATE_DIM, OBS_DIM, 0> lidar_ekf;

    // measurement error covariance
    Eigen::Matrix<double, OBS_DIM, OBS_DIM> R;
    R << 0.225, 0,
         0, 0.225;

    // prediction error covariance
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> P;
    P << 1,0,0,0,
         0,1,0,0,
         0,0,50,0,
         0,0,0,50;

    // process noise covariance ( changes over time )
    auto Q = Eigen::Matrix<double, STATE_DIM, STATE_DIM>::Zero();

    lidar_ekf.SetP(P);
    lidar_ekf.SetQ(Q);
    lidar_ekf.SetR(R);

    // setting process model in ekf
    const auto dt = 100000 / 1.e6;

    // process model
    auto g = [&](const autodiff::VectorXdual& x, const autodiff::VectorXdual&) {

      autodiff::VectorXdual ret(4);
      ret << x(0) + dt * x(2),  x(1) + dt * x(3), x(2), x(3);
      return ret;

    };

    lidar_ekf.SetProcessModel(g);

    // observation model
    auto h = [](const autodiff::VectorXdual& x) -> auto {

        Eigen::Matrix<autodiff::dual, OBS_DIM, 1> ret;
        ret << x(0), x(1);
        return ret;

    };

    lidar_ekf.SetMeasurementModel(h);

    // start
    // acceleration covariances
    const float ax = 9.0;
    const float ay = 9.0;

    // initialize state vector
    autodiff::VectorXdual x(4);
    x << lidar_measurements[0][0], lidar_measurements[0][1], 0, 0;

    lidar_ekf.SetInitialState(x);
    std::vector<double> est_x, est_y, est_vx, est_vy;

    for(auto&& m : lidar_measurements) {

        lidar_ekf.Predict();

        Eigen::Matrix<double, OBS_DIM, 1> measurement;
        measurement << m[0], m[1];

        lidar_ekf.Update(measurement);

        auto e = lidar_ekf.GetPredictedState();

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
        lidar_ekf.SetQ(new_Q);

    }

    plt::named_plot("actual motion", gt_x, gt_y);
    plt::named_plot("estimated motion", est_x, est_y);
    plt::legend();
    plt::show();

}

