#ifndef EKF_H
#define EKF_H

#include <concepts>
#include <functional>
#include <iostream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "autodiff/forward.hpp"
#include "autodiff/forward/eigen.hpp"

template<typename T>
concept SuitableEKFType = std::is_floating_point_v<T> && !std::is_const_v<T> && !std::is_volatile_v<T>;


template<typename TYPE, size_t STATE_DIM, size_t OBS_DIM, size_t CONTROL_DIM> 
    requires SuitableEKFType<TYPE>
struct ExtendedKalmanFilter {

    // just to keep it simple
    using StateVector = Eigen::Matrix<autodiff::dual, STATE_DIM, 1>;
    using ControlVector = Eigen::Matrix<autodiff::dual, CONTROL_DIM, 1>;
    using ObservationVector = Eigen::Matrix<TYPE, OBS_DIM, 1>;
    using ProcessModel = std::function<Eigen::Matrix<autodiff::dual, STATE_DIM, 1>(const autodiff::VectorXdual& x, const autodiff::VectorXdual& u)>;
    using ObservationModel = std::function<Eigen::Matrix<autodiff::dual, OBS_DIM, 1>(const autodiff::VectorXdual& x)>;

    constexpr explicit ExtendedKalmanFilter() = default;
    constexpr ~ExtendedKalmanFilter() = default;

    ExtendedKalmanFilter(const ExtendedKalmanFilter&) = delete;
    ExtendedKalmanFilter& operator=(const ExtendedKalmanFilter&) = delete;

    constexpr void SetP(const Eigen::Matrix<TYPE, STATE_DIM, STATE_DIM>& P) {
        m_P = std::move(P);
    }

    constexpr void SetQ(const Eigen::Matrix<TYPE, STATE_DIM, STATE_DIM>& Q) {
        m_Q = std::move(Q);
    }

    constexpr void SetR(const Eigen::Matrix<TYPE, OBS_DIM, OBS_DIM>& R) {
        m_R = std::move(R);
    }

    constexpr void SetProcessModel(ProcessModel&& g) {
        m_g = g;
    }

    constexpr void SetMeasurementModel(ObservationModel&& h) {
        m_h = std::move(h);
    }

    constexpr void SetInitialState(const StateVector& x) {
        m_x = std::move(x);
    }

    [[nodiscard]] constexpr StateVector GetPredictedState() const {
        return m_x;
    }

    constexpr void Predict(const ControlVector& u = {}) {
        auto F = autodiff::forward::jacobian(m_g, autodiff::forward::wrt(m_x), autodiff::forward::at(m_x, u));
        m_x = F * m_x;
        m_P = F * m_P * F.transpose() + m_Q;
    }

    template<typename Proc = std::identity>
    constexpr void Update(const ObservationVector& z, Proc&& proc = {}) {
        auto H = autodiff::forward::jacobian(m_h, autodiff::forward::wrt(m_x), autodiff::forward::at(m_x));
        auto H_t = H.transpose();
        auto S_inv = (H * m_P * H_t + m_R).inverse();
        auto K = m_P * H_t * S_inv;
        m_x = m_x + K * proc(z - m_h(m_x));
        m_P = (m_I - K * H) * m_P;
    }

private:

    Eigen::Matrix<TYPE, STATE_DIM, STATE_DIM> m_P; // prediction error covariance
    Eigen::Matrix<TYPE, STATE_DIM, STATE_DIM> m_Q; // process noise covariance
    Eigen::Matrix<TYPE, OBS_DIM, OBS_DIM> m_R;     // measurement error covariance

    ProcessModel m_g;            // state transition probability x = g(x,u)
    ObservationModel m_h;        // measurement probability      z = h(x)
    StateVector m_x;             // state

    // will need the identity
    Eigen::Matrix<TYPE, STATE_DIM, STATE_DIM> m_I = Eigen::Matrix<TYPE, STATE_DIM, STATE_DIM>::Identity();  

};



#endif // EKF_H
