int main() {
    std::cout << "EKF SLAM C++ Library - Minimal Usage Example" << std::endl;

    // 1. Initialize State and Covariance
    Matrix xEst(STATE_SIZE, 1); // Robot starts at [0,0,0]
    Matrix PEst = Matrix::identity(STATE_SIZE);

    // 2. Define Noise Covariances
    // Motion noise [velocity^2, yaw_rate^2]
    Matrix Q = Matrix::diag({0.1, 0.1}); 
    // Measurement noise [range^2, angle^2]
    Matrix R = Matrix::diag({0.2*0.2, (5.0*M_PI/180.0)*(5.0*M_PI/180.0)});

    // 3. Define Input and Observations (Dummy Data for example)
    // Control: 1.0 m/s forward, 0.1 rad/s turn
    Matrix u(2, 1);
    u(0, 0) = 1.0; 
    u(1, 0) = 0.1;

    // Measurement: Range=10.0m, Angle=0.0rad (seen straight ahead)
    Matrix z(1, 3); 
    z(0, 0) = 10.0; // Range
    z(0, 1) = 0.0;  // Angle
    z(0, 2) = 0.0;  // ID (ignored by default implementation)

    std::cout << "Initial State: [" << xEst(0,0) << ", " << xEst(1,0) << ", " << xEst(2,0) << "]" << std::endl;

    // 4. Run one step of EKF SLAM  
    auto t_start = std::chrono::high_resolution_clock::now();
    auto result = ekf_slam(xEst, PEst, u, z, Q, R);
    auto t_final = std::chrono::high_resolution_clock::now();
    xEst = result.first;
    PEst = result.second;
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t_final - t_start).count();
    std::cout << "State after 1 step: [" << xEst(0,0) << ", " << xEst(1,0) << ", " << xEst(2,0) << "]" << std::endl;
    std::cout << "Landmarks in map: " << calc_n_lm(xEst) << std::endl;
    std::cout << "ekf_slam execution time: " << us << " us (" << (us / 1000.0) << " ms)" << std::endl;

    
    // Verify a landmark was added at roughly x=10, y=0 (since robot was at 0,0 facing 0, seeing object 10m ahead)
    if (calc_n_lm(xEst) > 0) {
        Matrix lm1 = get_landmark_position_from_state(xEst, 0);
        std::cout << "Landmark 1 Est: [" << lm1(0,0) << ", " << lm1(1,0) << "]" << std::endl;
    }

    return 0;
}