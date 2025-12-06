#include <iostream>
#include <vector>
#include <iomanip>
#include "../src/ekf_slam.h"

int main() {
    std::cout << "--- EKF SLAM HLS Testbench ---" << std::endl;

    data_t x_in[MAX_ROWS] = {0};
    // Initial Robot State 
    float init_data[15] = {-5.62815962, 18.37374747, -2.54979418};
    for(int i=0; i<3; i++) x_in[i] = init_data[i];
    int x_rows = 3;

    // P Init to max size directly
    data_t P_in[MAX_ROWS*MAX_ROWS] = {0};
    for(int i=0; i<3; i++) P_in[i*MAX_COLS + i] = 1.0;
    int P_rows = 3;

    // Control
    data_t u_in[2] = {1.0, 0.1}; // v, yaw_rate

    // Observation (Range 10m, Angle 0)
    data_t z_in[3] = {10.0, 0.0, 0.0};

    // Noise
    data_t Q_in[2] = {0.1, 0.1};
    data_t R_in[2] = {0.04, 0.0076}; // 0.2^2, (5deg)^2

    // Outputs Buffers
    data_t x_out[MAX_ROWS] = {0};
    data_t P_out[MAX_ROWS*MAX_ROWS] = {0};
    int x_rows_out = 0;
    int P_rows_out = 0;

    // Call EKF SLAM function
    std::cout << "Running EKF Step..." << std::endl;
    ekf_slam(
        x_in, x_rows,
        P_in, P_rows,
        u_in,
        z_in,
        Q_in, R_in,
        x_out, x_rows_out,
        P_out, P_rows_out
    );

    // Verify
    std::cout << "Done." << std::endl;
    std::cout << "New State Size: " << x_rows_out << std::endl;
    // Access as flat vector (row-major)
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Robot State: [" << x_out[0] << ", " << x_out[1] << ", " << x_out[2] << "]" << std::endl;
    
    // If a landmark was added, size should augment by 2
    if (x_rows_out > 3) {
        std::cout << "Landmark: [" << x_out[3] << ", " << x_out[4] << "]" << std::endl;
    }

    return 0;
}
