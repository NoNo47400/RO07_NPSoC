#include <iostream>
#include <vector>
#include <iomanip>
#include "ekf_slam.h"

// Helper pour afficher depuis le tableau plat
void print_matrix(const char* name, data_t* data, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for(int i=0; i<rows; i++) {
        std::cout << "  ";
        for(int j=0; j<cols; j++) {
            std::cout << std::fixed << std::setprecision(4) << data[i*MAX_COLS + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "--- EKF SLAM HLS Testbench ---" << std::endl;

    // 1. Inputs Buffers
    data_t x_in[MAX_ROWS] = {0};
    // Initial Robot State (-5.62...)
    double init_data[15] = {-5.62815962, 18.37374747, -2.54979418};
    for(int i=0; i<3; i++) x_in[i] = init_data[i];
    int x_rows = 3;

    // P Init (Identity 3x3 effectively, but buffer is large)
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

    // 2. Call Top Level
    std::cout << "Running EKF Step..." << std::endl;
    ekf_slam_top(
        x_in, x_rows,
        P_in, P_rows,
        u_in,
        z_in,
        Q_in, R_in,
        x_out, x_rows_out,
        P_out, P_rows_out
    );

    // 3. Verify
    std::cout << "Done." << std::endl;
    std::cout << "New State Size: " << x_rows_out << std::endl;
    
    // On s'attend à ce que le robot ait bougé et qu'un landmark ait été ajouté (Size passe de 3 à 5)
    print_matrix("x_out", x_out, x_rows_out, 1);
    
    // Check Landmark pos
    if (x_rows_out >= 5) {
        double lm_x = x_out[3];
        double lm_y = x_out[4];
        std::cout << "Landmark detected at: [" << lm_x << ", " << lm_y << "]" << std::endl;
    }

    return 0;
}