
// ekf_slam_hls.h
#ifndef EKF_SLAM_HLS_H
#define EKF_SLAM_HLS_H

#include <cmath>
#include <stdexcept>
#include <utility>

// --- HLS Constants ---
#define M_PI 3.14159265358979323846

// Fixons une taille maximale pour l'état (le plus grand défi)
const int MAX_LANDMARKS = 5;          // Maximum de repères dans la carte
const int STATE_SIZE = 3;             // Robot state size [x,y,yaw]
const int LM_SIZE = 2;                // LM state size [x,y]
const int MAX_STATE_ROWS = STATE_SIZE + LM_SIZE * MAX_LANDMARKS; // Max size: 3 + 2*5 = 13

// --- Algorithm Constants ---
const double DT = 0.1;           // time tick [s]
const double M_DIST_TH = 2.0;    // Mahalanobis distance threshold

// Utilisation du type float/double pour la précision (ap_fixed est préférable en pratique)
typedef double data_t;
// --- 2. Classe Matrix Synthétisable (Matrice Statique) ---
// Note: Cette implémentation est une version simplifiée et plus rigide de votre classe
// pour garantir la synthétisabilité en évitant std::vector.
// Elle suppose que les dimensions max ne dépasseront pas MAX_STATE_ROWS.

class Matrix {
public:
    int rows;
    int cols;
    data_t data[MAX_STATE_ROWS * MAX_STATE_ROWS]; // Tableau statique de taille max

    Matrix(int r, int c, data_t val = 0.0);
    static Matrix identity(int n);
    static Matrix diag(const data_t vals[], int n);

    data_t& operator()(int r, int c);
    const data_t& operator()(int r, int c) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(data_t scalar) const;
    Matrix transpose() const;
    
    // Inversion 2x2: à décommenter si utilisé dans le corps EKF SLAM (calc_innovation)
    Matrix inv2x2() const; 
    
    // Simplification des fonctions bloc (moins d'erreurs en HLS)
    void set_block(int r, int c, const Matrix& other);
    Matrix get_block(int r, int c, int r_len, int c_len) const;

    // Utile pour le testbench C++
    void print() const; 
};
// --- Helpers ---

double pi_2_pi(double angle) {
    while (angle >= M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

int calc_n_lm(const Matrix& x) {
    return (x.rows - STATE_SIZE) / LM_SIZE;
}

// --- Motion Model ---

Matrix motion_model(const Matrix& x, const Matrix& u) {
    Matrix xp = x; // Copy state
    double yaw = x(2, 0);
    
    xp(0, 0) += u(0, 0) * DT * cos(yaw);
    xp(1, 0) += u(0, 0) * DT * sin(yaw);
    xp(2, 0) += u(1, 0) * DT;
    xp(2, 0) = pi_2_pi(xp(2, 0));
    
    return xp;
}

// Returns {A, B}
std::pair<Matrix, Matrix> jacob_motion(const Matrix& x, const Matrix& u) {
    Matrix A = Matrix::identity(3);
    double yaw = x(2, 0);
    double v = u(0, 0);
    
    A(0, 2) = -DT * v * sin(yaw);
    A(1, 2) = DT * v * cos(yaw);
    
    Matrix B(3, 2);
    B(0, 0) = DT * cos(yaw);
    B(1, 0) = DT * sin(yaw);
    B(2, 1) = DT;
    
    return {A, B};
}

// --- Observation Model ---

Matrix calc_landmark_position(const Matrix& x, const Matrix& z) {
    Matrix zp(2, 1);
    double range = z(0, 0);
    double angle = z(1, 0);
    double yaw = x(2, 0);
    
    zp(0, 0) = x(0, 0) + range * cos(yaw + angle);
    zp(1, 0) = x(1, 0) + range * sin(yaw + angle);
    return zp;
}

Matrix get_landmark_position_from_state(const Matrix& x, int ind) {
    int start = STATE_SIZE + LM_SIZE * ind;
    return x.get_block(start, 0, LM_SIZE, 1);
}

Matrix jacob_h(double q, const Matrix& delta, const Matrix& x, int i) {
    double sq = sqrt(q);
    Matrix G(2, 5);
    G(0, 0) = -sq * delta(0, 0); G(0, 1) = -sq * delta(1, 0); G(0, 2) = 0; G(0, 3) = sq * delta(0, 0); G(0, 4) = sq * delta(1, 0);
    G(1, 0) = delta(1, 0);       G(1, 1) = -delta(0, 0);      G(1, 2) = -q; G(1, 3) = -delta(1, 0);     G(1, 4) = delta(0, 0);
    
    G = G * (1.0 / q);
    
    int nLM = calc_n_lm(x);
    Matrix F(5, 3 + 2 * nLM); // Construct big F matrix
    
    // F is mostly sparse. 
    // Top 3 rows are Identity for robot state
    F(0, 0) = 1; F(1, 1) = 1; F(2, 2) = 1;
    // Bottom 2 rows pick the specific landmark
    int lm_idx_start = 3 + 2 * i;
    F(3, lm_idx_start) = 1;
    F(4, lm_idx_start + 1) = 1;
    
    return G * F;
}

std::pair<Matrix, Matrix> jacob_augment(const Matrix& x, const Matrix& z) {
    double r = z(0, 0);
    double angle = z(1, 0);
    double yaw = x(2, 0);
    
    Matrix Jr(2, 3);
    Jr(0, 0) = 1; Jr(0, 2) = -r * sin(yaw + angle);
    Jr(1, 1) = 1; Jr(1, 2) = r * cos(yaw + angle);
    
    Matrix Jz(2, 2);
    Jz(0, 0) = cos(yaw + angle); Jz(0, 1) = -r * sin(yaw + angle);
    Jz(1, 0) = sin(yaw + angle); Jz(1, 1) = r * cos(yaw + angle);
    
    return {Jr, Jz};
}

// --- EKF Core ---

struct InnovationResult {
    Matrix innov;
    Matrix S;
    Matrix H;
};

InnovationResult calc_innovation(const Matrix& xEst, const Matrix& PEst, const Matrix& z, int lm_id, const Matrix& R) {
    Matrix lm = get_landmark_position_from_state(xEst, lm_id);
    Matrix delta = lm - xEst.get_block(0, 0, 2, 1);
    double q = (delta.transpose() * delta)(0, 0);
    double z_angle = atan2(delta(1, 0), delta(0, 0)) - xEst(2, 0);
    
    Matrix zp(2, 1);
    zp(0, 0) = sqrt(q);
    zp(1, 0) = pi_2_pi(z_angle);
    
    Matrix z_meas = z.get_block(0, 0, 2, 1);
    Matrix innov = z_meas - zp;
    innov(1, 0) = pi_2_pi(innov(1, 0));
    
    Matrix H = jacob_h(q, delta, xEst, lm_id);
    Matrix S = H * PEst * H.transpose() + R;
    
    return {innov, S, H};
}

int search_correspond_landmark_id(const Matrix& xEst, const Matrix& PEst, const Matrix& z_meas, const Matrix& R) {
    int nLM = calc_n_lm(xEst);
    int min_id = nLM;
    double min_dist = M_DIST_TH;
    
    for (int i = 0; i < nLM; ++i) {
        InnovationResult res = calc_innovation(xEst, PEst, z_meas, i, R);
        Matrix dist_mat = res.innov.transpose() * res.S.inv2x2() * res.innov;
        double dist = dist_mat(0, 0);
        
        if (dist < min_dist) {
            min_dist = dist;
            min_id = i;
        }
    }
    return min_id;
}

/**
 * Main EKF SLAM Cycle
 * @param xEst Current State Vector [x, y, yaw, lm1_x, lm1_y, ...]
 * @param PEst Current Covariance Matrix
 * @param u Control input [v, yaw_rate]
 * @param z Observations matrix [range, angle, id] (id is ignored in unknown data association)
 * @param Q Motion Noise Covariance
 * @param R Measurement Noise Covariance
 * @return Pair of {Updated State, Updated Covariance}
 */
std::pair<Matrix, Matrix> ekf_slam(Matrix xEst, Matrix PEst, const Matrix& u, const Matrix& z, const Matrix& Q, const Matrix& R) {
    
    // 1. Predict
    int S = STATE_SIZE;
    auto motion_jac = jacob_motion(xEst.get_block(0, 0, S, 1), u);
    Matrix A = motion_jac.first;
    Matrix B = motion_jac.second;
    
    // xEst[0:3] = motion_model(...)
    Matrix xPred = motion_model(xEst.get_block(0, 0, S, 1), u);
    xEst.set_block(0, 0, xPred);
    
    // PEst[0:3, 0:3] = A*P*At + B*Q*Bt
    Matrix Pxx = PEst.get_block(0, 0, S, S);
    Matrix P_new_xx = A * Pxx * A.transpose() + B * Q * B.transpose();
    PEst.set_block(0, 0, P_new_xx);
    
    // Handle cross-covariances if landmarks exist
    if (PEst.rows > S) {
        Matrix Pxm = PEst.get_block(0, S, S, PEst.cols - S);
        Matrix P_new_xm = A * Pxm;
        PEst.set_block(0, S, P_new_xm);
        PEst.set_block(S, 0, P_new_xm.transpose());
    }
    
    xEst(2, 0) = pi_2_pi(xEst(2, 0));
    
    // 2. Update
    for (int i = 0; i < z.rows; ++i) {
        Matrix z_i = z.get_block(i, 0, 1, 3).transpose(); // [range, angle, id]
        
        int min_id = search_correspond_landmark_id(xEst, PEst, z_i, R);
        int nLM = calc_n_lm(xEst);
        
        if (min_id == nLM) {
            // New Landmark found - Extend State
            Matrix lm_pos = calc_landmark_position(xEst, z_i);
            Matrix xNew(xEst.rows + 2, 1);
            xNew.set_block(0, 0, xEst);
            xNew.set_block(xEst.rows, 0, lm_pos);
            xEst = xNew;
            
            // Expand covariance
            auto aug_jac = jacob_augment(xEst.get_block(0, 0, 3, 1), z_i);
            Matrix Jr = aug_jac.first;
            Matrix Jz = aug_jac.second;
            
            // Reconstruct PEst with new size
            int old_size = PEst.rows;
            Matrix PNew(old_size + 2, old_size + 2);
            PNew.set_block(0, 0, PEst);
            
            // Calculate new blocks
            // P[new_lm, map] = Jr * P[robot, map]
            Matrix P_robot_map = PEst.get_block(0, 0, 3, old_size);
            Matrix P_new_lm_map = Jr * P_robot_map;
            PNew.set_block(old_size, 0, P_new_lm_map);
            PNew.set_block(0, old_size, P_new_lm_map.transpose());
            
            // P[new_lm, new_lm]
            Matrix P_robot = PEst.get_block(0, 0, 3, 3);
            Matrix P_lm_lm = Jr * P_robot * Jr.transpose() + Jz * R * Jz.transpose();
            PNew.set_block(old_size, old_size, P_lm_lm);
            
            PEst = PNew;
        } else {
            // Update existing landmark
            auto innov_res = calc_innovation(xEst, PEst, z_i, min_id, R);
            Matrix K = PEst * innov_res.H.transpose() * innov_res.S.inv2x2();
            xEst = xEst + K * innov_res.innov;
            
            Matrix I = Matrix::identity(PEst.rows);
            PEst = (I - K * innov_res.H) * PEst;
        }
        
        xEst(2, 0) = pi_2_pi(xEst(2, 0));
    }
    
    return {xEst, PEst};
}

int main() {
    std::cout << "EKF SLAM C++ Library - Minimal Usage Example" << std::endl;

    // 1. Initialize State and Covariance
    double data[15] = {-5.62815962, 18.37374747, -2.54979418,-0.09302817, 4.97980269, 6.11371654, -6.91804677, 10.97515561,  1.16371694, 14.82922   , 10.21498871,  2.74484343, 15.0169684, -5.28442142, 19.95406455};
    Matrix xEst(15, 1); // Robot starts at [0,0,0]
    for (int i = 0; i < 15; ++i) {
        printf("%lf\n", data[i]);
        xEst(i, 0) = data[i];
    }
    Matrix PEst = Matrix::identity(15);

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
    xEst.print();
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