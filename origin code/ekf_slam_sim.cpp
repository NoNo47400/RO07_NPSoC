/**
 * Extended Kalman Filter SLAM example
 * Converted from Python to C++
 * * Original Python Author: Atsushi Sakai (@Atsushi_twi)
 * C++ Conversion: Gemini
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <iomanip>
#include <stdexcept>

// --- Constants ---
const double DT = 0.1;           // time tick [s]
const double SIM_TIME = 50.0;    // simulation time [s]
const double MAX_RANGE = 20.0;   // maximum observation range
const double M_DIST_TH = 2.0;    // Threshold of Mahalanobis distance
const int STATE_SIZE = 3;        // State size [x,y,yaw]
const int LM_SIZE = 2;           // LM state size [x,y]

// --- Matrix Class (Minimal implementation for SLAM) ---
class Matrix {
public:
    int rows;
    int cols;
    std::vector<double> data;

    Matrix(int r, int c, double val = 0.0) : rows(r), cols(c), data(r * c, val) {}

    static Matrix identity(int n) {
        Matrix m(n, n);
        for (int i = 0; i < n; ++i) m(i, i) = 1.0;
        return m;
    }

    static Matrix diag(const std::vector<double>& vals) {
        int n = vals.size();
        Matrix m(n, n);
        for (int i = 0; i < n; ++i) m(i, i) = vals[i];
        return m;
    }

    double& operator()(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) throw std::out_of_range("Matrix index out of bounds");
        return data[r * cols + c];
    }

    const double& operator()(int r, int c) const {
        if (r < 0 || r >= rows || c < 0 || c >= cols) throw std::out_of_range("Matrix index out of bounds");
        return data[r * cols + c];
    }

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix dimension mismatch (+)");
        Matrix res(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) res.data[i] = data[i] + other.data[i];
        return res;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix dimension mismatch (-)");
        Matrix res(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) res.data[i] = data[i] - other.data[i];
        return res;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) throw std::invalid_argument("Matrix dimension mismatch (*)");
        Matrix res(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < cols; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                res(i, j) = sum;
            }
        }
        return res;
    }

    Matrix operator*(double scalar) const {
        Matrix res = *this;
        for (double& v : res.data) v *= scalar;
        return res;
    }

    Matrix transpose() const {
        Matrix res(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                res(j, i) = (*this)(i, j);
            }
        }
        return res;
    }

    // Hardcoded inverse for 2x2 matrices (sufficient for this SLAM implementation's 'S' matrix)
    Matrix inv2x2() const {
        if (rows != 2 || cols != 2) throw std::runtime_error("Only 2x2 inversion implemented");
        double det = (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        if (std::abs(det) < 1e-9) throw std::runtime_error("Matrix is singular");
        Matrix res(2, 2);
        res(0, 0) = (*this)(1, 1) / det;
        res(0, 1) = -(*this)(0, 1) / det;
        res(1, 0) = -(*this)(1, 0) / det;
        res(1, 1) = (*this)(0, 0) / det;
        return res;
    }
    
    // Copy a submatrix into this matrix at (r,c)
    void set_block(int r, int c, const Matrix& other) {
        for(int i=0; i<other.rows; ++i) {
            for(int j=0; j<other.cols; ++j) {
                (*this)(r+i, c+j) = other(i,j);
            }
        }
    }

    // Get a submatrix
    Matrix get_block(int r, int c, int r_len, int c_len) const {
        Matrix res(r_len, c_len);
        for(int i=0; i<r_len; ++i) {
            for(int j=0; j<c_len; ++j) {
                res(i,j) = (*this)(r+i, c+j);
            }
        }
        return res;
    }
};

Matrix operator*(double scalar, const Matrix& m) { return m * scalar; }

// --- Helpers ---

double pi_2_pi(double angle) {
    while (angle >= M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double rand_normal(double mean, double stddev) {
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<> d(mean, stddev);
    return d(gen);
}

int calc_n_lm(const Matrix& x) {
    return (x.rows - STATE_SIZE) / LM_SIZE;
}

// --- Motion Model ---

Matrix calc_input() {
    double v = 1.0;       // [m/s]
    double yawrate = 0.1; // [rad/s]
    Matrix u(2, 1);
    u(0, 0) = v;
    u(1, 0) = yawrate;
    return u;
}

Matrix motion_model(const Matrix& x, const Matrix& u) {
    Matrix F(3, 3);
    F = Matrix::identity(3);
    
    double yaw = x(2, 0);
    Matrix xp = x;
    
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

// Returns {xTrue, z, xDR, u}
// z is N x 3 matrix where rows are [range, angle, landmark_id]
struct ObsResult {
    Matrix xTrue;
    Matrix z;
    Matrix xDR;
    Matrix u;
};

ObsResult observation(Matrix xTrue, Matrix xDR, Matrix u, const Matrix& landmarks, const Matrix& Q_sim, const Matrix& R_sim) {
    xTrue = motion_model(xTrue, u);
    
    std::vector<std::vector<double>> z_vec;
    
    for (int i = 0; i < landmarks.rows; ++i) {
        double dx = landmarks(i, 0) - xTrue(0, 0);
        double dy = landmarks(i, 1) - xTrue(1, 0);
        double d = sqrt(dx*dx + dy*dy);
        double angle = pi_2_pi(atan2(dy, dx) - xTrue(2, 0));
        
        if (d <= MAX_RANGE) {
            double dn = d + rand_normal(0, sqrt(R_sim(0, 0)));
            double anglen = angle + rand_normal(0, sqrt(R_sim(1, 1)));
            z_vec.push_back({dn, anglen, (double)i});
        }
    }
    
    Matrix z(z_vec.size(), 3);
    for(size_t i=0; i<z_vec.size(); ++i) {
        z(i, 0) = z_vec[i][0];
        z(i, 1) = z_vec[i][1];
        z(i, 2) = z_vec[i][2];
    }
    
    // Add noise to input
    Matrix ud(2, 1);
    ud(0, 0) = u(0, 0) + rand_normal(0, sqrt(Q_sim(0, 0)));
    ud(1, 0) = u(1, 0) + rand_normal(0, sqrt(Q_sim(1, 1)));
    
    xDR = motion_model(xDR, ud);
    
    return {xTrue, z, xDR, ud};
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

// Returns {xEst, PEst}
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
            // New Landmark found
            // std::cout << "New LM found!" << std::endl;
            
            // Expand state
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
    std::cout << "EKF SLAM C++ Simulation Start" << std::endl;

    // Simulation Parameters
    Matrix Q_sim = Matrix::diag({0.1*0.1, (1.0*M_PI/180.0)*(1.0*M_PI/180.0)});
    Matrix R_sim = Matrix::diag({0.2*0.2, (5.0*M_PI/180.0)*(5.0*M_PI/180.0)});
    
    Matrix Q = Matrix::diag({0.1, 0.1}); // Q_factor=1 for simplicity
    Matrix R = R_sim * 1.0; 

    // Landmarks
    Matrix landmarks(4, 2);
    landmarks(0,0)=10.0; landmarks(0,1)=-2.0;
    landmarks(1,0)=15.0; landmarks(1,1)=10.0;
    landmarks(2,0)=3.0;  landmarks(2,1)=15.0;
    landmarks(3,0)=-5.0; landmarks(3,1)=20.0;

    // State
    Matrix xEst(STATE_SIZE, 1); // [0,0,0]
    Matrix xTrue(STATE_SIZE, 1);
    Matrix xDR(STATE_SIZE, 1);
    
    Matrix PEst = Matrix::identity(STATE_SIZE);
    PEst(0,0)=1.0; PEst(1,1)=1.0; PEst(2,2)=1.0;

    double time = 0.0;

    while (time <= SIM_TIME) {
        time += DT;
        
        Matrix u = calc_input();
        
        // 1. Simulation (Get observations)
        ObsResult obs = observation(xTrue, xDR, u, landmarks, Q_sim, R_sim);
        xTrue = obs.xTrue;
        xDR = obs.xDR;
        Matrix z = obs.z;
        Matrix ud = obs.u;
        
        // 2. EKF SLAM
        auto result = ekf_slam(xEst, PEst, ud, z, Q, R);
        xEst = result.first;
        PEst = result.second;

        // Logging
        int nLM = calc_n_lm(xEst);
        std::cout << "Time: " << std::fixed << std::setprecision(1) << time 
                  << "s | Est: [" << xEst(0,0) << ", " << xEst(1,0) << "]"
                  << " | True: [" << xTrue(0,0) << ", " << xTrue(1,0) << "]"
                  << " | Landmarks: " << nLM << std::endl;
    }
    
    // Final result
    std::cout << "Final Estimated State:" << std::endl;
    for(int i=0; i<xEst.rows; ++i) std::cout << xEst(i,0) << " ";
    std::cout << std::endl;

    return 0;
}