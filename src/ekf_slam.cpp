#include "ekf_slam.h"
#include <cstdio>

// --- Helper Functions ---

// Angle Normalization
float pi_2_pi(float angle) {
    while (angle >= M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// Functions for Matrix Operations without using dynamic memory -> To synthesize in HLS
void mat_add(const Matrix &A, const Matrix &B, Matrix &C) {
    C.rows = A.rows; C.cols = A.cols;
    loop_add: for (int i = 0; i < MAX_ROWS; i++) {
        if (i < A.rows * A.cols) // Bound check logique
             C.data[i] = A.data[i] + B.data[i];
    }
}

void mat_sub(const Matrix &A, const Matrix &B, Matrix &C) {
    C.rows = A.rows; C.cols = A.cols;
    loop_sub: for (int i = 0; i < MAX_ROWS; i++) {
        if (i < A.rows * A.cols)
            C.data[i] = A.data[i] - B.data[i];
    }
}

void mat_mul(const Matrix &A, const Matrix &B, Matrix &C) {
    C.rows = A.rows; C.cols = B.cols;
    loop_mul_row: for (int i = 0; i < MAX_ROWS; ++i) {
        loop_mul_col: for (int j = 0; j < MAX_ROWS; ++j) {
            if (i < A.rows && j < B.cols) {
                data_t sum = 0;
                loop_mul_inner: for (int k = 0; k < MAX_ROWS; ++k) {
                    if (k < A.cols)
                        sum += A.get(i, k) * B.get(k, j);
                }
                C.at(i, j) = sum;
            }
        }
    }
}

void mat_transpose(const Matrix &A, Matrix &C) {
    C.rows = A.cols; C.cols = A.rows;
    loop_trans_row: for(int i=0; i<MAX_ROWS; i++) {
        loop_trans_col: for(int j=0; j<MAX_ROWS; j++) {
            if(i < A.rows && j < A.cols)
                C.at(j,i) = A.get(i,j);
        }
    }
}

// Inversion 2x2
bool mat_inv2x2(const Matrix &A, Matrix &C) {
    data_t det = A.get(0,0)*A.get(1,1) - A.get(0,1)*A.get(1,0);
    if (std::abs(det) < 1e-6) return false;
    data_t invDet = 1.0 / det;
    C.rows = 2; C.cols = 2;
    C.at(0,0) =  A.get(1,1) * invDet;
    C.at(0,1) = -A.get(0,1) * invDet;
    C.at(1,0) = -A.get(1,0) * invDet;
    C.at(1,1) =  A.get(0,0) * invDet;
    return true;
}

void get_block(const Matrix &src, int r, int c, int r_len, int c_len, Matrix &dst) {
    dst.rows = r_len; dst.cols = c_len;
    for(int i=0; i<MAX_ROWS; i++) {
        for(int j=0; j<MAX_ROWS; j++) {
            if(i < r_len && j < c_len)
                dst.at(i,j) = src.get(r+i, c+j);
        }
    }
}

void set_block(Matrix &dst, int r, int c, const Matrix &src) {
    for(int i=0; i<MAX_ROWS; i++) {
        for(int j=0; j<MAX_ROWS; j++) {
            if(i < src.rows && j < src.cols)
                dst.at(r+i, c+j) = src.get(i,j);
        }
    }
}

// --- EKF Logic Helpers ---

int calc_n_lm(const Matrix& x) {
    return (x.rows - STATE_SIZE) / LM_SIZE;
}

void motion_model(const Matrix& x, const Matrix& u, Matrix &x_pred) {
    x_pred = x; // Copy
    float yaw = x.get(2, 0);
    x_pred.at(0, 0) += u.get(0, 0) * DT * std::cos(yaw);
    x_pred.at(1, 0) += u.get(0, 0) * DT * std::sin(yaw);
    x_pred.at(2, 0) += u.get(1, 0) * DT;
    x_pred.at(2, 0) = pi_2_pi(x_pred.get(2, 0));
}

void jacob_motion(const Matrix& x, const Matrix& u, Matrix &A, Matrix &B) {
    Matrix::identity(3, A);
    float yaw = x.get(2, 0);
    float v = u.get(0, 0);
    
    A.at(0, 2) = -DT * v * std::sin(yaw);
    A.at(1, 2) = DT * v * std::cos(yaw);
    
    B.rows = 3; B.cols = 2;
    // Reset B data to calculate new jacobian
    for(int i=0; i<6; i++) B.data[i] = 0; 
    
    B.at(0, 0) = DT * std::cos(yaw);
    B.at(1, 0) = DT * std::sin(yaw);
    B.at(2, 1) = DT;
}

void jacob_h(float q, const Matrix& delta, const Matrix& x, int i, Matrix &H) {
    // We are calculating H = G*F without building F (which is too large)
    float sq = std::sqrt(q);
    
    // G (2x5) local Jacobian
    data_t G_data[10];
    G_data[0] = -sq * delta.get(0,0); G_data[1] = -sq * delta.get(1,0); G_data[2] = 0; G_data[3] = sq * delta.get(0,0); G_data[4] = sq * delta.get(1,0);
    G_data[5] = delta.get(1,0);       G_data[6] = -delta.get(0,0);      G_data[7] = -q; G_data[8] = -delta.get(1,0);    G_data[9] = delta.get(0,0);
    
    int nLM = calc_n_lm(x);
    // H (2 x Total_States)
    H.rows = 2; H.cols = 3 + 2 * nLM;
    
    // Reset H
    for(int k=0; k<MAX_ROWS*2; k++) H.data[k] = 0.0;

    float inv_q = 1.0 / q;

    // Filling Robot part (columns 0,1,2)
    // G * I(3) -> The first 3 columns of G divided by q
    H.at(0,0) = G_data[0] * inv_q; H.at(0,1) = G_data[1] * inv_q; H.at(0,2) = G_data[2] * inv_q;
    H.at(1,0) = G_data[5] * inv_q; H.at(1,1) = G_data[6] * inv_q; H.at(1,2) = G_data[7] * inv_q;

    // Filling Landmark part (columns corresponding to ID i)
    int lm_idx = 3 + 2 * i;

    H.at(0, lm_idx)   = G_data[3] * inv_q; H.at(0, lm_idx+1) = G_data[4] * inv_q;
    H.at(1, lm_idx)   = G_data[8] * inv_q; H.at(1, lm_idx+1) = G_data[9] * inv_q;
}

// Calculation of innovation and associated matrices
void calc_innovation(const Matrix& xEst, const Matrix& PEst, const Matrix& z_meas, int lm_id, const Matrix& R,
                     Matrix &innov, Matrix &S, Matrix &H) 
{
    // Extraction Landmark state
    int start = STATE_SIZE + LM_SIZE * lm_id;
    Matrix lm_pos; 
    get_block(xEst, start, 0, LM_SIZE, 1, lm_pos);
    
    Matrix robot_pos;
    get_block(xEst, 0, 0, 2, 1, robot_pos);
    
    Matrix delta;
    mat_sub(lm_pos, robot_pos, delta);
    
    float q = delta.get(0,0)*delta.get(0,0) + delta.get(1,0)*delta.get(1,0);
    float z_angle = std::atan2(delta.get(1, 0), delta.get(0, 0)) - xEst.get(2, 0);
    
    Matrix zp; zp.rows=2; zp.cols=1;
    zp.at(0, 0) = std::sqrt(q);
    zp.at(1, 0) = pi_2_pi(z_angle);
    
    mat_sub(z_meas, zp, innov);
    innov.at(1, 0) = pi_2_pi(innov.get(1, 0));
    
    jacob_h(q, delta, xEst, lm_id, H);
    
    // S = H * P * Ht + R
    Matrix Ht, HP, HPHt;
    mat_transpose(H, Ht);
    mat_mul(H, PEst, HP);
    mat_mul(HP, Ht, HPHt);
    mat_add(HPHt, R, S);
}

// EKF SLAM Function

void ekf_slam(
    data_t x_in[MAX_ROWS], int x_rows,
    data_t P_in[MAX_ROWS*MAX_ROWS], int P_rows,
    data_t u_in[2],
    data_t z_in[3], 
    data_t Q_in[2], 
    data_t R_in[2],
    data_t x_out[MAX_ROWS], int &x_rows_out,
    data_t P_out[MAX_ROWS*MAX_ROWS], int &P_rows_out
) {
    Matrix xEst; xEst.rows = x_rows; xEst.cols = 1;
    for(int i=0; i<x_rows; i++) xEst.at(i, 0) = x_in[i];

    Matrix PEst; PEst.rows = P_rows; PEst.cols = P_rows;
    for(int i=0; i<MAX_ROWS*MAX_ROWS; i++) PEst.data[i] = P_in[i]; // P is in 13x13 (explained in .h)

    Matrix u; u.rows = 2; u.cols = 1; 
    u.at(0,0) = u_in[0]; u.at(1,0) = u_in[1]; 
    
    Matrix Q; Q.rows=2; Q.cols=2; Q.at(0,0)=Q_in[0]; Q.at(1,1)=Q_in[1]; Q.at(0,1)=0; Q.at(1,0)=0;
    Matrix R; R.rows=2; R.cols=2; R.at(0,0)=R_in[0]; R.at(1,1)=R_in[1]; R.at(0,1)=0; R.at(1,0)=0;
    
    // --- PREDICTION ---
    int S = 3; 
    Matrix x_robot; get_block(xEst, 0, 0, S, 1, x_robot);
    
    Matrix A, B;
    jacob_motion(x_robot, u, A, B);
    
    Matrix xPred;
    motion_model(x_robot, u, xPred);
    set_block(xEst, 0, 0, xPred);
    
    Matrix Pxx; get_block(PEst, 0, 0, S, S, Pxx);
    Matrix At, Bt, AP, APA, BQ, BQB;
    mat_transpose(A, At); mat_transpose(B, Bt);
    mat_mul(A, Pxx, AP); mat_mul(AP, At, APA);
    mat_mul(B, Q, BQ); mat_mul(BQ, Bt, BQB);
    
    Matrix Pxx_new;
    mat_add(APA, BQB, Pxx_new);
    set_block(PEst, 0, 0, Pxx_new);
    
    if (PEst.rows > S) {
        Matrix Pxm; 
        get_block(PEst, 0, S, S, PEst.cols - S, Pxm);
        Matrix Pxm_new;
        mat_mul(A, Pxm, Pxm_new);
        set_block(PEst, 0, S, Pxm_new);
        
        Matrix Pxm_new_t;
        mat_transpose(Pxm_new, Pxm_new_t);
        set_block(PEst, S, 0, Pxm_new_t);
    }
    xEst.at(2, 0) = pi_2_pi(xEst.get(2, 0));

    // --- UPDATE ---
    // Ignore high measurements to simplify and match real sensor behavior
    if (z_in[0] < 20.0) {
        Matrix z_i; z_i.rows=2; z_i.cols=1; 
        z_i.at(0,0) = z_in[0];
        z_i.at(1,0) = z_in[1];

        int nLM = calc_n_lm(xEst);
        int min_id = nLM;
        float min_dist = M_DIST_TH;

        loop_da: for (int i = 0; i < MAX_LANDMARKS; ++i) {
            if (i < nLM) {
                Matrix innov, S_mat, H;
                calc_innovation(xEst, PEst, z_i, i, R, innov, S_mat, H);
                Matrix S_inv;
                if (mat_inv2x2(S_mat, S_inv)) {
                    Matrix innov_t, dist_tmp, Sinv_innov;
                    mat_transpose(innov, innov_t);
                    mat_mul(S_inv, innov, Sinv_innov);
                    mat_mul(innov_t, Sinv_innov, dist_tmp);
                    float dist = dist_tmp.get(0,0);
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_id = i;
                    }
                }
            }
        }

        // We are limiting the number of landmarks to MAX_LANDMARKS -> Maybe it would be better to overwrite the old ones
        if (min_id == nLM && nLM < MAX_LANDMARKS) {
            printf("[CPP] NEW LANDMARK FOUND! ID=%d Range=%.2f\n", nLM, z_in[0]);
            float r = z_i.get(0,0);
            float angle = z_i.get(1,0);
            float yaw = xEst.get(2,0);
            
            int old_rows = xEst.rows;
            xEst.rows += 2; 
            xEst.at(old_rows, 0)   = xEst.get(0,0) + r * std::cos(yaw + angle);
            xEst.at(old_rows+1, 0) = xEst.get(1,0) + r * std::sin(yaw + angle);

            PEst.rows += 2; PEst.cols += 2;
            
            Matrix Jr; Jr.rows=2; Jr.cols=3;
            Jr.at(0,0)=1; Jr.at(0,1)=0; Jr.at(0,2)=-r*std::sin(yaw+angle);
            Jr.at(1,0)=0; Jr.at(1,1)=1; Jr.at(1,2)= r*std::cos(yaw+angle);
            
            Matrix Jz; Jz.rows=2; Jz.cols=2;
            Jz.at(0,0)=std::cos(yaw+angle); Jz.at(0,1)=-r*std::sin(yaw+angle);
            Jz.at(1,0)=std::sin(yaw+angle); Jz.at(1,1)= r*std::cos(yaw+angle);

            Matrix P_robot_map; get_block(PEst, 0, 0, 3, old_rows, P_robot_map);
            Matrix P_new_lm_map; mat_mul(Jr, P_robot_map, P_new_lm_map);
            
            set_block(PEst, old_rows, 0, P_new_lm_map);
            Matrix P_new_lm_map_t; mat_transpose(P_new_lm_map, P_new_lm_map_t);
            set_block(PEst, 0, old_rows, P_new_lm_map_t);
            
            Matrix P_robot; get_block(PEst, 0, 0, 3, 3, P_robot);
            Matrix JrP, JrPJrt, JzR, JzRJzt, P_lmlm;
            Matrix Jrt, Jzt;
            mat_transpose(Jr, Jrt); mat_transpose(Jz, Jzt);
            
            mat_mul(Jr, P_robot, JrP); mat_mul(JrP, Jrt, JrPJrt);
            mat_mul(Jz, R, JzR); mat_mul(JzR, Jzt, JzRJzt);
            mat_add(JrPJrt, JzRJzt, P_lmlm);
            set_block(PEst, old_rows, old_rows, P_lmlm);

        } else if (min_id < nLM) {
            Matrix innov, S_mat, H;
            calc_innovation(xEst, PEst, z_i, min_id, R, innov, S_mat, H);
            Matrix S_inv; mat_inv2x2(S_mat, S_inv); 
            Matrix Ht, PHt, K;
            mat_transpose(H, Ht); mat_mul(PEst, Ht, PHt); mat_mul(PHt, S_inv, K); 
            Matrix K_innov; mat_mul(K, innov, K_innov);
            mat_add(xEst, K_innov, xEst);
            Matrix KH, I, I_KH, P_new;
            Matrix::identity(PEst.rows, I);
            mat_mul(K, H, KH); mat_sub(I, KH, I_KH); mat_mul(I_KH, PEst, P_new);
            PEst = P_new;
        }
        xEst.at(2, 0) = pi_2_pi(xEst.get(2, 0));
    } 

    x_rows_out = xEst.rows;
    P_rows_out = PEst.rows;
    
    for(int i=0; i<x_rows_out; i++) x_out[i] = xEst.at(i, 0); 
    
    for(int i=0; i<MAX_ROWS*MAX_ROWS; i++) P_out[i] = PEst.data[i];
}