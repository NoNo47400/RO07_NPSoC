#ifndef EKF_SLAM_HLS_H
#define EKF_SLAM_HLS_H

#include <cmath>

// --- HLS Constants ---
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Configuration de la taille (Compromis Surface vs CapacitÃ©)
#define MAX_LANDMARKS 5
#define STATE_SIZE 3
#define LM_SIZE 2
// Taille maximale absolue des matrices (Robots + 5 Landmarks = 3 + 10 = 13)
#define MAX_ROWS 13
#define MAX_COLS 13

// --- Algorithm Constants ---
const float DT = 0.1;
const float M_DIST_TH = 2.0;

typedef float data_t; // Peut Ãªtre changÃ© en float ou ap_fixed pour gagner de la surface

// --- Classe Matrix SynthÃ©tisable ---
struct Matrix {
    int rows;
    int cols;
    data_t data[MAX_ROWS * MAX_COLS];

    // Constructeurs
    Matrix() : rows(0), cols(0) {
		#pragma HLS RESOURCE variable=data core=RAM_1P_BRAM
        for(int i=0; i<MAX_ROWS*MAX_COLS; i++) data[i] = 0.0;
    }

    // AccÃ¨s (linÃ©arisation 2D -> 1D)
    data_t& at(int r, int c) {
        return data[r * MAX_COLS + c];
    }
    
    data_t get(int r, int c) const {
        return data[r * MAX_COLS + c];
    }

    // Initialisation IdentitÃ©
    static void identity(int n, Matrix &res) {
        res.rows = n; res.cols = n;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                res.at(i, j) = (i == j) ? 1.0 : 0.0;
            }
        }
    }
};

// Prototypes des fonctions Kernel
void ekf_slam_top(
    data_t x_in[MAX_ROWS], int x_rows,
    data_t P_in[MAX_ROWS*MAX_ROWS], int P_rows,
    data_t u_in[2],
    data_t z_in[3], // Une seule observation pour cet exemple (extensible)
    data_t Q_in[2], // Diagonale de Q
    data_t R_in[2], // Diagonale de R
    data_t x_out[MAX_ROWS], int &x_rows_out,
    data_t P_out[MAX_ROWS*MAX_ROWS], int &P_rows_out
);

#endif
