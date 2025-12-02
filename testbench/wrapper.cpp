// wrapper.cpp
#include "../src/ekf_slam.h"

extern "C" {
    // On utilise des pointeurs (*) pour les variables passées par référence (&)
    void ekf_slam_bridge(
        data_t x_in[MAX_ROWS], int x_rows,
        data_t P_in[MAX_ROWS*MAX_ROWS], int P_rows,
        data_t u_in[2],
        data_t z_in[3],
        data_t Q_in[2],
        data_t R_in[2],
        data_t x_out[MAX_ROWS], int *x_rows_out,       // <-- Pointeur pour récupérer la sortie
        data_t P_out[MAX_ROWS*MAX_ROWS], int *P_rows_out // <-- Pointeur pour récupérer la sortie
    ) {
        // Appel de la fonction C++ réelle
        ekf_slam_top(x_in, x_rows, P_in, P_rows, u_in, z_in, Q_in, R_in, 
                     x_out, *x_rows_out, P_out, *P_rows_out);
    }
}