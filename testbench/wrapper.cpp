#include "../src/ekf_slam.h" // Vérifiez ce chemin relatif !

extern "C" {
    void ekf_slam_bridge(
        data_t x_in[MAX_ROWS], int x_rows,
        data_t P_in[MAX_ROWS*MAX_ROWS], int P_rows,
        data_t u_in[2],
        data_t z_in[3],
        data_t Q_in[2],
        data_t R_in[2],
        data_t x_out[MAX_ROWS], int *x_rows_out,       
        data_t P_out[MAX_ROWS*MAX_ROWS], int *P_rows_out 
    ) {
        // On déréférence les pointeurs (*) pour passer les valeurs aux références (&) C++
        // C'est ici que la magie opère pour renvoyer la taille modifiée
        ekf_slam_top(x_in, x_rows, P_in, P_rows, u_in, z_in, Q_in, R_in, 
                     x_out, *x_rows_out, P_out, *P_rows_out);
    }
}