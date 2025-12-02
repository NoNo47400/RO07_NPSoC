from src.ekf_slam import ekf_slam_top, MAX_ROWS, MAX_COLS

# --- TESTBENCH SIMPLIFIÉ ---
if __name__ == "__main__":
    # Buffers d'entrée (Arrays plats)
    x_in = [0.0] * MAX_ROWS
    x_in[0] = -5.628
    x_in[1] = 18.373
    x_in[2] = -2.549
    x_rows = 3
    
    P_in = [0.0] * (MAX_ROWS * MAX_ROWS)
    # Identité 3x3 sur P_in
    for i in range(3):
        P_in[i * MAX_COLS + i] = 1.0
    P_rows = 3
    
    u_in = [1.0, 0.1]
    z_in = [10.0, 0.0, 0.0]
    Q_in = [0.1, 0.1]
    R_in = [0.04, 0.0076]
    
    print("Running EKF Step (Python fixed-size)...")
    
    x_res, P_res = ekf_slam_top(x_in, x_rows, P_in, P_rows, u_in, z_in, Q_in, R_in)
    
    print(f"New State Size: {x_res.rows}")
    print(f"Robot State: [{x_res.get(0,0):.4f}, {x_res.get(1,0):.4f}, {x_res.get(2,0):.4f}]")
    
    if x_res.rows > 3:
        print(f"Landmark: [{x_res.get(3,0):.4f}, {x_res.get(4,0):.4f}]")