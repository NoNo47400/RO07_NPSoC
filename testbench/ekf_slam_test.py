from src.ekf_slam import ekf_slam, MAX_ROWS, MAX_COLS

if __name__ == "__main__":
    # Initial Robot State
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
    
    x_res, P_res = ekf_slam(x_in, x_rows, P_in, P_rows, u_in, z_in, Q_in, R_in)
    
    print(f"New State Size: {x_res.rows}")
    print(x_res.get(0,1))
    print(f"Robot State: [{x_res.get(0,0):.4f}, {x_res.get(0,1):.4f}, {x_res.get(0,2):.4f}]")
    # Print landmark if exists
    if x_res.rows > 3:
        print(f"Landmark: [{x_res.get(0,3):.4f}, {x_res.get(0,4):.4f}]")