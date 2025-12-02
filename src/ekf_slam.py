import math

# --- HLS Constants ---
M_PI = 3.14159265358979323846
MAX_LANDMARKS = 5
STATE_SIZE = 3
LM_SIZE = 2
MAX_ROWS = 13  # 3 + 2*5 = 13
MAX_COLS = 13  # Carré pour simplifier l'adressage mémoire comme en HLS

# --- Algorithm Constants ---
DT = 0.1
M_DIST_TH = 2.0

# --- Classe Matrix Synthétisable (Simulée) ---
class Matrix:
    def __init__(self):
        self.rows = 0
        self.cols = 0
        # Simulation du tableau statique C++ : data[MAX_ROWS * MAX_COLS]
        # On initialise tout à 0.0 float
        self.data = [0.0] * (MAX_ROWS * MAX_COLS)

    def get(self, r, c):
        # Accès lecture : data[r * MAX_COLS + c]
        return self.data[r * MAX_COLS + c]

    def set(self, r, c, val):
        # Accès écriture
        self.data[r * MAX_COLS + c] = val

    @staticmethod
    def identity(n, res):
        res.rows = n
        res.cols = n
        for i in range(n):
            for j in range(n):
                val = 1.0 if i == j else 0.0
                res.set(i, j, val)

# --- Helper Functions ---

def pi_2_pi(angle):
    while angle >= M_PI:
        angle -= 2.0 * M_PI
    while angle < -M_PI:
        angle += 2.0 * M_PI
    return angle

# --- Opérations Matricielles Basiques (Style HLS) ---

def mat_add(A, B, C):
    C.rows = A.rows
    C.cols = A.cols
    # Boucle fixe comme en HLS
    for i in range(MAX_ROWS * MAX_COLS):
        # Bound check logique simulant la boucle pipelinée
        if i < (A.rows * A.cols): 
            # Comme A, B, C ont la même structure mémoire linéaire
            C.data[i] = A.data[i] + B.data[i]

def mat_sub(A, B, C):
    C.rows = A.rows
    C.cols = A.cols
    for i in range(MAX_ROWS * MAX_COLS):
        if i < (A.rows * A.cols):
            C.data[i] = A.data[i] - B.data[i]

def mat_mul(A, B, C):
    C.rows = A.rows
    C.cols = B.cols
    # Triple boucle explicite
    for i in range(MAX_ROWS):
        for j in range(MAX_ROWS):
            if i < A.rows and j < B.cols:
                sum_val = 0.0
                for k in range(MAX_ROWS):
                    if k < A.cols:
                        sum_val += A.get(i, k) * B.get(k, j)
                C.set(i, j, sum_val)

def mat_transpose(A, C):
    C.rows = A.cols
    C.cols = A.rows
    for i in range(MAX_ROWS):
        for j in range(MAX_ROWS):
            if i < A.rows and j < A.cols:
                C.set(j, i, A.get(i, j))

def mat_inv2x2(A, C):
    # Inversion sans try/catch, retour booléen
    det = A.get(0,0) * A.get(1,1) - A.get(0,1) * A.get(1,0)
    if abs(det) < 1e-6:
        return False
    
    invDet = 1.0 / det
    C.rows = 2
    C.cols = 2
    C.set(0,0,  A.get(1,1) * invDet)
    C.set(0,1, -A.get(0,1) * invDet)
    C.set(1,0, -A.get(1,0) * invDet)
    C.set(1,1,  A.get(0,0) * invDet)
    return True

def get_block(src, r, c, r_len, c_len, dst):
    dst.rows = r_len
    dst.cols = c_len
    for i in range(MAX_ROWS):
        for j in range(MAX_ROWS):
            if i < r_len and j < c_len:
                dst.set(i, j, src.get(r + i, c + j))

def set_block(dst, r, c, src):
    for i in range(MAX_ROWS):
        for j in range(MAX_ROWS):
            if i < src.rows and j < src.cols:
                dst.set(r + i, c + j, src.get(i, j))

# --- EKF Logic Helpers ---

def calc_n_lm(x):
    return int((x.rows - STATE_SIZE) / LM_SIZE)

def motion_model(x, u, x_pred):
    # Copie manuelle
    x_pred.rows = x.rows
    x_pred.cols = x.cols
    for i in range(MAX_ROWS * MAX_COLS):
        x_pred.data[i] = x.data[i]
        
    yaw = x.get(2, 0)
    
    # x += v * dt * cos(yaw)
    val0 = x_pred.get(0, 0) + u.get(0, 0) * DT * math.cos(yaw)
    x_pred.set(0, 0, val0)
    
    # y += v * dt * sin(yaw)
    val1 = x_pred.get(1, 0) + u.get(0, 0) * DT * math.sin(yaw)
    x_pred.set(1, 0, val1)
    
    # yaw += yaw_rate * dt
    val2 = x_pred.get(2, 0) + u.get(1, 0) * DT
    x_pred.set(2, 0, pi_2_pi(val2))

def jacob_motion(x, u, A, B):
    Matrix.identity(3, A)
    yaw = x.get(2, 0)
    v = u.get(0, 0)
    
    A.set(0, 2, -DT * v * math.sin(yaw))
    A.set(1, 2,  DT * v * math.cos(yaw))
    
    B.rows = 3
    B.cols = 2
    # Reset B data (only needed parts or full)
    for i in range(6): 
        B.data[i] = 0.0
        
    B.set(0, 0, DT * math.cos(yaw))
    B.set(1, 0, DT * math.sin(yaw))
    B.set(2, 1, DT)

def jacob_h(q, delta, x, i, H):
    sq = math.sqrt(q)
    
    # G (2x5) local Jacobian data
    G_data = [0.0] * 10
    dx = delta.get(0,0)
    dy = delta.get(1,0)
    
    G_data[0] = -sq * dx; G_data[1] = -sq * dy; G_data[2] = 0.0; G_data[3] = sq * dx; G_data[4] = sq * dy
    G_data[5] = dy;       G_data[6] = -dx;      G_data[7] = -q;  G_data[8] = -dy;     G_data[9] = dx
    
    nLM = calc_n_lm(x)
    H.rows = 2
    H.cols = 3 + 2 * nLM
    
    # Reset H
    for k in range(MAX_ROWS * 2):
        H.data[k] = 0.0
        
    inv_q = 1.0 / q
    
    # Partie Robot
    H.set(0,0, G_data[0] * inv_q); H.set(0,1, G_data[1] * inv_q); H.set(0,2, G_data[2] * inv_q)
    H.set(1,0, G_data[5] * inv_q); H.set(1,1, G_data[6] * inv_q); H.set(1,2, G_data[7] * inv_q)
    
    # Partie Landmark
    lm_idx = 3 + 2 * i
    H.set(0, lm_idx,   G_data[3] * inv_q); H.set(0, lm_idx+1, G_data[4] * inv_q)
    H.set(1, lm_idx,   G_data[8] * inv_q); H.set(1, lm_idx+1, G_data[9] * inv_q)

def calc_innovation(xEst, PEst, z_meas, lm_id, R, innov, S, H):
    # Extraction Landmark state
    start = STATE_SIZE + LM_SIZE * lm_id
    lm_pos = Matrix()
    get_block(xEst, start, 0, LM_SIZE, 1, lm_pos)
    
    robot_pos = Matrix()
    get_block(xEst, 0, 0, 2, 1, robot_pos)
    
    delta = Matrix()
    mat_sub(lm_pos, robot_pos, delta)
    
    q = delta.get(0,0)*delta.get(0,0) + delta.get(1,0)*delta.get(1,0)
    z_angle = math.atan2(delta.get(1, 0), delta.get(0, 0)) - xEst.get(2, 0)
    
    zp = Matrix()
    zp.rows = 2
    zp.cols = 1
    zp.set(0, 0, math.sqrt(q))
    zp.set(1, 0, pi_2_pi(z_angle))
    
    mat_sub(z_meas, zp, innov)
    val_angle = pi_2_pi(innov.get(1, 0))
    innov.set(1, 0, val_angle)
    
    jacob_h(q, delta, xEst, lm_id, H)
    
    # S = H * P * Ht + R
    Ht = Matrix()
    HP = Matrix()
    HPHt = Matrix()
    
    mat_transpose(H, Ht)
    mat_mul(H, PEst, HP)
    mat_mul(HP, Ht, HPHt)
    mat_add(HPHt, R, S)

# --- TOP LEVEL FUNCTION ---

def ekf_slam_top(x_in, x_rows, P_in, P_rows, u_in, z_in, Q_in, R_in):
    # Outputs simulés (modifiés par référence en C++, ici on retourne les objets)
    
    # 1. Reconstruction des objets Matrix locaux
    xEst = Matrix()
    xEst.rows = x_rows
    xEst.cols = 1
    for i in range(MAX_ROWS):
        xEst.data[i] = x_in[i]
        
    PEst = Matrix()
    PEst.rows = P_rows
    PEst.cols = P_rows
    for i in range(MAX_ROWS * MAX_ROWS):
        PEst.data[i] = P_in[i]
        
    u = Matrix()
    u.rows = 2; u.cols = 1
    u.set(0,0, u_in[0]); u.set(1,0, u_in[1])
    
    Q = Matrix()
    Q.rows = 2; Q.cols = 2
    Q.set(0,0, Q_in[0]); Q.set(1,1, Q_in[1])
    
    R = Matrix()
    R.rows = 2; R.cols = 2
    R.set(0,0, R_in[0]); R.set(1,1, R_in[1])
    
    # --- PREDICTION ---
    S_size = 3
    x_robot = Matrix()
    get_block(xEst, 0, 0, S_size, 1, x_robot)
    
    A = Matrix()
    B = Matrix()
    jacob_motion(x_robot, u, A, B)
    
    xPred = Matrix()
    motion_model(x_robot, u, xPred)
    set_block(xEst, 0, 0, xPred)
    
    # Update P
    Pxx = Matrix()
    get_block(PEst, 0, 0, S_size, S_size, Pxx)
    
    At = Matrix(); Bt = Matrix()
    AP = Matrix(); APA = Matrix()
    BQ = Matrix(); BQB = Matrix()
    
    mat_transpose(A, At)
    mat_transpose(B, Bt)
    
    mat_mul(A, Pxx, AP)
    mat_mul(AP, At, APA)
    
    mat_mul(B, Q, BQ)
    mat_mul(BQ, Bt, BQB)
    
    Pxx_new = Matrix()
    mat_add(APA, BQB, Pxx_new)
    set_block(PEst, 0, 0, Pxx_new)
    
    # Cross covariance
    if PEst.rows > S_size:
        Pxm = Matrix()
        cols_remaining = PEst.cols - S_size
        get_block(PEst, 0, S_size, S_size, cols_remaining, Pxm)
        
        Pxm_new = Matrix()
        mat_mul(A, Pxm, Pxm_new)
        set_block(PEst, 0, S_size, Pxm_new)
        
        Pxm_new_t = Matrix()
        mat_transpose(Pxm_new, Pxm_new_t)
        set_block(PEst, S_size, 0, Pxm_new_t)
        
    xEst.set(2, 0, pi_2_pi(xEst.get(2, 0)))
    
    # --- UPDATE ---
    z_i = Matrix()
    z_i.rows = 2; z_i.cols = 1
    z_i.set(0,0, z_in[0])
    z_i.set(1,0, z_in[1])
    
    nLM = calc_n_lm(xEst)
    min_id = nLM
    min_dist = M_DIST_TH
    
    # Data Association Loop
    for i in range(MAX_LANDMARKS):
        if i < nLM:
            innov = Matrix(); S_mat = Matrix(); H = Matrix()
            calc_innovation(xEst, PEst, z_i, i, R, innov, S_mat, H)
            
            S_inv = Matrix()
            success = mat_inv2x2(S_mat, S_inv)
            
            if success:
                innov_t = Matrix(); dist_tmp = Matrix(); Sinv_innov = Matrix()
                mat_transpose(innov, innov_t)
                mat_mul(S_inv, innov, Sinv_innov)
                mat_mul(innov_t, Sinv_innov, dist_tmp)
                
                dist = dist_tmp.get(0,0)
                if dist < min_dist:
                    min_dist = dist
                    min_id = i

    if min_id == nLM and nLM < MAX_LANDMARKS:
        # --- ADD NEW LANDMARK ---
        r = z_i.get(0,0)
        angle = z_i.get(1,0)
        yaw = xEst.get(2,0)
        
        old_rows = xEst.rows
        xEst.rows += 2
        xEst.set(old_rows, 0,   xEst.get(0,0) + r * math.cos(yaw + angle))
        xEst.set(old_rows+1, 0, xEst.get(1,0) + r * math.sin(yaw + angle))
        
        # Extend Covariance
        Jr = Matrix(); Jr.rows=2; Jr.cols=3
        Jr.set(0,0, 1); Jr.set(0,1, 0); Jr.set(0,2, -r*math.sin(yaw+angle))
        Jr.set(1,0, 0); Jr.set(1,1, 1); Jr.set(1,2,  r*math.cos(yaw+angle))
        
        Jz = Matrix(); Jz.rows=2; Jz.cols=2
        Jz.set(0,0, math.cos(yaw+angle)); Jz.set(0,1, -r*math.sin(yaw+angle))
        Jz.set(1,0, math.sin(yaw+angle)); Jz.set(1,1,  r*math.cos(yaw+angle))
        
        PEst.rows += 2
        PEst.cols += 2
        
        P_robot_map = Matrix()
        get_block(PEst, 0, 0, 3, old_rows, P_robot_map)
        P_new_lm_map = Matrix()
        mat_mul(Jr, P_robot_map, P_new_lm_map)
        
        set_block(PEst, old_rows, 0, P_new_lm_map)
        
        P_new_lm_map_t = Matrix()
        mat_transpose(P_new_lm_map, P_new_lm_map_t)
        set_block(PEst, 0, old_rows, P_new_lm_map_t)
        
        P_robot = Matrix()
        get_block(PEst, 0, 0, 3, 3, P_robot)
        
        JrP = Matrix(); JrPJrt = Matrix(); JzR = Matrix(); JzRJzt = Matrix(); P_lmlm = Matrix()
        Jrt = Matrix(); Jzt = Matrix()
        
        mat_transpose(Jr, Jrt)
        mat_transpose(Jz, Jzt)
        
        mat_mul(Jr, P_robot, JrP)
        mat_mul(JrP, Jrt, JrPJrt)
        mat_mul(Jz, R, JzR)
        mat_mul(JzR, Jzt, JzRJzt)
        mat_add(JrPJrt, JzRJzt, P_lmlm)
        
        set_block(PEst, old_rows, old_rows, P_lmlm)
        
    elif min_id < nLM:
        # --- UPDATE EXISTING ---
        innov = Matrix(); S_mat = Matrix(); H = Matrix()
        calc_innovation(xEst, PEst, z_i, min_id, R, innov, S_mat, H)
        
        S_inv = Matrix()
        mat_inv2x2(S_mat, S_inv)
        
        Ht = Matrix(); PHt = Matrix(); K = Matrix()
        mat_transpose(H, Ht)
        mat_mul(PEst, Ht, PHt)
        mat_mul(PHt, S_inv, K)
        
        K_innov = Matrix()
        mat_mul(K, innov, K_innov)
        mat_add(xEst, K_innov, xEst)
        
        KH = Matrix(); I = Matrix(); I_KH = Matrix(); P_new = Matrix()
        Matrix.identity(PEst.rows, I)
        mat_mul(K, H, KH)
        mat_sub(I, KH, I_KH)
        mat_mul(I_KH, PEst, P_new)
        PEst = P_new # Copie superficielle mais data est nouvelle instance
    
    xEst.set(2, 0, pi_2_pi(xEst.get(2, 0)))
    
    # Retourne les structures pour affichage/debug
    return xEst, PEst
