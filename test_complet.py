import ctypes
import os
import random
import math
import sys

# Import de votre version Python (assurez-vous que le fichier s'appelle ekf_slam.py)
# et qu'il est dans le même dossier ou accessible
try:
    import src.ekf_slam as py_slam
except ImportError:
    print("Erreur: ekf_slam.py introuvable. Vérifiez l'emplacement.")
    sys.exit(1)

# --- 1. DÉFINITION DE LA CARTE (VÉRITÉ TERRAIN) ---
# Le robot commence à (0,0). On place des amers devant lui.
TRUE_LANDMARKS = [
    [10.0, 2.0],   # Landmark #0 (A 10m devant, un peu à gauche)
    [20.0, -5.0],  # Landmark #1 (A 20m, à droite)
    [30.0, 0.0]    # Landmark #2
]

# --- 2. CONFIGURATION CTYPES ---
lib_name = "./libekf.so" if os.name != 'nt' else "./libekf.dll"
if not os.path.exists(lib_name):
    print(f"ERREUR: {lib_name} introuvable. Compilez avec:")
    print("g++ -shared -fPIC -o libekf.so ekf_slam.cpp wrapper.cpp")
    sys.exit(1)

cpp_lib = ctypes.CDLL(lib_name)

# Définition des types C++
c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)

# Signature de la fonction Wrapper
cpp_lib.ekf_slam_bridge.argtypes = [
    c_float_p, ctypes.c_int,  # x_in, x_rows
    c_float_p, ctypes.c_int,  # P_in, P_rows
    c_float_p,                # u_in
    c_float_p,                # z_in
    c_float_p,                # Q_in
    c_float_p,                # R_in
    c_float_p, c_int_p,       # x_out, x_rows_out (pointeur)
    c_float_p, c_int_p        # P_out, P_rows_out (pointeur)
]

def run_cpp_slam(x, x_rows, P, P_rows, u, z, Q, R):
    """Fonction helper pour appeler le C++ proprement"""
    # Conversion list -> C types
    c_x_in = (ctypes.c_float * py_slam.MAX_ROWS)(*x)
    c_P_in = (ctypes.c_float * (py_slam.MAX_ROWS**2))(*P)
    c_u = (ctypes.c_float * 2)(*u)
    c_z = (ctypes.c_float * 3)(*z)
    c_Q = (ctypes.c_float * 2)(*Q)
    c_R = (ctypes.c_float * 2)(*R)
    
    # Buffers de sortie vides
    c_x_out = (ctypes.c_float * py_slam.MAX_ROWS)()
    c_P_out = (ctypes.c_float * (py_slam.MAX_ROWS**2))()
    
    # Variables pour récupérer les nouvelles tailles (pointeurs)
    c_x_rows_out = ctypes.c_int(0)
    c_P_rows_out = ctypes.c_int(0)

    # Appel
    cpp_lib.ekf_slam_bridge(
        c_x_in, x_rows,
        c_P_in, P_rows,
        c_u, c_z, c_Q, c_R,
        c_x_out, ctypes.byref(c_x_rows_out),
        c_P_out, ctypes.byref(c_P_rows_out)
    )
    
    # Conversion C array -> Python list
    return list(c_x_out), c_x_rows_out.value, list(c_P_out), c_P_rows_out.value

# --- 3. GÉNÉRATEUR DE SCÉNARIO ---

def generate_sensor_data(true_robot_state):
    """Simule un capteur qui détecte l'amer le plus proche"""
    rx, ry, ryaw = true_robot_state
    
    # Paramètres capteur
    MAX_RANGE = 15.0
    FOV = math.pi / 2.0 # Champ de vision +/- 90 deg
    
    visible_lm = None
    min_dist = MAX_RANGE
    
    for lm in TRUE_LANDMARKS:
        dx = lm[0] - rx
        dy = lm[1] - ry
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Calcul de l'angle relatif
        angle_global = math.atan2(dy, dx)
        angle_rel = angle_global - ryaw
        # Normalisation pi_2_pi
        while angle_rel > math.pi: angle_rel -= 2*math.pi
        while angle_rel < -math.pi: angle_rel += 2*math.pi
        
        if dist < min_dist and abs(angle_rel) < FOV:
            min_dist = dist
            # On ajoute un peu de bruit gaussien pour le réalisme
            z_dist = dist + random.gauss(0, 0.05)
            z_angle = angle_rel + random.gauss(0, 0.01)
            visible_lm = [z_dist, z_angle, 0.0]
            
            # SLAM simple: on ne traite qu'un seul amer par step ici
            break 
            
    if visible_lm:
        return visible_lm
    else:
        # Code pour "rien vu" : distance très grande
        return [100.0, 0.0, 0.0]

# --- 4. MAIN LOOP ---

def main():
    print("=== TEST COMPARATIF SLAM HLS vs PYTHON ===")
    print(f"Carte: {len(TRUE_LANDMARKS)} landmarks à découvrir.")
    
    # Init Buffers
    max_rows = py_slam.MAX_ROWS
    max_cols = py_slam.MAX_COLS 
    
    py_x = [0.0] * max_rows
    py_P = [0.0] * (max_rows * max_cols)
    
    # Init P (Identité 3x3 pour le robot)
    for i in range(3):
        py_P[i * max_cols + i] = 1.0
        
    current_x_rows = 3
    current_P_rows = 3
    
    # Vérité terrain robot
    true_robot = [0.0, 0.0, 0.0] # x, y, yaw
    
    # Paramètres
    u = [1.0, 0.1] # v=1 m/s, yaw_rate=0.1 rad/s
    Q = [0.1, 0.1]
    R = [0.04, 0.0076]
    
    # Simulation sur 30 pas
    for step in range(30):
        # 1. Mise à jour Vérité Terrain (Déplacement)
        true_robot[0] += u[0] * 0.1 * math.cos(true_robot[2])
        true_robot[1] += u[0] * 0.1 * math.sin(true_robot[2])
        true_robot[2] += u[1] * 0.1
        
        # 2. Mesure Capteur
        z = generate_sensor_data(true_robot)
        
        print(f"\n--- STEP {step} ---")
        if z[0] < 20.0:
            print(f"   [Capteur] Amer détecté à dist={z[0]:.2f}, angle={z[1]:.2f}")
        else:
            print("   [Capteur] Rien à l'horizon...")

        # 3. Exécution PYTHON
        res_x_py_obj, res_P_py_obj = py_slam.ekf_slam_top(
            list(py_x), current_x_rows, 
            list(py_P), current_P_rows, 
            u, z, Q, R
        )
        
        # 4. Exécution C++
        res_x_cpp, rows_cpp, res_P_cpp, _ = run_cpp_slam(
            py_x, current_x_rows, 
            py_P, current_P_rows, 
            u, z, Q, R
        )
        
        # 5. Comparaison et Mise à jour
        # On vérifie si un landmark a été ajouté
        n_lm_prev = (current_x_rows - 3) // 2
        n_lm_now  = (rows_cpp - 3) // 2
        
        if n_lm_now > n_lm_prev:
            print(f"   >>> NOUVEAU LANDMARK AJOUTÉ ! Total: {n_lm_now}")
            
        # Vérification Cohérence
        if res_x_py_obj.rows != rows_cpp:
            print(f"   [ERREUR] Divergence de taille ! Py={res_x_py_obj.rows} vs Cpp={rows_cpp}")
            break
            
        # Comparaison des valeurs (Robot X)
        diff_x = abs(res_x_py_obj.get(0,0) - res_x_cpp[0])
        if diff_x > 1e-4:
            print(f"   [ERREUR] Divergence calcul X! Diff={diff_x}")
        
        # Mise à jour des buffers pour le tour suivant (Feedback loop sur C++)
        # Cela force les deux algo à rester synchro sur les inputs
        py_x = res_x_cpp
        py_P = res_P_cpp
        current_x_rows = rows_cpp
        current_P_rows = rows_cpp # P est carré
        
        # Affichage État
        print(f"   Est. Robot: x={py_x[0]:.2f}, y={py_x[1]:.2f} (True: {true_robot[0]:.2f}, {true_robot[1]:.2f})")
        if n_lm_now > 0:
            lm_x = py_x[3]
            lm_y = py_x[4]
            print(f"   Est. LM#0 : x={lm_x:.2f}, y={lm_y:.2f} (True: {TRUE_LANDMARKS[0]})")

if __name__ == "__main__":
    main()