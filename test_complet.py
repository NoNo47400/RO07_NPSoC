import ctypes
import os
import random
import math
import sys

# Import de votre version Python
try:
    import src.ekf_slam as py_slam
except ImportError:
    # Fallback si exécuté depuis le dossier src
    import ekf_slam as py_slam

# --- CONFIGURATION CTYPES ---
lib_name = "./libekf.so" if os.name != 'nt' else "./libekf.dll"
if not os.path.exists(lib_name):
    print(f"ERREUR CRITIQUE: {lib_name} introuvable.")
    print("Compilez avec: g++ -shared -fPIC -o libekf.so src/ekf_slam.cpp testbench/wrapper.cpp -I src")
    sys.exit(1)

cpp_lib = ctypes.CDLL(lib_name)

# Types C++
c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)

cpp_lib.ekf_slam_bridge.argtypes = [
    c_float_p, ctypes.c_int, c_float_p, ctypes.c_int, 
    c_float_p, c_float_p, c_float_p, c_float_p,
    c_float_p, c_int_p, c_float_p, c_int_p
]

def run_cpp_slam(x, x_rows, P, P_rows, u, z, Q, R):
    c_x_in = (ctypes.c_float * py_slam.MAX_ROWS)(*x)
    c_P_in = (ctypes.c_float * (py_slam.MAX_ROWS**2))(*P)
    c_u = (ctypes.c_float * 2)(*u)
    c_z = (ctypes.c_float * 3)(*z)
    c_Q = (ctypes.c_float * 2)(*Q)
    c_R = (ctypes.c_float * 2)(*R)
    
    c_x_out = (ctypes.c_float * py_slam.MAX_ROWS)()
    c_P_out = (ctypes.c_float * (py_slam.MAX_ROWS**2))()
    
    # Init outputs avec les valeurs courantes
    c_x_rows_out = ctypes.c_int(x_rows) 
    c_P_rows_out = ctypes.c_int(P_rows)

    cpp_lib.ekf_slam_bridge(
        c_x_in, x_rows, c_P_in, P_rows, c_u, c_z, c_Q, c_R,
        c_x_out, ctypes.byref(c_x_rows_out), c_P_out, ctypes.byref(c_P_rows_out)
    )
    
    return list(c_x_out), c_x_rows_out.value, list(c_P_out), c_P_rows_out.value


def compare_vectors(name, cpp_vec, py_vec, length, tol=1e-5):
    # length: number of elements to compare
    diffs = [abs(cpp_vec[i] - py_vec[i]) for i in range(length)]
    max_diff = max(diffs) if diffs else 0.0
    mse = sum(d * d for d in diffs) / length if length > 0 else 0.0
    rmse = math.sqrt(mse)
    ok = max_diff <= tol
    print(f"Comparaison {name}: rows={length} | max_diff={max_diff:.6g} | rmse={rmse:.6g} | tol={tol} | {'OK' if ok else 'DIFF'}")
    if not ok:
        # Afficher les indices problématiques
        for i, d in enumerate(diffs):
            if d > tol:
                print(f"  - idx {i}: C++={cpp_vec[i]:.6g} Py={py_vec[i]:.6g} diff={d:.6g}")


def compare_matrices(name, cpp_mat, py_mat, rows, cols, tol=1e-4):
    # matrices are stored row-major in flat lists
    length = rows * cols
    diffs = [abs(cpp_mat[i] - py_mat[i]) for i in range(length)]
    max_diff = max(diffs) if diffs else 0.0
    mse = sum(d * d for d in diffs) / length if length > 0 else 0.0
    rmse = math.sqrt(mse)
    ok = max_diff <= tol
    print(f"Comparaison {name}: {rows}x{cols} | max_diff={max_diff:.6g} | rmse={rmse:.6g} | tol={tol} | {'OK' if ok else 'DIFF'}")
    if not ok:
        # montrer quelques coefficients divergents (max 10)
        bad = [(i, cpp_mat[i], py_mat[i], diffs[i]) for i in range(length) if diffs[i] > tol]
        for idx, cval, pval, d in bad[:10]:
            r = idx // cols; c = idx % cols
            print(f"  - [{r},{c}] C++={cval:.6g} Py={pval:.6g} diff={d:.6g}")

def main():
    print("=== DEBUG SLAM ===")
    
    # Init Buffers
    max_rows = py_slam.MAX_ROWS
    max_cols = py_slam.MAX_COLS 
    
    py_x = [0.0] * max_rows
    py_P = [0.0] * (max_rows * max_cols)
    # Init P identité 3x3
    py_P[0] = 1.0; py_P[max_cols+1] = 1.0; py_P[2*max_cols+2] = 1.0
    
    current_x_rows = 3
    current_P_rows = 3
    
    # Paramètres
    u = [1.0, 0.1] # Avance et tourne
    Q = [0.1, 0.1]
    R = [0.04, 0.0076]
    
    # Simulation FORCEE : On voit un landmark à 5m droit devant
    # C'est un test unitaire, on ne simule pas la physique complexe ici
    # Z = [Range, Angle, ID]
    z_measurements = [
        [100.0, 0.0, 0.0], # Step 0: Rien vu (100m)
        [100.0, 0.0, 0.0], # Step 1: Rien vu
        [5.0, 0.1, 0.0],   # Step 2: LANDMARK VU à 5m !
        [4.9, 0.1, 0.0],   # Step 3: Toujours vu (un peu plus près)
        [100.0, 0.0, 0.0]  # Step 4: Perdu de vue
    ]
    
    for k, z in enumerate(z_measurements):
        print(f"\n--- STEP {k} (Mesure: {z[0]}m) ---")
        
        # 1. Exécution C++ (Référence)
        res_x, res_rows, res_P, res_P_rows = run_cpp_slam(
            py_x, current_x_rows, py_P, current_P_rows, u, z, Q, R
        )
        
        print(f"C++ Output Rows: {res_rows}")
        
        if res_rows > current_x_rows:
            print(">>> C++ A AJOUTÉ UN LANDMARK !")
            
        # 2. Exécution Python
        py_res_x_obj, py_res_P = py_slam.ekf_slam_top(
            list(py_x), current_x_rows, list(py_P), current_P_rows, u, z, Q, R
        )
        print(f"Py  Output Rows: {py_res_x_obj.rows} | Py P rows: {py_res_P.rows}")

        # Comparaison numérique entre C++ et Python
        py_x_out = py_res_x_obj.data
        py_P_out = py_res_P.data

        # Vérification dimensionnelle
        if res_rows != py_res_x_obj.rows:
            print(f"ATTENTION: nombre de lignes différent (C++={res_rows} vs Py={py_res_x_obj.rows})")

        comp_rows = min(res_rows, py_res_x_obj.rows)
        # Comparer l'état (seulement les premières comp_rows entrées)
        compare_vectors("Etat", res_x, py_x_out, comp_rows, tol=1e-5)

        # Comparer la covariance (comparer la sous-matrice comp_P_rows x comp_P_rows)
        comp_P_rows = min(res_P_rows, py_res_P.rows)
        compare_matrices("Covariance", res_P, py_P_out, comp_P_rows, comp_P_rows, tol=1e-4)

        # Mise à jour pour le prochain tour (on garde la référence C++ comme source de vérité)
        current_x_rows = res_rows
        current_P_rows = res_P_rows
        py_x = res_x # On copie les données brutes du C++
        py_P = res_P
        
        # Affichage position
        if current_x_rows > 3:
            lm_x = py_x[3]
            lm_y = py_x[4]
            print(f"Etat: Robot=[{py_x[0]:.2f}, {py_x[1]:.2f}] | LM=[{lm_x:.2f}, {lm_y:.2f}]")
        else:
            print(f"Etat: Robot=[{py_x[0]:.2f}, {py_x[1]:.2f}] | Pas de LM")

if __name__ == "__main__":
    main()