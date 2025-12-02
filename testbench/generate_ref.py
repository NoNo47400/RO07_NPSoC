"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)

Modified : Goran Frehse, David Filliat
"""

import math
import numpy as np
import time

DT = 0.1  # time tick [s]
SIM_TIME = 100.0  # simulation time [s]
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

# Initial estimate of pose covariance
initPEst = 0.01 * np.eye(STATE_SIZE)
initPEst[2,2] = 0.0001  # low orientation error

# --- Helper functions

def calc_n_lm(x):
    """
    Computes the number of landmarks in state vector
    """

    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def calc_landmark_position(x, y):
    """
    Computes absolute landmark position from robot pose and observation
    """

    y_abs = np.zeros((2, 1))

    y_abs[0, 0] = x[0, 0] + y[0] * math.cos(x[2, 0] + y[1])
    y_abs[1, 0] = x[1, 0] + y[0] * math.sin(x[2, 0] + y[1])

    return y_abs


def get_landmark_position_from_state(x, ind):
    """
    Extract landmark position from state vector
    """

    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]

    return lm


def pi_2_pi(angle):
    """
    Put an angle between -pi / pi
    """

    return (angle + math.pi) % (2 * math.pi) - math.pi


# --- Motion model related functions

def calc_input(speed, radius):
    """
    Generate a control vector to make the robot follow a circular trajectory
    """

    v = speed  # [m/s]
    yaw_rate = radius  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u


def motion_model(x, u):
    """
    Compute future robot position from current position and control
    """
    
    xp = np.array([[x[0,0] + u[0,0]*DT * math.cos(x[2,0])],
                  [x[1,0] + u[0,0]*DT * math.sin(x[2,0])],
                  [x[2,0] + u[1,0]*DT]])
    xp[2] = pi_2_pi(xp[2])

    return xp.reshape((3, 1))


def jacob_motion(x, u):
    """
    Compute the jacobians of motion model wrt x and u
    """

    # Jacobian of f(X,u) wrt X
    A = np.array([[1.0, 0.0, float(-DT * u[0,0] * math.sin(x[2, 0]))],
                  [0.0, 1.0, float(DT * u[0,0] * math.cos(x[2, 0]))],
                  [0.0, 0.0, 1.0]])

    # Jacobian of f(X,u) wrt u
    B = np.array([[float(DT * math.cos(x[2, 0])), 0.0],
                  [float(DT * math.sin(x[2, 0])), 0.0],
                  [0.0, DT]])

    return A, B


# --- Observation model related functions

def observation(xTrue, xd, uTrue, Landmarks, Q_sim, Py_sim, MAX_RANGE):
    """
    Generate noisy control and observation and update true position and dead reckoning
    """
    xTrue = motion_model(xTrue, uTrue)

    # add noise to gps x-y
    y = np.zeros((0, 3))

    for i in range(len(Landmarks[:, 0])):

        dx = Landmarks[i, 0] - xTrue[0, 0]
        dy = Landmarks[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Py_sim[0, 0] ** 0.5  # add noise
            dn = max(dn,0)
            angle_n = angle + np.random.randn() * Py_sim[1, 1] ** 0.5  # add noise
            yi = np.array([dn, angle_n, i])
            y = np.vstack((y, yi))

    # add noise to input
    u = np.array([[
        uTrue[0, 0] + np.random.randn() * Q_sim[0, 0] ** 0.5,
        uTrue[1, 0] + np.random.randn() * Q_sim[1, 1] ** 0.5]]).T

    xd = motion_model(xd, u)
    
    return xTrue, y, xd, u


def search_correspond_landmark_id(xEst, PEst, yi, M_DIST_TH, Py):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = calc_n_lm(xEst)

    min_dist = []

    for i in range(nLM):
        innov, S, H = calc_innovation(xEst, PEst, yi, i, Py)
        min_dist.append(innov.T @ np.linalg.inv(S) @ innov)
        

    min_dist.append(M_DIST_TH)  # new landmark

    min_id = min_dist.index(min(min_dist))

    return min_id


def jacob_h(q, delta, x, i):
    """
    Compute the jacobian of observation model
    """

    sq = math.sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], -delta[0, 0], -q,  -delta[1, 0], delta[0, 0]]])

    G = G / q
    nLM = calc_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * i)),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * (i + 1)))))

    F = np.vstack((F1, F2))

    H = G @ F

    return H


def jacob_augment(x, y):
    """
    Compute the jacobians for extending covariance matrix
    """
    
    Jr = np.array([[1.0, 0.0, -y[0] * math.sin(x[2,0] + y[1])],
                   [0.0, 1.0, y[0] * math.cos(x[2,0] + y[1])]])

    Jy = np.array([[math.cos(x[2,0] + y[1]), -y[0] * math.sin(x[2,0] + y[1])],
                   [math.sin(x[2,0] + y[1]), y[0] * math.cos(x[2,0] + y[1])]])

    return Jr, Jy

# --- Kalman filter related functions

def calc_innovation(xEst, PEst, y, LMid, Py):
    """
    Compute innovation and Kalman gain elements
    """

    # Compute predicted observation from state
    lm = get_landmark_position_from_state(xEst, LMid)
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    y_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    yp = np.array([[math.sqrt(q), pi_2_pi(y_angle)]])

    # compute innovation, i.e. diff with real observation
    innov = (y - yp).T
    innov[1] = pi_2_pi(innov[1])

    # compute matrixes for Kalman Gain
    H = jacob_h(q, delta, xEst, LMid)
    S = H @ PEst @ H.T + Py
    
    return innov, S, H
    

def ekf_slam(xEst, PEst, u, y, M_DIST_TH, KNOWN_DATA_ASSOCIATION, Q, Py):
    """
    Apply one step of EKF predict/correct cycle
    """
    
    S = STATE_SIZE
    
    # Predict
    A, B = jacob_motion(xEst[0:S], u)

    xEst[0:S] = motion_model(xEst[0:S], u)

    PEst[0:S, 0:S] = A @ PEst[0:S, 0:S] @ A.T + B @ Q @ B.T
    PEst[0:S,S:] = A @ PEst[0:S,S:]
    PEst[S:,0:S] = PEst[0:S,S:].T

    PEst = (PEst + PEst.T) / 2.0  # ensure symetry
    
    # Update
    for iy in range(len(y[:, 0])):  # for each observation
        nLM = calc_n_lm(xEst)
        
        if KNOWN_DATA_ASSOCIATION:
            try:
                min_id = trueLandmarkId.index(y[iy, 2])
            except ValueError:
                min_id = nLM
                trueLandmarkId.append(y[iy, 2])
        else:
            min_id = search_correspond_landmark_id(xEst, PEst, y[iy, 0:2], M_DIST_TH, Py)


        # Extend map if required
        if min_id == nLM:
            #print("New LM")
            
            # Extend state and covariance matrix
            xEst = np.vstack((xEst, calc_landmark_position(xEst, y[iy, :])))

            Jr, Jy = jacob_augment(xEst[0:3], y[iy, :])
            bottomPart = np.hstack((Jr @ PEst[0:3, 0:3], Jr @ PEst[0:3, 3:]))
            rightPart = bottomPart.T
            PEst = np.vstack((np.hstack((PEst, rightPart)),
                              np.hstack((bottomPart,
                              Jr @ PEst[0:3, 0:3] @ Jr.T + Jy @ Py @ Jy.T))))

        else:
            # Perform Kalman update
            innov, S, H = calc_innovation(xEst, PEst, y[iy, 0:2], min_id, Py)
            K = (PEst @ H.T) @ np.linalg.inv(S)
            
            xEst = xEst + (K @ innov)
                        
            PEst = (np.eye(len(xEst)) - K @ H) @ PEst
            PEst = 0.5 * (PEst + PEst.T)  # Ensure symetry
        
    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst


def get_c_array(name: str, arr: np.array):
    """
    Get C-style array
    """

    arr = np.asarray(arr)

    def format_recursive(sub):
        # Base case: scalar
        if sub.ndim == 0:
            return str(sub.item())

        # Recursive case: format each element
        inner = ", ".join(format_recursive(x) for x in sub)
        return "{ " + inner + " }"

    return f"float {name}" + "".join(f"[{d}]" for d in arr.shape) + " = " + format_recursive(arr)

# --- Main script

def main(Landmarks,Q_sim_factor=3, Py_sim_factor=1, Q_factor=2, Py_factor=2, known_data_association=1, max_range=10.0, m_dist_th=9.0, speed=1.0, radius=0.1):
    # print(__file__ + " start!!")
    # Simulation parameter
    KNOWN_DATA_ASSOCIATION = known_data_association  # Whether we use the true landmarks id or not
    MAX_RANGE = max_range  # maximum observation range
    M_DIST_TH = m_dist_th  # Threshold of Mahalanobis distance for data association.

    # noise on control input
    Q_sim = np.diag([0.1, 0.1])
    # noise on measurement
    Py_sim = np.diag([0.2*0.2, (5.0*math.pi/180.0)*(5.0*math.pi/180.0)])

    # Kalman filter Parameters
    # Estimated input noise for Kalman Filter
    Q = Q_factor * Q_sim
    # Estimated measurement noise for Kalman Filter
    Py = Py_factor * Py_sim
    

    # Init state vector [x y yaw]' and covariance for Kalman
    #xEst = np.zeros((STATE_SIZE, 1))
    xEst = np.array([[-5.62815962],
            [18.37374747],
            [-2.54979418],
            [-0.09302817],
            [ 4.97980269],
            [ 6.11371654],
            [-6.91804677],
            [10.97515561],
            [ 1.16371694],
            [14.82922   ],
            [10.21498871],
            [ 2.74484343],
            [15.0169684 ],
            [-5.28442142],
            [19.95406455]])
    PEst = np.eye(15)

    # Init true state for simulator
    xTrue = np.zeros((STATE_SIZE, 1))

    # Init dead reckoning (sum of individual controls)
    xDR = np.zeros((STATE_SIZE, 1))


    NB_STEPS = 10
    file = open("golden_ref.h", "w")

    # Simulate motion and generate u and y
    for _ in range(NB_STEPS):
        file.write(get_c_array("xEst_IN", xEst))


        uTrue = calc_input(speed, radius)

        xTrue, y, xDR, u = observation(xTrue, xDR, uTrue, Landmarks, Q_sim, Py_sim, MAX_RANGE)
        u = np.array([[
            1.0,
            0.1]]).T

        y = np.array([[10.0,0.0,0.0]])

        xEst, PEst = ekf_slam(xEst, PEst, u, y, M_DIST_TH, KNOWN_DATA_ASSOCIATION, Q, Py)
        # file.write("xEst_OUT: " + str(xEst))
        # file.write("PEst_OUT: " + str(PEst))
        file.write(get_c_array("xEst_OUT", xEst))
        file.write(get_c_array("PEst_OUT", PEst))
        # print("xEst: ", xEst)
        # print("PEst: ", PEst)

    file.close()

trueLandmarkId = []
landmark_hypotheses = {}
# Define landmark positions [x, y]
Landmarks_Default = np.array([[0.0, 5.0],
                             [11.0, 1.0],
                             [3.0, 15.0],
                             [-5.0, 20.0]])

main(Landmarks=Landmarks_Default, 
     Q_sim_factor=2,
     Py_sim_factor=0.5)