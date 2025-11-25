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

# --- Main script

def main(Landmarks,Q_sim_factor=3, Py_sim_factor=1, Q_factor=2, Py_factor=2, known_data_association=1, max_range=10.0, m_dist_th=9.0, speed=1.0, radius=0.1, bearing_only=False):
    # print(__file__ + " start!!")
    # Simulation parameter
    KNOWN_DATA_ASSOCIATION = known_data_association  # Whether we use the true landmarks id or not
    MAX_RANGE = max_range  # maximum observation range
    M_DIST_TH = m_dist_th  # Threshold of Mahalanobis distance for data association.

    # noise on control input
    Q_sim = (Q_sim_factor * np.diag([0.1, np.deg2rad(1)])) ** 2
    # noise on measurement
    Py_sim = (Py_sim_factor * np.diag([0.1, np.deg2rad(5)])) ** 2

    # Kalman filter Parameters
    # Estimated input noise for Kalman Filter
    Q = Q_factor * Q_sim
    # Estimated measurement noise for Kalman Filter
    Py = Py_factor * Py_sim

    time_sim = 0.0

    

    # Init state vector [x y yaw]' and covariance for Kalman
    xEst = np.zeros((STATE_SIZE, 1))
    PEst = initPEst

    # Init true state for simulator
    xTrue = np.zeros((STATE_SIZE, 1))

    # Init dead reckoning (sum of individual controls)
    xDR = np.zeros((STATE_SIZE, 1))

    # Init landmark hypotheses for bearing-only SLAM
    landmark_hypotheses = {}

    # Init history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hxError = np.abs(xEst-xTrue)  # pose error
    hxVar = np.sqrt(np.diag(PEst[0:STATE_SIZE,0:STATE_SIZE]).reshape(3,1))  #state std dev


    count = 0
    time_exec = 0.0
    while  time_sim <= SIM_TIME:
        
        count = count + 1
        time_sim += DT

        # Simulate motion and generate u and y
        uTrue = calc_input(speed, radius)

        if bearing_only:
            xTrue, y, xDR, u = observation_bearing_only(xTrue, xDR, uTrue, Landmarks, Q_sim, Py_sim, MAX_RANGE)
            start_time = time.time()
            xEst, PEst, landmark_hypotheses = ekf_slam_bearing_only(xEst, PEst, u, y, landmark_hypotheses, Q, Py)
        else:
            xTrue, y, xDR, u = observation(xTrue, xDR, uTrue, Landmarks, Q_sim, Py_sim, MAX_RANGE)
            start_time = time.time()
            xEst, PEst = ekf_slam(xEst, PEst, u, y, M_DIST_TH, KNOWN_DATA_ASSOCIATION, Q, Py)
        time_exec = time_exec + time.time()-start_time
        # store data history
        hxEst = np.hstack((hxEst, xEst[0:STATE_SIZE]))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        err = xEst[0:STATE_SIZE]-xTrue
        err[2] = pi_2_pi(err[2])
        hxError = np.hstack((hxError,err))
        hxVar = np.hstack((hxVar,np.sqrt(np.diag(PEst[0:STATE_SIZE,0:STATE_SIZE]).reshape(3,1))))
    print("Mean execution time: {:.6f} [us]".format(time_exec/count*1e6))

# Bearing-only SLAM parameters
HYPOTHESIS_NUM = 5  # Number of hypotheses for undelayed initialization
HYPOTHESIS_DISTANCES = np.linspace(2.0, 15.0, HYPOTHESIS_NUM)  # Distance range for hypotheses
HYPOTHESIS_COV_FACTOR = np.array([0.1, 0.5, 1.0, 2.0, 5.0])  # Covariance factors for each hypothesis
PRUNING_THRESHOLD = 0.01  # Probability threshold for pruning hypotheses

def observation_bearing_only(xTrue, xd, uTrue, Landmarks, Q_sim, Py_sim, MAX_RANGE):

    xTrue = motion_model(xTrue, uTrue)
    
    # Bearing-only observations: [angle_to_landmark, landmark_id]
    y = np.zeros((0, 2))
    
    for i in range(len(Landmarks[:, 0])):
        dx = Landmarks[i, 0] - xTrue[0, 0]
        dy = Landmarks[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        
        if d <= MAX_RANGE:
            # Add noise to bearing only
            angle_n = angle + np.random.randn() * Py_sim[1, 1] ** 0.5
            yi = np.array([angle_n, i])
            y = np.vstack((y, yi))
    
    # Add noise to input
    u = np.array([[
        uTrue[0, 0] + np.random.randn() * Q_sim[0, 0] ** 0.5,
        uTrue[1, 0] + np.random.randn() * Q_sim[1, 1] ** 0.5]]).T
    
    xd = motion_model(xd, u)
    
    return xTrue, y, xd, u

def initialize_landmark_hypotheses(xEst, bearing_measurement):

    hypotheses = []
    robot_x, robot_y, robot_yaw = xEst[0, 0], xEst[1, 0], xEst[2, 0]
    bearing = bearing_measurement
    
    for i, distance in enumerate(HYPOTHESIS_DISTANCES):
        # Calculate landmark position for this hypothesis
        lm_x = robot_x + distance * math.cos(robot_yaw + bearing)
        lm_y = robot_y + distance * math.sin(robot_yaw + bearing)
        
        # Covariance increases with distance uncertainty
        cov_factor = HYPOTHESIS_COV_FACTOR[i]
        
        hypothesis = {
            'position': np.array([[lm_x], [lm_y]]),
            'cov_factor': cov_factor,
            'weight': 1.0 / HYPOTHESIS_NUM,  # Equal initial weight
            'likelihood': 1.0,
            'measurement_count': 0
        }
        hypotheses.append(hypothesis)
    
    return hypotheses

def update_hypothesis_weights(hypotheses, xEst, bearing_measurement, Py_bearing):

    likelihoods = []
    total_likelihood = 0.0
    robot_x, robot_y, robot_yaw = xEst[0, 0], xEst[1, 0], xEst[2, 0]
    
    for hyp in hypotheses:
        lm_x, lm_y = hyp['position'][0, 0], hyp['position'][1, 0]
        dx = lm_x - robot_x
        dy = lm_y - robot_y
        
        predicted_bearing = pi_2_pi(math.atan2(dy, dx) - robot_yaw)
        
        innovation = pi_2_pi(bearing_measurement - predicted_bearing)
        
        # Add distance-dependent measurement uncertainty
        distance = math.hypot(dx, dy)
        measurement_std = math.sqrt(Py_bearing[1, 1]) * (1.0 + 0.1 * distance)
        likelihood = math.exp(-0.5 * (innovation ** 2) / (measurement_std ** 2))
        
        likelihoods.append(likelihood)
        total_likelihood += likelihood
    
    # Normalize weights
    if total_likelihood > 0:
        for i, hyp in enumerate(hypotheses):
            hyp['weight'] = likelihoods[i] / total_likelihood
            hyp['likelihood'] = likelihoods[i]
            hyp['measurement_count'] += 1
    
    return hypotheses

def prune_hypotheses(hypotheses, threshold=PRUNING_THRESHOLD):

    pruned = [h for h in hypotheses if h['weight'] > threshold]
    
    if len(pruned) == 0:  # Keep best hypothesis
        best_idx = np.argmax([h['weight'] for h in hypotheses])
        pruned = [hypotheses[best_idx]]
    
    return pruned

def jacob_h_bearing_only(delta, x, i):

    q = (delta.T @ delta)[0, 0]
    
    nLM = calc_n_lm(x)
    
    # Jacobian is 1 x (3 + 2*nLM)
    H = np.zeros((1, 3 + 2 * nLM))
    
    # Robot pose part: [dh/dx, dh/dy, dh/dyaw]
    H[0, 0] = delta[1, 0] / q  # dh/dx_robot
    H[0, 1] = -delta[0, 0] / q  # dh/dy_robot
    H[0, 2] = -1.0  # dh/dyaw_robot
    
    # Landmark position part: only landmark i is relevant
    H[0, 3 + 2*i] = -delta[1, 0] / q  # dh/dlm_x
    H[0, 3 + 2*i + 1] = delta[0, 0] / q  # dh/dlm_y
    
    return H

def calc_innovation_bearing_only(xEst, PEst, y, LMid, Py):

    # Predicted bearing from state
    lm = get_landmark_position_from_state(xEst, LMid)
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    
    # Avoid division by zero
    if q < 1e-10:
        q = 1e-10
    
    y_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    yp = pi_2_pi(y_angle)
    
    innov = np.array([[pi_2_pi(y - yp)]])
    
    H = jacob_h_bearing_only(delta, xEst, LMid)
    
    S = H @ PEst @ H.T + Py[1:2, 1:2]
    
    return innov, S, H

def ekf_slam_bearing_only(xEst, PEst, u, y, landmark_hypotheses, Q, Py):

    S = STATE_SIZE
    
    A, B = jacob_motion(xEst[0:S], u)
    xEst[0:S] = motion_model(xEst[0:S], u)
    PEst[0:S, 0:S] = A @ PEst[0:S, 0:S] @ A.T + B @ Q @ B.T
    PEst[0:S, S:] = A @ PEst[0:S, S:]
    PEst[S:, 0:S] = PEst[0:S, S:].T
    PEst = (PEst + PEst.T) / 2.0
    
    for iy in range(len(y[:, 0])):
        bearing = y[iy, 0]
        lm_id = int(y[iy, 1])
        
        nLM = calc_n_lm(xEst)
        
        if lm_id not in landmark_hypotheses:
            landmark_hypotheses[lm_id] = initialize_landmark_hypotheses(xEst, bearing)
        
        else:
            landmark_hypotheses[lm_id] = update_hypothesis_weights(
                landmark_hypotheses[lm_id], xEst, bearing, Py)
            
            landmark_hypotheses[lm_id] = prune_hypotheses(landmark_hypotheses[lm_id])
            
            # Use best hypothesis
            best_hyp = max(landmark_hypotheses[lm_id], key=lambda h: h['weight'])
            
            # Add to state only after measurement count threshold
            if best_hyp['measurement_count'] >= 3 and lm_id >= nLM:
                xEst = np.vstack((xEst, best_hyp['position']))
                
                P_lm_cov = (best_hyp['cov_factor'] ** 2) * np.eye(2) * 5.0
                PEst = np.vstack((np.hstack((PEst, np.zeros((len(xEst)-2, 2)))),
                                  np.hstack((np.zeros((2, len(xEst)-2)), P_lm_cov))))
            
            if lm_id < calc_n_lm(xEst):
                try:
                    innov, S, H = calc_innovation_bearing_only(xEst, PEst, bearing, lm_id, Py)
                    S_inv = np.linalg.inv(S)
                    K = (PEst @ H.T) @ S_inv
                    xEst = xEst + (K @ innov).reshape(xEst.shape)
                    PEst = (np.eye(len(xEst)) - K @ H) @ PEst
                    PEst = 0.5 * (PEst + PEst.T)
                except (np.linalg.LinAlgError, ValueError):
                    pass  # Skip update if singular or dimension mismatch
    
    xEst[2] = pi_2_pi(xEst[2])
    
    return xEst, PEst, landmark_hypotheses

trueLandmarkId = []
landmark_hypotheses = {}
# Define landmark positions [x, y]
Landmarks_Default = np.array([[0.0, 5.0],
                             [11.0, 1.0],
                             [3.0, 15.0],
                             [-5.0, 20.0]])

main(Landmarks=Landmarks_Default, 
     bearing_only=False,
     Q_sim_factor=2,
     Py_sim_factor=0.5)