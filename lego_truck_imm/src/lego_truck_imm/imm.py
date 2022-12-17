"""
This module implements the IMM-filter. The purpose of the IMM-filter is to
predict the future trajectory of a dynamic obstacle (pedestrian/ground vehicle)
by utilizing the mixed estimate from different Kalman Filters with different
motion models. Input is given as measurements of the dynamic obstacle (x,y,heading)
and the output is a trajectory (list) of coordinates and corresponding covariance
matrices. The trajectory should correspond to the most likely one, given the
last measurement.

Tuning parameters:
mu - the inital weights (proabilities) of each filter
transition_probability_matrix - probability to go from state i to state j
self.q - Process noise
self.Rk - Measurement noise

Classes:
IMM
ModesPed
ModesVeh

INPUT:
A measurement from either Qualisys or the Obstacle Simulator.
The measurement should include the x- and y-position, the heading angle
theta as well as the type, which is 0 if it is a pedestrian and 1 if it
is a ground vehicle.

Update row 284 and 285 to get correct meas from either obstacle simulator or
from the tests files for pedestrian.

Do the corresponding change on line 404 and 405 for the ground vehicle.

OUTPUT:
Will publish on the /predictor/predicted_trajectory_imm topic.
The message has the following attributes:
x = list of the next n x-positions according to the corresponding TU
y = list of the next n y-positions according to the corresponding TU
pxx = list of the next n xx-values in the covariance matrix
pxy = list of the next n xy-values in the covariance matrix
pyx = list of the next n yx-values in the covariance matrix
pyy = list of the next n yy-values in the covariance matrix

For examples on how to test and use the filters, see the tests folder or the
User manual where it is explained how to start the obstacle simulator with the
IMM-filter in RViz.

"""
# Pylint Warnings

#pylint: disable=invalid-name
# Naming convention is appropriate given how the math describes the IMM-Filter
# and the Kalman Filters.

#pylint: disable=too-many-instance-attributes
#Need a lot of instances in the constructor to make the filter work properly.

#pylint: disable=no-self-use
#These functions does not use self, but should still be seen as members of the
#class.

import sys
from enum import Enum
from math import exp
import queue
import rospy
import numpy as np
from scipy.stats import multivariate_normal

from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from lego_truck_messages.msg import DynamicObstacleIMM, PredictionIMM

class ModesPed(Enum):
    """
    Enum class for the different modes used by pedestrians.
    """
    IDLE = 0
    CV = 1
    CTPV = 2


class ModesVeh(Enum):
    """
    Enum class for the different modes used by ground vehicles.
    """
    IDLE = 0
    CTPV = 1
    CTPVA = 2


class IMM():
    """
    Main class for the IMM-filter
    """

# --------------- Init ---------------

    def __init__(self, mu, transition_probability_matrix):
        """
        Constructor for the IMM-filter. It initializes the different class variables
        and the publishers and subscribes.

        mu = np.array([probforfilter1, probforfilter2, ...])

        transition_probability_matrix = np.matrix of transitions probabilies, for
        example trans_prob = np.array([[0.97, 0.03], [0.03, 0.97]]) if only
        2 modes exist.

        Both of these are tuning parameters
        """
        #Init the ROS node
        rospy.init_node("lego_truck_imm")
        self.name = "IMM-filter: "

        ## Time variables
        #Queue size for Publishers, how many elements to store in queue.
        self.rate = 5
        #Sampling time, should be set accordingly to the dynamics of the
        #current problem.
        self.Ts = 0.2
        self.prev_t = 0
        self.curr_t = 0

        #Variables to help init the filter
        self.meas_0 = []
        self.meas_0_received = False
        self.is_obs_initialized = False

        ## Number of states that we predict into the future.
        self.n = 30

        ## IMM variables
        self.N = 3 #Number of modes/filers used for pedestrian/vehicle
        self.mu = mu #List of probabilies for each filter
        self.trans_prob_matrix = transition_probability_matrix
        self.type = 0 # 0 = pedestrian and 1 = ground vehicle
        self.mu_matrix = np.zeros((3,3)) # Temp variable
        self.pred_vec_xk = []   #n xks ahead in time
        self.pred_vec_xk2 = []
        self.pred_vec_Pk = []   #n Pks ahead in time
        self.output = PredictionIMM() #Format for output on Topic

        ##State variables
        self.xk = np.zeros(5)
        self.Pk = np.zeros((5,5))

        ##Lists of state vectors, covariances and likelihoods for each filter
        self.xks = []
        self.Pks = []
        self.likelihood = np.array([sys.float_info.min, sys.float_info.min, sys.float_info.min])

        ##Measurement noise matrix
        self.Rk = np.array([ # This correlates to the specified measurement noise
            [0.15, 0, 0],
            [0, 0.15, 0],
            [0, 0, 0.01],
        ])
        ## Initial Q, set to something in the magnitude of Rk
        self.q = np.array([
        [0.1, 0],
        [0, 0.1]
        ])

        ##Misc
        self.meas_q = queue.Queue() # Queue that the subscriber puts messages in

        # --- Ros subsciber ---
        # Subscribes to the dynamic obstacle simulator
        self._dynamic_obstacle_sub = rospy.Subscriber(
            "/obstacle_simulator/dynamic_obstacle_imm",
            DynamicObstacleIMM,
            self._input_callback,
        )

        # --- Ros publishers ---
        # Publishes predicted trajectory data
        self._predictor_pub = rospy.Publisher(
            "/predictor/predicted_trajectory_imm",
            PredictionIMM,
            queue_size = self.rate,
        )

        # Publishes visualization data for rviz about the predicted path
        self.predicted_path_pub = rospy.Publisher(
            "/predictor/predicted_path_imm",
            Path,
            queue_size=self.rate,
        )

        # Publishes visualisation data for rviz about the tracking of
        # the dynamic obstacle
        self._visualization_dynamic_obstacle_pub = rospy.Publisher(
            "/predictor/tracking_imm",
            Marker, # x,y,theta
            queue_size = self.rate,
        )

# --------------------- Callback functions -----------------

    def _input_callback(self, meas):
        """
        Callback function that pushes the measurements received to queue
        """
        self.meas_q.put(meas)


## ------------ Kalman Filter TU and MU for pedestrian --------------

    def _time_update_pedestrian(self, xk, Pk, mode):
        """
        General Time Update function for pedestrian. Uses mode
        to determine which motion model to use.
        """

        #Heading
        h = xk[3]

        #G - matrix to set the process noise properly
        G = np.array([
            [((self.Ts**2)/2)*np.cos(h), 0],
            [((self.Ts**2)/2)*np.sin(h), 0],
            [self.Ts, 0],
            [0, (self.Ts**2)/2],
            [0, self.Ts]
        ])

        #Determine correct Fk depeing on mode
        if mode == ModesPed.IDLE:
            Fk = np.array([
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0]
            ])
        elif mode == ModesPed.CV:
            Fk = np.array([
                [1, 0, (self.Ts*np.cos(h)), 0, 0],
                [0, 1, (self.Ts*np.sin(h)), 0, 0 ],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0]
            ])
        elif mode == ModesPed.CTPV:
            w = xk[4]

            xcorr = (2/w) * np.sin(((w*self.Ts)/2)) * \
                   np.cos(h + ((w*self.Ts)/2))

            ycorr = (2/w) * np.sin(((w*self.Ts)/2)) * \
                   np.sin(h + ((w*self.Ts)/2))

            Fk = np.array([
                [1, 0, xcorr, 0, 0],
                [0, 1, ycorr, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, self.Ts],
                [0, 0, 0, 0, 1]
            ])
        else:
            print(f"Got in default for Pedestrian TU with mode {mode}")
            Fk = np.array([
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0]
            ])

        #Update xk and Pk
        xk = Fk@xk
        Qk = G@self.q@np.transpose(G)
        Pk = Fk@Pk@np.transpose(Fk) + Qk

        return xk, Pk

    def _measurement_update_pedestrian(self, xk, Pk, meas, mode):
        """
        Measurement update for the pedestrian model
        using a Kalman Filter.
        """

        #Measurement x, y, theta
        #yk = meas #Use if we simulate with visualize_imm.py
        yk = np.array([meas.x, meas.y, meas.theta]) #Use we are using ROS

        # Observation matrix H
        Hk = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0]
        ])

        #Mean for measurment
        yk_hat = Hk@xk

        #Innovation covariance
        Sk = Hk@Pk@np.transpose(Hk) + self.Rk

        #Kalman gain
        Kk = Pk@np.transpose(Hk)@np.linalg.inv(Sk)

        # Measurement update
        xk = xk + Kk@(yk - yk_hat)
        Pk = Pk - Kk@Sk@np.transpose(Kk)

        # Compute the likelihood given the current measurement
        likelihood = self._calc_likelihood(yk, yk_hat, Sk)

        #Assign each likelihood with mode
        self.likelihood[mode.value] = likelihood

        return xk, Pk

## ---------- Kalman Filter TU and MU for ground vehicle -------------

    def _time_update_vehicle(self, xk, Pk, mode):
        """
        General Time Update function for Ground Vehicle. Uses mode
        to determine which motion model to use.
        """

        h = xk[4] #Heading

        #G - matrix to set the process noise properly
        G = np.array([
            [((self.Ts**2)/2)*np.cos(h), 0],
            [((self.Ts**2)/2)*np.sin(h), 0],
            [self.Ts, 0],
            [0, 0],
            [0, (self.Ts**2)/2],
            [0, self.Ts]
        ])

        #Set the correct Fk according to mode
        if mode == ModesVeh.IDLE:
            Fk = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0]
            ])
        elif mode == ModesVeh.CTPV:
            w = xk[5]

            xcorr = (2/w) * np.sin(((w*self.Ts)/2)) * \
                   np.cos(h + ((w*self.Ts)/2))

            ycorr = (2/w) * np.sin(((w*self.Ts)/2)) * \
                   np.sin(h + ((w*self.Ts)/2))

            Fk = np.array([
                [1, 0, xcorr, 0, 0, 0],
                [0, 1, ycorr, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, self.Ts],
                [0, 0, 0, 0, 0, 1]
            ])
        elif mode == ModesVeh.CTPVA:
            w = xk[5]

            xcorr = (2/w) * np.sin(((w*self.Ts)/2)) * \
                   np.cos(h + ((w*self.Ts)/2))

            ycorr = (2/w) * np.sin(((w*self.Ts)/2)) * \
                   np.sin(h + ((w*self.Ts)/2))

            Fk = np.array([
                [1, 0, xcorr, 0, 0, 0],
                [0, 1, ycorr, 0, 0, 0],
                [0, 0, 1, self.Ts, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, self.Ts],
                [0, 0, 0, 0, 0, 1]
            ])
        else:
            print(f"Got in default for Vehicle TU with mode {mode}")
            Fk = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0]
            ])

        #Update xk and Pk
        xk = Fk@xk
        Qk = G@self.q@np.transpose(G)
        Pk = Fk@Pk@np.transpose(Fk) + Qk

        return xk, Pk

    def _measurement_update_vehicle(self, xk, Pk, meas, mode):
        """
        Measurement update for the pedestrian model
        using a Kalman Filter.
        """

        #Measurement x, y, theta
        #yk = meas #Use if we visualize with visualize_imm.py
        yk = np.array([meas.x, meas.y, meas.theta]) #Use if we use ROS

        # Observation matrix H
        Hk = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])

        #Mean for measurment
        yk_hat = Hk@xk # (48)

        #Innovation covariance
        Sk = Hk@Pk@np.transpose(Hk) + self.Rk # (49)

        #Kalman gain
        Kk = Pk@np.transpose(Hk)@np.linalg.inv(Sk) # (50)

        # Measurement update
        xk = xk + Kk@(yk - yk_hat) # (46)
        Pk = Pk - Kk@Sk@np.transpose(Kk) # (47)

        # Compute the likelihood given the current measurement
        likelihood = self._calc_likelihood(yk, yk_hat, Sk)

        #Assign each likelihood with mode
        self.likelihood[mode.value] = likelihood

        return xk, Pk

# ---------------------- IMM functions -----------------------
    # This section takes inspiration from the IMMderivation.pdf provided
    # by the supervisor. A link to this PDF can be found in the references in
    # the Technical Documentation.

    def _mixing(self):
        """
        Calculate the mixing probabilities and calculate the mixed estimates.
        This will be the input to each filter.
        """

        # Equation 41
        cbar = self.mu@self.trans_prob_matrix
        for i in range(self.N):
            for j in range(self.N):
                self.mu_matrix[j,i] = (self.trans_prob_matrix[j,i]*self.mu[j]) / cbar[i]

        #Equation 42
        xk0 = []
        for i in range(self.N):
            sumx, sumP = self._get_empty_container()
            for j in range(self.N):
                sumx += self.mu_matrix[j,i] * self.xks[j]
            xk0.append(sumx)

        #Equation 43
        Pk0 = []
        for i in range(self.N):
            sumx, sumP = self._get_empty_container()
            for j in range(self.N):
                sumP += self.mu_matrix[j,i] * (self.Pks[j] + ((self.xks[j] -\
                 xk0[i]) @ np.transpose(self.xks[j] - xk0[i])))
            Pk0.append(sumP)

        self.xks = xk0
        self.Pks = Pk0


    def _mode_matched_prediction_update(self):
        """
        Prediction update using each mode. Here
        simply call each Kalman Filter with the corresponding
        xk, Pk and mode.
        """
        #Equation 44 and 45
        modes_pedestrian = [ModesPed.IDLE, ModesPed.CV, ModesPed.CTPV]
        modes_vehicle = [ModesVeh.IDLE, ModesVeh.CTPV, ModesVeh.CTPVA]

        for i in range(self.N):
            if self.type == 0: #Pedestrian
                xk, Pk = self._time_update_pedestrian(self.xks[i], self.Pks[i], modes_pedestrian[i])
            else:   #Vehicle
                xk, Pk = self._time_update_vehicle(self.xks[i], self.Pks[i], modes_vehicle[i])

            self.xks[i] = xk
            self.Pks[i] = Pk


    def _mode_matched_measurement_update(self, meas):
        """
        Measurement update using each mode. Simply call
        the measurement update for each filter with xk, Pk, meas and mode.
        Also update our mode probability.
        """
        #Calculate the updated estimate and covariane from the predicted
        #estimate and covariance
        #Equation 46 - 50 and partly 51 (Calculating the likelihood)
        modes_pedestrian = [ModesPed.IDLE, ModesPed.CV, ModesPed.CTPV]
        modes_vehicle = [ModesVeh.IDLE, ModesVeh.CTPV, ModesVeh.CTPVA]

        for i in range(self.N):
            if self.type == 0: #Pedestrian
                xk, Pk = self._measurement_update_pedestrian(self.xks[i],
                              self.Pks[i], meas, modes_pedestrian[i])
            else:
                xk, Pk = self._measurement_update_vehicle(self.xks[i],
                              self.Pks[i], meas, modes_vehicle[i])
            self.xks[i] = xk
            self.Pks[i] = Pk

        #Calculate the updated mode probablity
        #Equation 51
        for i in range(self.N):
            likelihood = self.likelihood[i]
            sum1 = 0
            for j in range(self.N):
                sum1 = sum1 + self.trans_prob_matrix[j][i]*self.mu[j]
            sum2 = 0
            for l in range(self.N):
                sum3 = 0
                for jj in range(self.N):
                    sum3 = sum3 + self.trans_prob_matrix[jj][l]*self.mu[jj]
                sum2 = sum2 + self.likelihood[l] * sum3
            self.mu[i] = (likelihood * sum1) / (sum2)


    def _output_estimate_calculation(self):
        """
        Calculate the overall estimate xk and Pk
        """
        #Equation 52
        sumx, sumP = self._get_empty_container()
        for i in range(self.N):
            sumx = sumx + self.mu[i] * self.xks[i]
        self.xk = sumx

        #Equation 53
        for i in range(0, self.N):
            sumP = sumP + self.mu[i] * (self.Pks[i] + ((self.xks[i] - self.xk) *\
                   np.transpose(self.xks[i] - self.xk)))
        self.Pk = sumP


    def _imm_filter(self, meas):
        """
        Perform a cycle of the IMM-filter,
        this should update the final xk, Pk and our mu-list
        """
        self._mixing()
        self._mode_matched_prediction_update()
        self._mode_matched_measurement_update(meas)
        self._output_estimate_calculation()


    def _init_filters(self):
        """
        This function should init each Kalman Filter with some xk and Pk.
        These are added to the self.xks_p list and the self.Pks_p list.
        This function is used for the test functions such as visualize_imm.py
        """
        if self.type == 0:  #If pedestrian
            x0_IDLE = np.array([0, 0, 1.5, np.pi/4, 0])
            x0_CV = np.array([0, 0, 1.5, np.pi/4, 0])
            x0_CTPV = np.array([0, 0, 1.5, np.pi/4, np.deg2rad(0.1)])
            P0 = np.array([
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]
            ])
            self.xks = [x0_IDLE, x0_CV, x0_CTPV]
            self.Pks = [P0, P0, P0]
        else: # If vehicle
            x0_IDLE = np.array([0, 0, 10, 0, np.pi/4, 0])
            x0_CTPV = np.array([0, 0, 10, 0, np.pi/4, np.deg2rad(0.1)])
            x0_CTPVA = np.array([0, 0, 10, 0, np.pi/4, np.deg2rad(0.1)])
            P0 = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            self.xks = [x0_IDLE, x0_CTPV, x0_CTPVA]
            self.Pks = [P0, P0, P0]


    def _init_filters_from_meas(self, meas):
        """
        Initializes the filters when two sufficient measurements
        have been collected from the Obstacle Simulator.
        """
        #Set the previous time
        self.prev_t = meas.header.stamp.secs + meas.header.stamp.nsecs*1e-9

        # Set the time step to current - previous
        self.Ts = meas.header.stamp.secs + meas.header.stamp.nsecs*1e-9 - \
                        (self.meas_0.header.stamp.secs + self.meas_0.header.stamp.nsecs)

        # Find an inital velocity for the dynamic obstacle
        v0 = np.linalg.norm(np.array([meas.x - self.meas_0.x, \
                        meas.y - self.meas_0.y]))/self.Ts

        w0 = (meas.theta - self.meas_0.theta) / self.Ts


        if self.type == 0:  #If pedestrian
            x0_IDLE = np.array([meas.x, meas.y, v0, meas.theta, w0])
            x0_CV = np.array([meas.x, meas.y, v0, meas.theta, w0])
            x0_CTPV = np.array([meas.x, meas.y, v0, meas.theta, w0])
            P0 = np.array([
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]
            ])
            self.xks = [x0_IDLE, x0_CV, x0_CTPV]
            self.Pks = [P0, P0, P0]

        else: # If vehicle
            x0_IDLE = np.array([meas.x, meas.y, v0, 0,  meas.theta, w0])
            x0_CTPV = np.array([meas.x, meas.y, v0, 0, meas.theta, w0])
            x0_CTPVA = np.array([meas.x, meas.y, v0, 0, meas.theta, w0])
            P0 = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            self.xks = [x0_IDLE, x0_CTPV, x0_CTPVA]
            self.Pks = [P0, P0, P0]


#--------------------- IMM predict/correct functions ----------------------

    def predict(self, k):

        """
        Make a predict step of the IMM filter. it also performs the initial
        mixing of the states. This function is mainly used in the test files.
        """
        if k == 0:
            self._init_filters()
        else:
            self._mixing()
            self._mode_matched_prediction_update()
            self._output_estimate_calculation()


    def update(self, meas):
        """
        Make a update step of the IMM filter. This function
        also calculates the mixed output. This function is mainly used in the test files.
        """
        self._mode_matched_measurement_update(meas)
        self._output_estimate_calculation()

# -------------------- Prediction functions -------------------------------

    def _predict_trajectory(self):
        """
        Predicts future trajectory and covariance matrix of either a pedestrian
        or a vehicle n steps forward from the current state.

        OBS: This function requires the filter to be initalized first.
        """
        #Make sure the lists are empty every time we need
        # a new trajectory
        self.pred_vec_xk = []
        self.pred_vec_Pk = []

        if self.type == 0: #Pedestrian
            modes = [ModesPed.IDLE, ModesPed.CV, ModesPed.CTPV]
        else:   #Vehicle
            modes = [ModesVeh.IDLE, ModesVeh.CTPV, ModesVeh.CTPVA]

        max_index = np.argmax(self.mu)
        mode = modes[max_index] #Extract most likely mode

        #After measurement update, this is the current xk and Pk
        xk = self.xk
        Pk = self.Pk
        self.pred_vec_xk.append(xk)
        self.pred_vec_Pk.append(Pk)

        #Predict n times forward in time and save each xk and Pk
        for k in range(self.n):
            if self.type == 0: #Pedestrian
                xk, Pk = self._time_update_pedestrian(xk, Pk, mode)
            else: #Vehicle
                xk, Pk = self._time_update_vehicle(xk, Pk, mode)

            #Save the trajectory
            self.pred_vec_xk.append(xk)
            self.pred_vec_Pk.append(Pk)

# ----------------- Misc helper functions ------------------

    def _get_empty_container(self):
        """
        Return temp container of right dimension given the type
        of the dynamic obstacle.
        """
        if self.type == 0:
            xk = np.zeros(5)
            Pk = np.zeros((5,5))
        else:
            xk = np.zeros(6)
            Pk = np.zeros((6,6))
        return xk, Pk


    def _calc_likelihood(self, yk, yk_hat, Sk):
        """
        Compute log-likelihood for a filter. This can be a large negative value
        hence we take exp() of it.
        yk - last measurement
        yk_hat - mean for last measurement
        Sk - innovation covariance
        """
        likelihood = exp(self._logpdf(yk, yk_hat, Sk))
        if likelihood == 0:
            likelihood = sys.float_info.min
        return likelihood


    def _logpdf(self, x, mean = None, cov = 1, allow_singular=True):
        """
        Log of the multivariate normal probability density function.
        x - point at which to evalute
        mean - mean of the normal distribution
        cov - covariance of the normal distribution
        Utilizes the scipy implemenation of the logpdf.
        """
        if mean is not None:
            flat_mean = np.asarray(mean).flatten()
        else:
            flat_mean = None

        flat_x = np.asarray(x).flatten()
        return multivariate_normal.logpdf(flat_x, flat_mean, cov, allow_singular)


    def _update_Ts(self):
        """
        Update the time intervall Ts according
        to the current time compared to the previous one.
        """
        self.Ts = self.curr_t - self.prev_t

    def _set_dimensions(self):
        "Set the dimensions of xk and Pk accoring to type"
        if self.type == 0:
            self.xk = np.zeros(5)
            self.Pk = np.zeros((5,5))
        else:
            self.xk = np.zeros(6)
            self.Pk = np.zeros((6,6))

    def _get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion. 
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) \
        - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) \
        + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) \
        - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) \
        + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]


    def _format_output(self, stamp):
        """
        Format the output to publish on topic to the Motion Planner.
        The lists are transformed into Float32MultiArray
        """

        x = []
        y = []
        Pxx = []
        Pxy = []
        Pyx = []
        Pyy = []
        time_list = []

        #Update the lists with x and y
        for xk in self.pred_vec_xk:
            x.append(xk[0])
            y.append(xk[1])

        #Update the lists with the elements of the covariane
        for Pk in self.pred_vec_Pk:
            Pxx.append(Pk[0,0])
            Pxy.append(Pk[0,1])
            Pyx.append(Pk[1,0])
            Pyy.append(Pk[1,1])

        for t in range(len(self.pred_vec_xk)):
            time = self.Ts * t
            time_list.append(time)

        self.output.x = x
        self.output.y = y
        self.output.pxx = Pxx
        self.output.pxy = Pxy
        self.output.pyx = Pyx
        self.output.pyy = Pyy
        self.output.header.stamp = stamp
        self.output.time = time_list

# ------------------Publisher functions --------------------

    def _publish_predicted_trajectory(self):
        """
        Publishes the currently predicted trajectory (for visualization)
        """
        t = rospy.Time.now()
        pp_msg = Path()
        pp_msg.header.frame_id = "odom"
        pp_msg.header.stamp = t

        for x_pred, y_pred in zip(self.output.x, self.output.y):
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "odom"
            pose_msg.pose.position.x = x_pred
            pose_msg.pose.position.y = y_pred
            pose_msg.header.stamp = t
            pp_msg.poses.append(pose_msg)

        # Publishes predicted path message:
        self.predicted_path_pub.publish(pp_msg)


    def _visualization_pub(self):
        """
        This function visualizes the estimated position of
        the dynamic obstacle in RViz.
        """
        marker = Marker()

        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 0
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = 0.4
        marker.scale.z = 0.1
        marker.scale.y = 0.1

        if self.type == 0: #Pedestrian
            modes = [ModesPed.IDLE, ModesPed.CV, ModesPed.CTPV]
        else:   #Vehicle
            modes = [ModesVeh.IDLE, ModesVeh.CTPV, ModesVeh.CTPVA]

        max_index = np.argmax(self.mu)
        mode = modes[max_index] #Extract most likely mode

        # Set the color according to current mode
        if mode in (ModesPed.IDLE, ModesVeh.IDLE):
            # IDLE == RED
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        elif mode in (ModesPed.CV, ModesVeh.CTPV):
            # CV / CTPV = GREEN
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        else: #Gotta be CTPV or CTPVA = BLUE
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

        if self.type == 0: #Pedestrian
            quat = self._get_quaternion_from_euler(0, 0, self.xk[3])
        else: #Vehicle
            quat = self._get_quaternion_from_euler(0, 0, self.xk[4])

        # Set the pose of the marker
        marker.pose.position.x = self.xk[0]
        marker.pose.position.y = self.xk[1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        self._visualization_dynamic_obstacle_pub.publish(marker)

# ----------------- Run function ---------------------------

    def run(self):
        """
        Main function to run the IMM-filter using ROS.
        """

        rospy.loginfo(f"{self.name} Waiting for measurement...")

        while not rospy.is_shutdown():

            meas = self.meas_q.get() #Extract the first meas from the queue

            #Init the filter with the first two measurements
            if not self.is_obs_initialized: #Not init yet
                if not self.meas_0_received: #We have not received meas 0 yet
                    #Save the first measurement
                    rospy.loginfo(f"{self.name} Received first measurement")
                    self.type = meas.type
                    self._set_dimensions()
                    self.meas_0 = meas
                    self.meas_0_received = True
                else:
                    #We have received the second measurement => Init the Filter
                    rospy.loginfo(f"{self.name} Received second measurement")
                    self._init_filters_from_meas(meas)
                    self.is_obs_initialized = True
                    rospy.loginfo(f"{self.name} Filters initalized")
                    rospy.loginfo(f"{self.name} starting...")
            else:
                #Filter has been initalized, so run the prediction loop
                # Update time
                self.curr_t = meas.header.stamp.secs + meas.header.stamp.nsecs*1e-9
                self._update_Ts()

                # Perform one cycle of the IMM-algorithm
                self._imm_filter(meas)

                #Estimate a trajectory n steps into time
                #with the most likely filter
                self._predict_trajectory()

                # Publish
                self._format_output(meas.header.stamp)
                self._predictor_pub.publish(self.output)
                self._publish_predicted_trajectory()
                self._visualization_pub()

                # Update previous time to current
                self.prev_t = self.curr_t
