"""
This file aims to test the Coordinated Turn Polar Velocity model for the pedestrian.
Should work similairly for the ground vehicle.
https://liu.diva-portal.org/smash/get/diva2:734112/FULLTEXT01.pdf


OBS: Modification is needed in the imm.py file to make this test file work.
     See the imm.py file's module doc-string.
"""

from imm import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def main():

    #---------------------------- Variables -----------------------------------

    weights = np.array([1/3, 1/3, 1/3])
    trans_prob_matrix = np.array([[0.96, 0.02, 0.02], [0.02, 0.96, 0.02], [0.02, 0.02, 0.96]])

    imm = IMM(weights, trans_prob_matrix)
    meas_rate = 10 #Determines how often we should perform a meas_update

    #Ground Truth lists
    xlist = []
    ylist = []

    # Thetas - Angle to create measurement data (but not part of meas)
    # headings - The actual heading of the object moving in the circle
    thetas = []
    headings =[]

    #Prediction lists
    xpred = []
    ypred = []

    #Corrections lists
    xcorr = []
    ycorr = []

    #Sliced Covariance matrices since we only need Px and Py to properly draw
    #the covariance ellipses
    Pks_sliced = []

    #Create fixgure and ax object to draw to
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    #Number of iterations
    n = 90
    #------------------Generate some measurements-----------------------
    for i in range(n):

        if i == 0:
            theta = -np.pi/2
            heading = -np.pi
        else:
            theta = theta - np.deg2rad(4)
            heading = heading - np.deg2rad(4)
        thetas.append(theta)
        headings.append(heading)

        # make some noisy data
        x = 30*np.cos(theta) + np.random.randn()*0.1
        y = 30*np.sin(theta) + np.random.randn()*0.1

        #Measurements of the dynamic object
        meas = np.array([x, y, heading])

        #Save ground truth for drawing
        xlist.append(x)
        ylist.append(y)

    #------------------ Perform predict/update cycle --------------------

        #Initial guess
        x0 = np.array([30*np.cos(theta), 30*np.sin(theta), 10, -np.pi, -np.deg2rad(4)])
        P0 = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        mode = Modes.CTPV
        if i == 0:
            xk, Pk = imm._time_update_pedestrian(x0, P0, mode)
        else:
            xk, Pk = imm._time_update_pedestrian(xk, Pk, mode)

        #Save the predction x,y to draw
        xpred.append(xk[0])
        ypred.append(xk[1])

        Pk_sliced = Pk[0:2, 0:2]
        Pks_sliced.append(Pk_sliced)

        if i % meas_rate == 0 or i == 0:
            xk, Pk = imm._measurement_update_pedestrian(xk, Pk, meas, mode)

        #Save the measurement update x,y
            xcorr.append(xk[0])
            ycorr.append(xk[1])

#--------------------------- Drawing ----------------------------------------

    for i in range(n):
        draw_arrow(ax, xlist[i], ylist[i], thetas[i], r=1)
        draw_arrow(ax, xlist[i], ylist[i], headings[i], r=2)
        error_ellipse(ax, xpred[i], ypred[i], Pks_sliced[i], ec='black')

    plt.scatter(xlist[0], ylist[0], c="blue", label = "Start point")
    plt.scatter(xlist[1:],ylist[1:], c="green", label="Ground Truth")
    plt.scatter(xpred, ypred, color='k', label="Predictions")
    plt.scatter(xcorr, ycorr, c="red", label="Correction")

    # ------------------ Error calculation with MSE --------------------------
    euc_dists = []
    for p in range(len(xlist)):
        true_coord = np.array([xlist[p], ylist[p]])
        pred_coord = np.array([xpred[p], ypred[p]])
        dist = np.linalg.norm(true_coord - pred_coord)
        euc_dists.append(dist)

    MSE_curve = np.square(euc_dists).mean()
    print("MSE curve: ", MSE_curve)

    #Finally show the plot
    ax.legend()
    plt.grid()
    plt.show()

#-------------------------- Misc functions -------------------------------------

def error_ellipse(ax, xc, yc, cov, sigma=1, **kwargs):
    '''
    Plot an error ellipse contour over your data.
    Inputs:
    ax : matplotlib Axes() object
    xc : x-coordinate of ellipse center
    yc : x-coordinate of ellipse center
    cov : covariance matrix
    sigma : # sigma to plot (default 1)
    additional kwargs passed to matplotlib.patches.Ellipse()
    '''
    w, v = np.linalg.eigh(cov) # assumes symmetric matrix
    order = w.argsort()[::-1]
    w, v = w[order], v[:,order]
    theta = np.degrees(np.arctan2(*v[:,0][::-1]))
    ellipse = Ellipse(xy=(xc,yc),
                    width=2.*sigma*np.sqrt(w[0]),
                    height=2.*sigma*np.sqrt(w[1]),
                    angle=theta, **kwargs)
    ellipse.set_facecolor('none')
    ax.add_artist(ellipse)

def draw_arrow(ax, x, y, angle, r):
    """
    Draw an arrow from x,y with direction angle and length r. This is drawn to
    an ax object.
    """
    arrow = plt.arrow(x, y, r*np.cos(angle), r*np.sin(angle))
    ax.add_artist(arrow)

if __name__ == "__main__":
    main()
