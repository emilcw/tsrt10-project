from imm import IMM, ModesPed, ModesVeh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

"""
This file aims to test the IMM filter using the three different motion models.
First we move forward using like straight line (USE CV), then we turn left (USE CTPV)
and finally we stop (USE IDLE.) 


OBS: Modification is needed in the imm.py file to make this test file work.
     See the imm.py file's module doc-string.
"""

def main():

    #---------------------------- Variables -----------------------------------

    #Initally all models are equal in probability
    mu = np.array([1/3, 1/3, 1/3])

    #Inital values for the trans_prob_matrix
    trans_prob_matrix = np.array([[0.96, 0.02, 0.02], [0.02, 0.96, 0.02], [0.02, 0.02, 0.96]])

    IMMestimator = IMM(mu, trans_prob_matrix)

    IMMestimator.Ts = 0.7 #For this specific problem to work.

    meas_rate = 2 #Determines how often we should perform a meas_update

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

    measurements = []

    #Create fixgure and ax object to draw to
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    #Number of iterations
    n = 90
    thresh = 50
    #----------------------Generate some measurements---------------------------
    for i in range(n):
        if i == 0:
            theta = 0
            heading = np.pi/4

        # Make some noisy data
        if i >= thresh:
            #If we are above some threshold, make it circular
            if i == thresh:
                theta = theta - np.deg2rad(2) + np.pi/2.2
                heading = -np.pi/12
            else:
                theta = theta - np.deg2rad(6)
                heading = heading - np.deg2rad(6)
            if i > n - 20:
                pass
            else:
                x = 45 + 30*np.cos(theta) #+ np.random.randn()*0.1
                y = 20 + 30*np.sin(theta) #+ np.random.randn()*0.1
        else:
            #Otherwise a straight line
            x = i #+ np.random.randn()*0.1
            y = i #+ np.random.randn()*0.1
            heading = np.pi/4

        #Save the measuremetns
        meas = np.array([x, y, heading])
        measurements.append(meas)
        thetas.append(theta)
        headings.append(heading)

        #Save ground truth for drawing
        xlist.append(x)
        ylist.append(y)

    #------------------ Perform predict/update cycle --------------------
    mus =[]
    for k in range(n):

        IMMestimator.predict(k)

        if k % meas_rate == 0:
            IMMestimator.update(measurements[k])
            xcorr.append(measurements[k][0])
            ycorr.append(measurements[k][1])

        xpred.append(IMMestimator.xk[0])
        ypred.append(IMMestimator.xk[1])
        index = np.argmax(IMMestimator.mu)
        if IMMestimator.type == 0:
            modelist = [ModesPed.IDLE, ModesPed.CV, ModesPed.CTPV]
        else:
            modelist = [ModesVeh.IDLE, ModesVeh.CTPV, ModesVeh.CTPVA]
        mus.append(modelist[index])

        Pk_sliced = IMMestimator.Pk[0:2, 0:2]
        Pks_sliced.append(Pk_sliced)


#--------------------------- Drawing
    for i in range(n):
        #draw_arrow(ax, xlist[i], ylist[i], thetas[i], r=1)
        #draw_arrow(ax, xlist[i], ylist[i], headings[i], r=2)
        error_ellipse(ax, xpred[i], ypred[i], Pks_sliced[i], ec='black')

        if mus[i] == ModesVeh.IDLE:
            plt.scatter(xpred[i], ypred[i], color='k')
        elif mus[i] == ModesVeh.CTPV:
            plt.scatter(xpred[i], ypred[i], color='pink')
        else:
            plt.scatter(xpred[i], ypred[i], color='cyan')

        plt.annotate(i, (xpred[i], ypred[i]))

    plt.scatter(xlist[0], ylist[0], c="blue", label = "Start point")
    plt.scatter(xlist[1:],ylist[1:], c="green", label="Ground Truth")
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
                    width=2.*sigma*np.sqrt(abs(w[0])),
                    height=2.*sigma*np.sqrt(abs(w[1])),
                    angle=theta, **kwargs)
    ellipse.set_facecolor('none')
    ax.add_artist(ellipse)

def draw_arrow(ax, x, y, angle, r):
    """
    Draw an arrow from x,y with direction angle and length r. This is drawn to
    an Axes() object.
    """
    arrow = plt.arrow(x, y, r*np.cos(angle), r*np.sin(angle))
    ax.add_artist(arrow)

if __name__ == "__main__":
    main()
