import sys, math
import re
import time
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from copy import deepcopy
from scipy import optimize
from scipy import stats
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore");



###################################################################################
def main():
    verbosity = 1


    val = 0.33
    err = 0.68
    boundEdge = 0
    boundType = "lower"


    stepSize = min(abs(val), err)/200.0
    yRange = None
    if   boundType in ["l", "L", "lower", "Lower"]: yRange = [boundEdge, val+5*err]
    elif boundType in ["u", "U", "upper", "Upper"]: yRange = [val-5*err, boundEdge]
    xRange = [yRange[0]-0.5*(yRange[1]-yRange[0]),\
              yRange[1]+0.5*(yRange[1]-yRange[0])]
    alpha = 0.682689492137086 #1-sigma, two-sided significant prob
    ########################### 
    if verbosity >= 1: print("Processing Feldman&Cousins...")
    ySpan         = []
    xConfIntUList = []
    xConfIntLList = []
    ySE = err
    yRangeN = int((yRange[1] - yRange[0])/stepSize) + 1
    for i in (tqdm(range(yRangeN)) if verbosity>=1 else range(yRangeN)):
        yVal = yRange[0] + i*stepSize;
        optResult = FCopt(boundType, boundEdge, yVal, ySE, alpha)
        confInt1 = optResult.x
        confInt2 = FCconfIntPartner(boundType, boundEdge, confInt1, yVal,ySE,alpha)
        ySpan.append(yVal)
        xConfIntUList.append(max(confInt1, confInt2))
        xConfIntLList.append(min(confInt1, confInt2))

    xSpan = []
    yEstList  = []
    xRangeN = int((xRange[1] - xRange[0])/stepSize)
    for i in range(xRangeN): 
        xVal = xRange[0] + i*stepSize
        xSpan   .append(xVal)
        yEstList.append(estFunc(boundType, boundEdge, xVal))

    xEst = val
    upperIdx = min(enumerate(xConfIntLList), key=lambda u: abs(u[1]-xEst))[0]
    estIdx   = min(enumerate(xSpan),         key=lambda u: abs(u[1]-xEst))[0]
    lowerIdx = min(enumerate(xConfIntUList), key=lambda u: abs(u[1]-xEst))[0]
    yEst = yEstList[estIdx]
    yConfInt = [ySpan[lowerIdx], ySpan[upperIdx]]
#plots
    fig = plt.figure(figsize=(6*(xRange[1]-xRange[0])/(yRange[1]-yRange[0]), 7))
    gs = gridspec.GridSpec(1, 1)
    mpl.rc("xtick", labelsize=16)
    mpl.rc("ytick", labelsize=16)
    ax0 = fig.add_subplot(gs[0])
    ax0.ticklabel_format(style="sci", scilimits=(-2, 2))
    ax0.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax0.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax0.tick_params(which="major", width=2, length=5)
    ax0.tick_params(which="minor", width=1, length=3)
    
    ax0.plot(xSpan,         yEstList, color="black",    linewidth=2)
    ax0.plot(xConfIntLList, ySpan,    color="darkgrey", linewidth=2)
    ax0.plot(xConfIntUList, ySpan,    color="darkgrey", linewidth=2)
    title  = "Feldman-Cousins Confidence Interval, "
    title += "Est=" + scientificStr(val) + ", "
    title += "SE="  + scientificStr(err)
    ax0.set_title(title, fontsize=24, y=1.03)
    ax0.set_xlabel("est span", fontsize=18)
    ax0.set_ylabel("FC conf",  fontsize=18)
    ax0.set_xlim(xRange[0], xRange[1])
    ax0.set_ylim(yRange[0], yRange[1])
    ax0.set_aspect(1, adjustable="box")
    ax0.axvline(x=xEst, ymin=0, ymax=1, color="green")
    plotRatio = (xEst - xRange[0])/(xRange[1] - xRange[0])
    ax0.axhline(y=yConfInt[0], xmin=0, xmax=plotRatio, color="green",linestyle=":")
    ax0.axhline(y=yConfInt[1], xmin=0, xmax=plotRatio, color="green",linestyle=":")
 
    labelStr  = "est = " + scientificStr(yEst)
    labelStr += "+" + scientificStr(yConfInt[1]-yEst)
    labelStr += "-" + scientificStr(yEst-yConfInt[0]) + "\n"
    labelStr += "inv = [" + scientificStr(yConfInt[0]) + ", "
    labelStr +=             scientificStr(yConfInt[1]) + "]"
    font = {"family": "serif", "color": "green", "weight": "bold", "size": 18}
    ax0.text(xEst, yRange[1] - 0.045*(yRange[1]-yRange[0]), labelStr,fontdict=font)
    if verbosity >= 1: 
        print("F&C CI: est = "+str(yEst)+"+"+str(yConfInt[1]-yEst)+\
                                         "-"+str(yEst-yConfInt[0]))
        print("        conf inv =", yConfInt)
        print("        stepSize = "+str(stepSize))
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__))
    filenameFig = exepath + "/gausFeldmanCousinsEvaluate.png"
    gs.tight_layout(fig)
    plt.savefig(filenameFig)

    if verbosity >= 1:
        print("Creating the following files:")
        print(filenameFig)

















###################################################################################
TOLERANCE = pow(10.0, -10);
SNUMBER   = pow(10.0, -124);
def gaussian(mu, sig, x):
    X = np.array(x);
    vals = np.exp(-np.power(X-mu,2.0)/(2.0*np.power(sig,2.0)))\
         *(1.0/(sig*np.sqrt(2.0*np.pi)));
    vals[vals < SNUMBER] = SNUMBER;
    return vals;
def logGaus(mu, sig, x):
    X = np.array(x);
    vals = -np.log(sig*np.sqrt(2.0*np.pi))\
           -np.power(X-mu,2.0)/(2.0*np.power(sig,2.0));
    LL = sum(vals);
    return LL;
def negLogLikelihood(x):
    return lambda par : -1.0*logGaus(par[0], par[1], x);
def estFunc(boundDir, bound, parStat):  #(estFunc(x) = x) except at the boundary
    if boundDir in ["u", "U", "upper", "Upper"]:
        return min(bound, parStat);
    elif boundDir in ["l", "L", "lower", "Lower"]:
        return max(bound, parStat);
    else:
        print("ERROR: estFunc: unrecognized boundDir input.");
        sys.exit(0);
def FCconfIntPartner(boundDir, bound, confInt1, par, parErr, alpha):
    if par == bound:
        if boundDir in ["u", "U", "upper", "Upper"]:
            return 1.0/SNUMBER;
        elif boundDir in ["l", "L", "lower", "Lower"]:
            return -1.0/SNUMBER;
        else:
            print("ERROR: FCconfIntPartner: unrecognized boundDir input.");
            sys.exit(0);
    alphaCov = alpha + (1.0 - alpha)/2.0;
    errBarRatio = stats.norm.ppf(alphaCov);
    confInt2 = 2*par*confInt1 - pow(confInt1, 2) - pow(bound, 2);
    confInt2 = confInt2/(2*(par - bound));
    if boundDir in ["u", "U", "upper", "Upper"]:
        if par + errBarRatio*parErr <= bound:
            confInt2 = par + errBarRatio*parErr;
    elif boundDir in ["l", "L", "lower", "Lower"]:
        if par - errBarRatio*parErr >= bound:
            confInt2 = par - errBarRatio*parErr;
    else:
        print("ERROR: FCconfIntPartner: unrecognized boundDir input.");
        sys.exit(0);
    return confInt2;
def FCconfIntProbAlpha(boundDir, bound, confInt1, par, parErr, alpha):
    if parErr <= 0:
        print("ERROR: FCconfIntProbAlpha: parErr < 0.")
        sys.exit(0);
    if (boundDir in ["u", "U", "upper", "Upper"]) and (par > bound) or\
       (boundDir in ["l", "L", "lower", "Lower"]) and (par < bound):
        print("ERROR: FCconfIntProbAlpha: bound condition violated.")
        sys.exit(0);
    confInt2 = FCconfIntPartner(boundDir, bound, confInt1, par, parErr, alpha);
    prob = abs(stats.norm.cdf(confInt1, loc=par, scale=parErr)\
              -stats.norm.cdf(confInt2, loc=par, scale=parErr));
    return prob;
def FCoptFunc(boundDir, bound, par, parErr, alpha):
    return lambda confInt1 : \
        abs(FCconfIntProbAlpha(boundDir, bound, confInt1, par, parErr, alpha)\
           -alpha);
def FCopt(boundDir, bound, par, parErr, alpha):
    optBound = (0.0, 0.0);
    if boundDir in ["u", "U", "upper", "Upper"]:
        optBound = (par-10.0*parErr, par);
    elif boundDir in ["l", "L", "lower", "Lower"]:
        optBound = (par, par+10.0*parErr);
    else:
        print("ERROR: FCopt: unrecognized boundDir input.");
        sys.exit(0);
    FCconfIntProb = FCoptFunc(boundDir, bound, par, parErr, alpha);
    return optimize.minimize_scalar(FCconfIntProb, tol=TOLERANCE,
                                    method="bounded", bounds=optBound);
#round a number
def roundSig(val, sigFig=2):
    if val == 0:
        return val;
    return round(val, sigFig-int(np.floor(np.log10(abs(val))))-1);
#give scientific figure string output of a number
def scientificStr(val, sigFig=2):
    valStr = ""
    if val == 0:
        valStr = "0.0"
    elif abs(np.floor(np.log10(abs(val)))) < sigFig:
        valStr = str(roundSig(val, sigFig=sigFig))
        if valStr[-2:] == ".0": valStr = valStr[:-2]
    else:
        valStr = "{:." + str(sigFig-1) + "E}"
        valStr = valStr.format(val)
        valStr = valStr.replace("E+0", "E+")
        valStr = valStr.replace("E+", "E")
        valStr = valStr.replace("E0", "")
        valStr = valStr.replace("E-0", "E-")
    return valStr
if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");
