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


#NOTE: to estimate the number of counts, error bars on estimating the amplitude
#      won't be Gaussian when count is low; need to use Poisson, which on its
#      own provides the standard error (see Ref.1). However, Poissin error 
#      would neglect the shape of the signal pdf if known, which would then 
#      require the assumption on the number of counts of the noise events
#######################################################################################
def main():
    verbosity = 2
    binN = 200
    rangeX = [-10.0, 10.0]

    np.random.seed(2)
    dataMu  = 0.1
    dataSig = 0.8
    dataN   = 30

    fitRange = [-2.0, 2.0]      #for least square/chi2 fit

    alpha = 0.95                #two-sided significant prob
    FCstepSize = 0.01
    
    muRange     = [0.0, 2.0]    #range for FC span
    muBoundDir  = "lower"
    muTitle     = "Feldman-Cousins Confidence Interval: Gaus, mu>0"
    muBound     = muRange[0]
    muStatRange = [muRange[0]-1.0, muRange[1]]

    sigRange     = [-1.0, 1.0]
    sigBoundDir  = "upper"
    sigTitle     = "Feldman-Cousins Confidence Interval: Gaus, sigma<1.0"
    sigBound     = sigRange[1]
    sigStatRange = [sigRange[0], sigRange[1]+1.0]
#data
    dataSamp = np.random.normal(dataMu, dataSig, dataN)
    xVals = np.linspace(*rangeX, binN+1)[:-1]
    dataHist = np.histogram(dataSamp, bins=binN, range=rangeX)[0]   #bin2left  
#point estimate
    valMu  = np.average(dataSamp)
    errMu  = np.std(dataSamp)/np.sqrt(dataN)
    valSig = np.sqrt(np.var(dataSamp))
    errSig = -1
#maximum likelihood 
    if verbosity >= 1: print("Processing maximum likelihood...")
    optInitVals = [valMu, valSig]
    negMaxLL = negLogLikelihood(dataSamp)
    optResult = optimize.minimize(negMaxLL, optInitVals)
    maxLikeMu, maxLikeSig = optResult.x
    if verbosity >= 2: print(optResult, "\n")

    maxErrMu  = maxLikeSig*np.sqrt(1.0/dataN)
    maxErrSig = maxLikeSig*np.sqrt(1.0/(2.0*dataN))
#adjust for fit range
    fitRange = [-2.0, 1.0]

    fitMask = ((fitRange[0] <= xVals)&(xVals < fitRange[1]))
    xValsFit    = xVals[fitMask]
    dataHistFit = dataHist[fitMask]
    #WARNING: least square/chi2 error bars are always small; related to 0 count bins?
#least square
    if verbosity >= 1: print("Processing least square...")
    resSquareFunc = squareResGaus(xValsFit, dataHistFit)
    optInitVals = [np.max(dataHistFit), valMu, valSig]
    optResult = optimize.minimize(resSquareFunc, optInitVals)
    leastSquareA, leastSquareMu, leastSquareSig = optResult.x
    if verbosity >= 2: print(optResult, "\n")

    hessianInv      = optResult.hess_inv
    resSquare       = optResult.fun
    degreeOfFreedom = len(dataHistFit) - len(optInitVals) - 1 
    lsErrA   = np.sqrt(hessianInv[0][0]*resSquare/degreeOfFreedom)
    lsErrMu  = np.sqrt(hessianInv[1][1]*resSquare/degreeOfFreedom)
    lsErrSig = np.sqrt(hessianInv[2][2]*resSquare/degreeOfFreedom)
#least chi2
    if verbosity >= 1: print("Processing least chi2...")
    dataHistFitErr = [count if count > 0 else 1 for count in dataHistFit]
    dataHistFitErr = np.sqrt(dataHistFitErr)
    resChi2Func = chiSquareGaus(xValsFit, dataHistFit, dataHistFitErr)
    optInitVals = [np.max(dataHistFit), valMu, valSig]
    optResult = optimize.minimize(resChi2Func, optInitVals)
    leastChi2A, leastChi2Mu, leastChi2Sig = optResult.x
    if verbosity >= 2: print(optResult, "\n")

    hessianInv      = optResult.hess_inv
    chi2            = optResult.fun
    degreeOfFreedom = len(dataHistFit) - len(optInitVals) - 1
    chi2ErrA   = np.sqrt(hessianInv[0][0]*chi2/degreeOfFreedom)
    chi2ErrMu  = np.sqrt(hessianInv[1][1]*chi2/degreeOfFreedom)
    chi2ErrSig = np.sqrt(hessianInv[2][2]*chi2/degreeOfFreedom)
#parameters and errors used for FC correction
    muHat  = maxLikeMu
    muSE   = maxErrMu
    sigHat = maxLikeSig 
    sigSE  = maxErrSig
#Feldman&Cousins with condition mu > 0
    if verbosity >= 1: print("Processing Feldman&Cousins for mu...")
    muSpan         = []
    muConfIntUList = []
    muConfIntLList = []
    muRangeN = int((muRange[1]-muRange[0])/FCstepSize) + 1
    for i in (tqdm(range(muRangeN)) if verbosity>=1 else range(muRangeN)):
        mu      = muRange[0] + i*FCstepSize
        muErr   = muSE
        optResult = FCopt(muBoundDir, muBound, mu, muErr, alpha)
        confInt1 = optResult.x
        confInt2 = FCconfIntPartner(muBoundDir, muBound, confInt1, mu, muErr, alpha)
        muSpan.append(mu)
        muConfIntUList.append(max(confInt1, confInt2))
        muConfIntLList.append(min(confInt1, confInt2))
    muStatSpan = []
    muEstList  = []
    muStatRangeN = int((muStatRange[1]-muStatRange[0])/FCstepSize)
    for i in range(muStatRangeN): 
        muStat = muStatRange[0] + i*FCstepSize
        muEst = estFunc(muBoundDir, muBound, muStat)
        muStatSpan.append(muStat)
        muEstList.append(muEst)

    muUpperIdx = min(enumerate(muConfIntUList), key=lambda u: abs(u[1]-muHat))[0]
    muLowerIdx = min(enumerate(muConfIntLList), key=lambda u: abs(u[1]-muHat))[0]
    muEstIdx   = min(enumerate(muStatSpan),     key=lambda u: abs(u[1]-muHat))[0]
    muEst = muEstList[muEstIdx]
    muConfInt = [muEst - muSpan[muUpperIdx], muSpan[muLowerIdx] - muEst]
    if muConfInt[0] < 1.1*FCstepSize: muConfInt[0] = 0.0
    if muConfInt[1] < 1.1*FCstepSize: muConfInt[1] = 0.0
#Feldman&Cousins with condition sigma < 1.0
    if verbosity >= 1: print("Processing Feldman&Cousins for sigma...")
    sigSpan         = []
    sigConfIntUList = []
    sigConfIntLList = []
    sigRangeN = int((sigRange[1]-sigRange[0])/FCstepSize)
    for i in (tqdm(range(sigRangeN)) if verbosity>=1 else range(sigRangeN)):
        sig    = sigRange[0] + i*FCstepSize
        sigErr = sigSE
        optResult = FCopt(sigBoundDir, sigBound, sig, sigErr, alpha)
        confInt1 = optResult.x
        confInt2 = FCconfIntPartner(sigBoundDir, sigBound, confInt1, sig, sigErr,alpha)
        sigSpan.append(sig)
        sigConfIntUList.append(max(confInt1, confInt2))
        sigConfIntLList.append(min(confInt1, confInt2))

    sigStatSpan = []
    sigEstList  = []
    sigStatRangeN = int((sigStatRange[1]-sigStatRange[0])/FCstepSize)
    for i in range(sigStatRangeN): 
        sigStat = sigStatRange[0] + i*FCstepSize
        sigEst = estFunc(sigBoundDir, sigBound, sigStat)
        sigStatSpan.append(sigStat)
        sigEstList.append(sigEst)

    sigUpperIdx=min(enumerate(sigConfIntUList), key=lambda u: abs(u[1]-sigHat))[0]
    sigLowerIdx=min(enumerate(sigConfIntLList), key=lambda u: abs(u[1]-sigHat))[0]
    sigEstIdx  =min(enumerate(sigStatSpan),     key=lambda u: abs(u[1]-sigHat))[0]
    sigEst = sigEstList[sigEstIdx]
    sigConfInt = [sigEst - sigSpan[sigUpperIdx], sigSpan[sigLowerIdx] - sigEst]
    if sigConfInt[0] < 1.1*FCstepSize: sigConfInt[0] = 0.0
    if sigConfInt[1] < 1.1*FCstepSize: sigConfInt[1] = 0.0
#alpha confidence convertion
    alphaCov = alpha + (1.0 - alpha)/2.0
    errBarRatio = stats.norm.ppf(alphaCov)
    errMu       = errBarRatio*errMu
    if errSig > 0: errSig = errBarRatio*errSig
    maxErrMu   = errBarRatio*maxErrMu
    maxErrSig  = errBarRatio*maxErrSig
#plots
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    #plot 0
    gausPlotOrig         = gaussian(dataMu,        dataSig,        xVals)
    gausPlotOrig        *= np.sum(dataHist)/np.sum(gausPlotOrig)
    gausPlotPtEst        = gaussian(valMu,         valSig,         xVals)
    gausPlotPtEst       *= np.sum(dataHist)/np.sum(gausPlotPtEst)
    gausPlotMaxLike      = gaussian(maxLikeMu,     maxLikeSig,     xVals)
    gausPlotMaxLike     *= np.sum(dataHist)/np.sum(gausPlotMaxLike)
    gausPlotLeastSquare  = gaussian(leastSquareMu, leastSquareSig, xValsFit)
    gausPlotLeastSquare *= leastSquareA
    gausPlotChi2         = gaussian(leastChi2Mu,   leastChi2Sig, xValsFit)
    gausPlotChi2        *= leastChi2A
    
    ##Choose one to plot
    xValsPlot = xValsFit
    gausPlot = gausPlotLeastSquare
    ##################################################################################
    
    ax0.plot(xVals, dataHist, color="blue", drawstyle="steps-post")
    ax0.hist(dataSamp, bins=binN, range=rangeX, histtype="step", color="black")

    ax0.plot(xValsPlot, gausPlot, linewidth=3, alpha=0.8, color="red")
    ax0.axhline(y=0, color="black", linestyle="-")
    ax0.axvline(x=np.average(dataSamp), ymin=0, ymax=1, color="green", linestyle="--")
    ax0.set_title("Point Estimate vs Maximum Likelihood", fontsize=24, y=1.03)
    ax0.set_xlabel("x", fontsize=18)
    ax0.set_ylabel("count", fontsize=18)
    ax0.set_xlim(rangeX[0]-1.0, rangeX[1]+1.0)
    ax0.set_ylim(bottom=-0.1)    

    digit0       = -math.floor(math.log10(errMu)) + 1
    valMu0r      = ("{:." + str(digit0) + "f}").format(valMu)
    errMu0r      = ("{:." + str(digit0) + "f}").format(errMu)
    valSig0r     = ("{:." + str(digit0) + "f}").format(valSig)
    errSig0r     = "NA"
    if errSig > 0: errSig0r = ("{:." + str(digit0) + "f}").format(errSig)
    maxLikeMu0r  = ("{:." + str(digit0) + "f}").format(maxLikeMu)
    maxErrMu0r   = ("{:." + str(digit0) + "f}").format(maxErrMu)
    maxLikeSig0r = ("{:." + str(digit0) + "f}").format(maxLikeSig)
    maxErrSig0r  = ("{:." + str(digit0) + "f}").format(maxErrSig)
    leastSquareMu0r  = ("{:." + str(digit0) + "f}").format(leastSquareMu)
    lsErrMu0r        = ("{:." + str(digit0) + "f}").format(lsErrMu)
    leastSquareSig0r = ("{:." + str(digit0) + "f}").format(leastSquareSig)
    lsErrSig0r       = ("{:." + str(digit0) + "f}").format(lsErrSig)
    leastChi2Mu0r  = ("{:." + str(digit0) + "f}").format(leastChi2Mu)
    chi2ErrMu0r    = ("{:." + str(digit0) + "f}").format(chi2ErrMu)
    leastChi2Sig0r = ("{:." + str(digit0) + "f}").format(leastChi2Sig)
    chi2ErrSig0r   = ("{:." + str(digit0) + "f}").format(chi2ErrSig)
    ymin, ymax = ax0.get_ylim()
    font = {"family": "serif", "color": "green", "weight": "bold", "size": 18}

    ax0.text(rangeX[0], 0.92*(ymax-ymin), "Pt Est: ", fontdict=font)
    strTemp = "    mu = " + str(valMu0r) + "$\pm$" + str(errMu0r)
    ax0.text(rangeX[0], 0.88*(ymax-ymin), strTemp, fontdict=font)
    strTemp = "    sig = " + str(valSig0r) + "$\pm$" + str(errSig0r)
    ax0.text(rangeX[0], 0.84*(ymax-ymin), strTemp, fontdict=font)
    ax0.text(rangeX[0], 0.78*(ymax-ymin), "Max Like: ", fontdict=font)
    strTemp = "    mu = " + str(maxLikeMu0r) + "$\pm$" + str(maxErrMu0r)
    ax0.text(rangeX[0], 0.74*(ymax-ymin), strTemp, fontdict=font)
    strTemp = "    sig = " + str(maxLikeSig0r) + "$\pm$" + str(maxErrSig0r)
    ax0.text(rangeX[0], 0.70*(ymax-ymin), strTemp, fontdict=font)
    ax0.text(rangeX[0], 0.64*(ymax-ymin), "Least Square: ", fontdict=font)
    strTemp = "    mu = " + str(leastSquareMu0r) + "$\pm$" + str(lsErrMu0r)
    ax0.text(rangeX[0], 0.60*(ymax-ymin), strTemp, fontdict=font)
    strTemp = "    sig = " + str(leastSquareSig0r) + "$\pm$" + str(lsErrSig0r)
    ax0.text(rangeX[0], 0.56*(ymax-ymin), strTemp, fontdict=font)
    ax0.text(rangeX[0], 0.50*(ymax-ymin), "Least Chi2: ", fontdict=font)
    strTemp = "    mu = " + str(leastChi2Mu0r) + "$\pm$" + str(chi2ErrMu0r)
    ax0.text(rangeX[0], 0.46*(ymax-ymin), strTemp, fontdict=font)
    strTemp = "    sig = " + str(leastChi2Sig0r) + "$\pm$" + str(chi2ErrSig0r)
    ax0.text(rangeX[0], 0.42*(ymax-ymin), strTemp, fontdict=font)
    #plot 2
    ax2.plot(muStatSpan,     muEstList, color="black",    linewidth=2)
    ax2.plot(muConfIntLList, muSpan,    color="darkgrey", linewidth=2)
    ax2.plot(muConfIntUList, muSpan,    color="darkgrey", linewidth=2)
    ax2.set_title(muTitle, fontsize=18, y=1.03)
    ax2.set_xlabel("mu_stat", fontsize=18)
    ax2.set_ylabel("mu", fontsize=18)
    ax2.set_xlim(muStatRange[0], muStatRange[1])
    ax2.set_ylim(muRange[0], muRange[1])
    ax2.set_aspect(1, adjustable="box")
    ax2.axvline(x=muHat, ymin=0, ymax=1, color="green")
    muHatRatio = (muHat - muStatRange[0])/(muStatRange[1] - muStatRange[0])
    ax2.axhline(y=(muEst-muConfInt[0]), xmin=0, xmax=muHatRatio, color="green",\
                linestyle=":")
    ax2.axhline(y=(muEst+muConfInt[1]), xmin=0, xmax=muHatRatio, color="green",\
                linestyle=":")
 
    font = {"family": "serif", "color": "green", "weight": "bold", "size": 18}
    digit2  = -math.floor(math.log10(max(muConfInt))) + 1
    valMu2r  = ("{:." + str(digit2) + "f}").format(muEst)
    errMu2rN = ("{:." + str(digit2) + "f}").format(muConfInt[0])
    errMu2rP = ("{:." + str(digit2) + "f}").format(muConfInt[1])
    strTemp = "mu = " + str(valMu2r)
    if muConfInt[1] > 0: strTemp = strTemp + "+" + str(errMu2rP)
    if muConfInt[0] > 0: strTemp = strTemp + "-" + str(errMu2rN)
    ax2.text(muHat, muRange[1]-0.04*(muRange[1]-muRange[0]), strTemp, fontdict=font)
    #plot 3
    ax3.plot(sigStatSpan,     sigEstList, color="black",    linewidth=2)
    ax3.plot(sigConfIntLList, sigSpan,    color="darkgrey", linewidth=2)
    ax3.plot(sigConfIntUList, sigSpan,    color="darkgrey", linewidth=2)
    ax3.set_title(sigTitle, fontsize=18, y=1.03)
    ax3.set_xlabel("sigma_stat", fontsize=18)
    ax3.set_ylabel("sigma", fontsize=18)
    ax3.set_xlim(sigStatRange[0], sigStatRange[1])
    ax3.set_ylim(sigRange[0], sigRange[1])
    ax3.set_aspect(1, adjustable="box")
    ax3.axvline(x=sigHat, ymin=0, ymax=1, color="green")
    sigHatRatio = (sigHat - sigStatRange[0])/(sigStatRange[1] - sigStatRange[0])
    ax3.axhline(y=(sigEst-sigConfInt[0]), xmin=0, xmax=sigHatRatio, color="green",\
                linestyle=":")
    ax3.axhline(y=(sigEst+sigConfInt[1]), xmin=0, xmax=sigHatRatio, color="green",\
                linestyle=":")
 
    font = {"family": "serif", "color": "green", "weight": "bold", "size": 18}
    digit3  = -math.floor(math.log10(max(sigConfInt))) + 1
    valSig3r  = ("{:." + str(digit3) + "f}").format(sigEst)
    errSig3rN = ("{:." + str(digit3) + "f}").format(sigConfInt[0])
    errSig3rP = ("{:." + str(digit3) + "f}").format(sigConfInt[1])
    strTemp = "sig = " + str(valSig3r)
    if sigConfInt[1] > 0: strTemp = strTemp + "+" + str(errSig3rP)
    if sigConfInt[0] > 0: strTemp = strTemp + "-" + str(errSig3rN)
    ax3.text(sigHat, sigRange[0]+0.01*(sigRange[1]-sigRange[0]), strTemp,fontdict=font)

    if verbosity >= 1:
        print("Pt Est: ")
        print("    mu  = " + str(valMu) + " +/- " + str(errMu))
        print("    sig = " + str(valSig), end = "")
        if errSig > 0: print(" +/- " + str(errSig))
        else:          print("");
        print("Max Like: ")
        print("    mu  = " + str(maxLikeMu)  + " +/- " + str(maxErrMu))
        print("    sig = " + str(maxLikeSig) + " +/- " + str(maxErrSig))
        print("Least Square: ")
        print("    mu  = " + str(leastSquareMu)  + " +/- " + str(lsErrMu))
        print("    sig = " + str(leastSquareSig) + " +/- " + str(lsErrSig))
        print("    A   = " + str(leastSquareA)   + " +/- " + str(lsErrA))
        print("Least Chi2: ")
        print("    mu  = " + str(leastChi2Mu)  + " +/- " + str(chi2ErrMu))
        print("    sig = " + str(leastChi2Sig) + " +/- " + str(chi2ErrSig))
        print("    A   = " + str(leastChi2A)   + " +/- " + str(chi2ErrA))
        print("F&C CI:")
        print("    mu  = " + str(muEst)  + \
              " + " + str(muConfInt[1])  + " - " + str(muConfInt[0]))
        print("    sig = " + str(sigEst) + \
              " + " + str(sigConfInt[1]) + " - " + str(sigConfInt[0]))
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__))
    filenameFig = exepath + "/gausFeldmanCousins.png"
    gs.tight_layout(fig)
    plt.savefig(filenameFig)
    if verbosity >= 1: print("Creating the following files:\n" + filenameFig)

#######################################################################################
TOLERANCE = pow(10.0, -10)
SNUMBER   = pow(10.0, -124)
def gaussian(mu, sig, x):
    X = np.array(x)
    vals = np.exp(-np.power(X-mu,2.0)/(2.0*np.power(sig,2.0)))\
         *(1.0/(sig*np.sqrt(2.0*np.pi)))
    vals[vals < SNUMBER] = SNUMBER
    return vals
def squareResGaus(x, y):
    return lambda par : np.sum(np.square(y - par[0]*gaussian(par[1], par[2], x)))
def chiSquareGaus(x, y, yerr):
    if np.sum(yerr[yerr <= 0]) > 0: 
        raise ValueError("chiSquareGaus(): non-positive yerr values") 
    return lambda par : np.sum(np.square((y - par[0]*gaussian(par[1], par[2],x))/yerr))
def logGaus(mu, sig, x):
    X = np.array(x)
    vals = -np.log(sig*np.sqrt(2.0*np.pi))\
           -np.power(X-mu,2.0)/(2.0*np.power(sig,2.0))
    LL = sum(vals)
    return LL
def negLogLikelihood(x):
    return lambda par : -1.0*logGaus(par[0], par[1], x)

def estFunc(boundDir, bound, parStat):
    if boundDir in ["u", "U", "upper", "Upper"]:   return min(bound, parStat)
    elif boundDir in ["l", "L", "lower", "Lower"]: return max(bound, parStat)
    else: raise ValueError("estFunc(): unrecognized boundDir input")
def FCconfIntPartner(boundDir, bound, confInt1, par, parErr, alpha):
    if par == bound:
        if boundDir in ["u", "U", "upper", "Upper"]:   return 1.0/SNUMBER
        elif boundDir in ["l", "L", "lower", "Lower"]: return -1.0/SNUMBER
        else: raise ValueError("FCconfIntPartner(): unrecognized boundDir input")
    alphaCov = alpha + (1.0 - alpha)/2.0
    errBarRatio = stats.norm.ppf(alphaCov)
    confInt2 = 2*par*confInt1 - pow(confInt1, 2) - pow(bound, 2)
    confInt2 = confInt2/(2*(par - bound))
    if boundDir in ["u", "U", "upper", "Upper"]:
        if par + errBarRatio*parErr <= bound: confInt2 = par + errBarRatio*parErr
    elif boundDir in ["l", "L", "lower", "Lower"]:
        if par - errBarRatio*parErr >= bound: confInt2 = par - errBarRatio*parErr
    else: raise ValueError("FCconfIntPartner(): unrecognized boundDir input")
    return confInt2;
def FCconfIntProbAlpha(boundDir, bound, confInt1, par, parErr, alpha):
    if parErr <= 0: raise ValueError("FCconfIntProbAlpha(): parErr < 0")
    if (boundDir in ["u", "U", "upper", "Upper"]) and (par > bound) or\
       (boundDir in ["l", "L", "lower", "Lower"]) and (par < bound):
       raise ValueError("FCconfIntProbAlpha(): bound condition violated")
    confInt2 = FCconfIntPartner(boundDir, bound, confInt1, par, parErr, alpha)
    prob = abs(stats.norm.cdf(confInt1, loc=par, scale=parErr)\
              -stats.norm.cdf(confInt2, loc=par, scale=parErr))
    return prob
def FCoptFunc(boundDir, bound, par, parErr, alpha):
    return lambda confInt1 : \
        abs(FCconfIntProbAlpha(boundDir, bound, confInt1, par, parErr, alpha) - alpha)
def FCopt(boundDir, bound, par, parErr, alpha):
    optBound = (0.0, 0.0)
    if boundDir in ["u", "U", "upper", "Upper"]:   optBound = (par-10.0*parErr, par)
    elif boundDir in ["l", "L", "lower", "Lower"]: optBound = (par, par+10.0*parErr)
    else: ValueError("FCopt(): unrecognized boundDir input")
    FCconfIntProb = FCoptFunc(boundDir, bound, par, parErr, alpha) 
    return optimize.minimize_scalar(FCconfIntProb, tol=TOLERANCE,\
                                    method="bounded", bounds=optBound)

#######################################################################################
if __name__ == "__main__":
    print("\n####################################################################Head")
    main()
    print("######################################################################Tail")




