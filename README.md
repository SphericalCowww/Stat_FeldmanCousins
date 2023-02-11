# Using the Feldman Cousins Method to Fit a Gaussian from scratch
For a parameter that has a bound, the "classical" confidence interval doesn't take the bound into the account.
The Feldman Cousin method is an unified approach (both powerful and converges correctly) to "updates" the classical confidence interval to account for the bound. 

In this example, the parameters are the &mu; and &sigma; of a Gaussian with the following constraints:<br/>
&ensp;&ensp;&mu; > 0.0 and &sigma; < 1.0,<br/>
while the samples are drown from a Gaussian with<br/>
&ensp;&ensp;&mu; = 0.1 and &sigma; = 0.8.

The code runs on python3 with additional packages:

    pip3 install scipy
    pip3 install tqdm
    python3 FeldmanCousinsGaus.py
The code outputs the following image:

<img src="https://github.com/SphericalCowww/Stat_FeldmanCousins/blob/master/gausFeldmanCousins_Display.png" width="630" height="490">

- Top-left: blue distribution are the sample drawn from the red Gaussian curve. The top-left coner gives the confidence interval using the point estimate (Pt Est), maximum likelihood method (Max Like), least square, and least chi-square (Least Chi2). We can see that these confidence intervals does not respect the constraints (&mu; > 0.0) and (&sigma; < 1.0). The point estimate confident interval for &sigma; is shown to be not availabe; it's possible to do, but very difficult. The maximum likelihood confidence intervals are the ones passed down to the Feldman Cousin analysis.
- Bottom-left: the Feldman Cousin confidence belt of parameter &mu;. We can see that although the estimate gives &mu; = -0.16 (shown by the green vertical line) that is outside the bound purely out of statistical reason, the Feldman Cousin method gives a one-sided confidence interval of [0.0, 0.14]. This updated interval not only satisfies the constraint of (&mu; > 0.0), but it also contains the true value of &mu; = 0.1.
- Bottom-right: the Feldman Cousin confidence belt of parameter &sigma;. We can see that if the estimate, i.e. &sigma; = 0.76 is somewhat away from the bound of (&sigma; < 1.0), then the Feldman Cousin method recovers the two-side confidence interval the same as the classical one.

Alternatively:

    python3 FeldmanCousinsEvaluate.py
The code outputs the 1-sigma Feldman-Cousins bound of a given mean and standard error with the constraint that the mean is greater than 0. The parameters should be adjustable within the code near the beginning.

As an example, the code can outputs the following image, which should be equivalent to Fig.10 from Ref.1:

<img src="https://github.com/SphericalCowww/Stat_FeldmanCousins/blob/master/gausFeldmanCousinsGausEvaluate_Display.png" width="400" height="400">

Additional note:
- In term of Neyman construction, the "classical" method is a "shortcut" to get the frequentist confidence interval by using only Eq.25 or Eq.26 of Ref.1. This "shortcut" breaks down in case of, for instance, when the parameter has a bound. A "full" Neyman construction requires varying the confidence interval direction through constructing the entire "confidence belt", and the Feldman Cousins method is one such "full" Neyman construction.
- The document fomularFeldmanCousins.pdf provides the analysical formulas for the Feldman Cousin confidence belt for lower and upper bounds. The derivation is done on the &mu; of a Gaussian, but the idea is similar for all other parameters and distributions.



References:
- G. J. Feldman and R. D. Cousins, Phys. Rev. D 57, 3873 (1998) (<a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.57.3873">Phy Rev D</a>, <a href="https://arxiv.org/abs/physics/9711021">arXiv</a>)
- B. Cousins, Virvual Talk, Univ. of California, Los Angeles (2011) (<a href="http://www.physics.ucla.edu/~cousins/stats/cousins_bounded_gaussian_virtual_talk_12sep2011.pdf">PPT</a>)
