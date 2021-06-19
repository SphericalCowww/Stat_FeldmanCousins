# Feldman Cousins Method Fitting a Gaussian 
For a parameter that has a bound, the classical confidence interval doesn't take the bound into the account.
The Feldman Cousin method is an unified approach (both powerful and converges correctly) to "updates" the classical confidence interval to account for the bound.

In this example, the parameters are the &mu; and &sigma; of a Gaussian with the following constraints:<br/>
&ensp;&ensp;&mu; > 0.0 and &sigma; < 1.0,<br/>
while the samples are drown from a Gaussian with &mu;=0.1 and &sigma;=0.8.

The code runs on python3 with additional packages:

    pip3 install scipy
    pip3 install tqdm
    python3 FeldmanCousinsGaus.py
The code outputs the following image:
- Top-left: blue distribution are the sample drone from the red Gaussian curve. The top-left coner gives the confidence interval using the point estimate (Pt Est) and maximum likelihood method (Max Like). We can see that these confidence intervals does not respect the constraints &mu; > 0.0 and &sigma; < 1.0. The point estimate confident interval for &sigma; is not availabe; it's possible to do, but very difficult. The maximum likelihood confidence intervals are the ones passed down to the Feldman Cousin analysis.
- Bottom-left: the Feldman Cousin confidence belt of parameter &mu;.

<img src="https://github.com/Rabbitybunny/Stat_FeldmanCousins/blob/master/gausFeldmanCousins.png" width="630" height="490">


References:
- G. J. Feldman and R. D. Cousins, Phys. Rev. D 57, 3873 (1998) (<a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.57.3873">Phy Rev D</a>, <a href="https://arxiv.org/abs/physics/9711021">arxiv</a>)
- B. Cousins, Virvual Talk, Univ. of California, Los Angeles (2011) (<a href="http://www.physics.ucla.edu/~cousins/stats/cousins_bounded_gaussian_virtual_talk_12sep2011.pdf">PPT</a>)
