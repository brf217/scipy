#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 21:00:28 2018

@author: feebr01
"""

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt



#take random choices with probabilities of each from discrete list
from random import choices
population = [1, 2, 3, 4, 5, 6]
weights = [0.1, 0.05, 0.05, 0.2, 0.4, 0.2]
choices(population, weights)



#BINOMIAL DISTRIBUTIONS 
#################################################################################
#draw binomial pmf for n = 10 tries with p success = 0.5. Define distribution
x = np.arange(0,11) # set up points to plot at
binom_probs = stats.binom.pmf(x, n = 10, p = .5 )
plt.bar(x, binom_probs)


# sample binomial values from 10k samples and plot the resulting curve
binom_sim = stats.binom.rvs(n=10, p=0.5, size = 10000)
fig, ax = plt.subplots()
ax.hist(binom_sim, bins = np.arange(11)-.5, normed = True) # align bins properly
ax.set_xticks(range(10))


# figure out prob of success of a specific #successes (3 successes  n=16, psuccess = 1/6)
stats.binom.pmf(3,16,1/6)



#POISSON DISTRIBUTIONS
#############################################################################
#draw poisson plot with mu as number of times something occurs over time/dist/etc.
mu = 15 # mean expected occurences per continuous interval/unit
n = np.arange(0,30) # points to plot dist at (linspace doesn't work great)
poisson = stats.poisson.pmf(n, mu, loc=0) # use loc to shift the distribution - optional
plt.plot(n, poisson, 'o-')


# sample poisson values from 1k samples and plot the resulting curve
poisson_sim = stats.poisson.rvs(mu=10, loc=0, size = 100000)
plt.hist(poisson_sim, bins=50, normed = True)


# get cdf or pdf from poisson
stats.poisson.cdf(12.5, mu) #p of up to observed value, mu
stats.poisson.ppf(.95, mu)  #percentile you want value for, mu
stats.poisson.sf(7, mu)     #chances of observing value greater than first arg

# typically see 8 per hour, what is prob of 4
stats.poisson.pmf(4, mu=8)



#GAMMA DIST
##############################################################################
x = np.linspace(0,10, 50)
y = stats.gamma.pdf(x, a=2, scale = .7)
plt.plot(x, y)

# mean = alpha * beta (scale)
# variance = alpha * beta ^2



#NORMAL DISTRIBUTIONS
##############################################################################
mean = 0
sd = 1
x = np.arange(-5,5, 0.1)
normal = stats.norm.pdf(x, mean, sd)
plt.plot(x, normal, 'o-')


# sample normal values from 10k samples and plot the resulting curve
normal_sim = stats.norm.rvs(loc=0, scale =1, size = 10000)  # use loc to shift the distribution
plt.hist(normal_sim, normed = True)


#get cdf or pdf from normal dist
stats.norm(100,10).cdf(100) #p measuring any value up to and including x
stats.norm(100,10).ppf(.95) # get value at percentile defined (95th)
stats.norm.sf(110,100,10) #observation, mean, sd - output prob of value greater than first arg


#confidence interval
stats.norm.interval(.95, 0,1)
#can check interval like this
stats.norm.ppf(.025, 0,1)
stats.norm.ppf(.975, 0,1)


# compute based on SE vs. knowing pop SD. Adjust pop SD by sqrt(n)
stats.norm.cdf(2.875, 3.125, .7/np.sqrt(40))

# no pmf-like function for individual values on curve like Poisson (since p() indiv. value = 0)



#T DIST df ~n1+n2-2
#########################################################################
stats.t.cdf(1.5,df=10) #cdf using observation and degrees of freedom of t dist
stats.t.ppf(0.95,df=10) #ppf defining cdf and df of t dist


# one sample t test example - dist and value to see if significantly different
female_doctor_bps = [128, 127, 118, 115, 144, 142, 133, 140, 132, 131, 
                     111, 132, 149, 122, 139, 119, 136, 129, 126, 128]
stats.ttest_1samp(female_doctor_bps, 120)


# students t test for mean difference. 2 tail default
female_doctor_bps = [128, 127, 118, 115, 144, 142, 133, 140, 132, 131, 
                     111, 132, 149, 122, 139, 119, 136, 129, 126, 128]

male_consultant_bps = [118, 115, 112, 120, 124, 130, 123, 110, 120, 121,
                      123, 125, 129, 130, 112, 117, 119, 120, 123, 128]

# 2 tail version by default so /2 for one tail
two_tail = stats.ttest_ind(female_doctor_bps, male_consultant_bps, equal_var = False).pvalue
one_tail = two_tail/2


# paired t test for before and after
control = [8.0, 7.1, 6.5, 6.7, 7.2, 5.4, 4.7, 8.1, 6.3, 4.8]
treatment = [9.9, 7.9, 7.6, 6.8, 7.1, 9.9, 10.5, 9.7, 10.9, 8.2]

stats.ttest_rel(control, treatment)



#CHI2 comparing expected to observed
#############################################################################

stats.chisquare(f_obs=[19,31,18,32], f_exp=[25,20,15,40])

stats.chi2.cdf(12,df=1)
stats.chi2.ppf(0.95, df=7)

stats.chi2.cdf(2, df=1)



#FITTER TO FIND THE PROPER DISTRIBUTION TO USE
############################################################################
data = stats.norm.rvs(loc=0, scale =1, size = 10000)

from fitter import Fitter
f = Fitter(data)
f.fit()

# get output
f.summary()

# get params of output you want to see (mean and sd for normal etc.)
fit.fitted_param['norm']




# ANOVA and PostHoc  / F distribution
############################################################################

# ANOVA for two or more samples
ctrl = [4.17, 5.58, 5.18, 6.11, 4.5, 4.61, 5.17, 4.53, 5.33, 5.14]
trt1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
trt2 = [6.31, 5.12, 5.54, 5.5, 5.37, 5.29, 4.92, 6.15, 5.8, 5.26]

stats.f_oneway(ctrl, trt1, trt2)


f_value, p_value = stats.f_oneway(data1, data2, data3, data4, ...)

#TukeysHSD post hoc (slightly different library)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
print (pairwise_tukeyhsd(Data, group_split_dimension))



# simple regression in statsmodels
############################################################################
X_price = [12,14,16,18,20]
y_units = [54,57,49,48,42]
import statsmodels.api as sm
X_price = sm.add_constant(X_price)
reg = sm.OLS(y_units, X_price).fit()
reg.summary()
reg.fittedvalues
reg.resid

X_price = np.array([12,14,16,18,20])
y_units = np.array([54,57,49,48,42])

import seaborn as sns
sns.regplot(x=X_price, y=y_units)
sns.residplot(x=X_price, y=y_units)

# Matrix ops and linear algebra notes
###########################################################################
m = sympy.Matrix([[1,2,4,5], [2,4,5,4], [4,5,4,2]])
# solve system of equations reduced row echelon form
m.rref()

# rank of matrix
m.rank()

# determinant (square only). 0 = singular / non invertable (linearly dependent cols)
m_square = sympy.Matrix([[2,7,5], [2,4,5], [4,5,4]])
m_square.det()

# getting inverse / dividing. Inv * original = identity. Square and full rank only.
m_square.inv()
# check that inverse creates identity matrix (not element-wise)
m_square.inv()*m_square


# example 1 - left 3x3 augmented with identity matrix gives inverted left
m_ex = sympy.Matrix([[1,2,3,1,0,0], [1,3,4,0,1,0], [1,2,5,0,0,1]])
m_ex.rref()


# Regression implementations manual
#############################################################################
# linear algebra method on least squares - left inverse
# coefficent_beta = inverse of (XtX) * X transpose * y

x = m_square = sympy.Matrix([[2,7,5], [2,4,5], [4,5,4]])
y = sympy.Matrix([11.2, 4.5, 6.7])


# get coefficient using inversion #1
(x.T*x).inv() * x.T*y

# solve coefficients using rref method #2
mod = x.col_insert(4,y)
mod.rref()










