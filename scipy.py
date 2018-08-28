#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:02:11 2018

@author: feebr01
"""

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


#check where a number sits in the distribution
prob = stats.norm.pdf(3, loc=0, scale=1.0)  #loc is mean, scale is sd, First number is check pdf



#create distribtuion with mean 3.5 and sd2
dist = stats.norm(loc=3.5, scale =2)
dist.rvs() #draw a random sample from the dist



#one sample t test example - dist and value to see if significantly different
female_doctor_bps = [128, 127, 118, 115, 144, 142, 133, 140, 132, 131, 
                     111, 132, 149, 122, 139, 119, 136, 129, 126, 128]
stats.ttest_1samp(female_doctor_bps, 120)



#two sample difference for independence
female_doctor_bps = [128, 127, 118, 115, 144, 142, 133, 140, 132, 131, 
                     111, 132, 149, 122, 139, 119, 136, 129, 126, 128]

male_consultant_bps = [118, 115, 112, 120, 124, 130, 123, 110, 120, 121,
                      123, 125, 129, 130, 112, 117, 119, 120, 123, 128]

stats.ttest_ind(female_doctor_bps, male_consultant_bps)


#paired t test for before and after
control = [8.0, 7.1, 6.5, 6.7, 7.2, 5.4, 4.7, 8.1, 6.3, 4.8]
treatment = [9.9, 7.9, 7.6, 6.8, 7.1, 9.9, 10.5, 9.7, 10.9, 8.2]

stats.ttest_rel(control, treatment)



#ANOVA for two or more samples
ctrl = [4.17, 5.58, 5.18, 6.11, 4.5, 4.61, 5.17, 4.53, 5.33, 5.14]
trt1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
trt2 = [6.31, 5.12, 5.54, 5.5, 5.37, 5.29, 4.92, 6.15, 5.8, 5.26]

stats.f_oneway(ctrl, trt1, trt2)



#take random choices with probabilities of each from discrete list
from random import choices
population = [1, 2, 3, 4, 5, 6]
weights = [0.1, 0.05, 0.05, 0.2, 0.4, 0.2]
choices(population, weights)



#BINOMIAL DISTRIBUTIONS

########################## draw binomial pmf for 10 tries with p success of 0.3. Define distribution
n=10 
p=0.3
k = np.arange(0,20) # of tries to plot
binomial = stats.binom.pmf(k,n,p)
plt.plot(binomial, 'o-')


# sample binomial values from 10k samples and plot the resulting curve
binom_sim = stats.binom.rvs(n=10, p=0.3, size = 10000)
plt.hist(binom_sim, bins=10, normed = True)




#POISSON DISTRIBUTIONS

########################## draw poisson plot with mu as number of times something occurs over time
mu = 5 
n = np.arange(0,20) # of tries to plot
poisson = stats.poisson.pmf(n, mu, loc=0) # use loc to shift the distribution
plt.plot(n, poisson, 'o-')


# sample poisson values from 1k samples and plot the resulting curve
poisson_sim = stats.poisson.rvs(mu=2, loc=0, size = 10000)
plt.hist(poisson_sim, bins=10, normed = True)

# get cdf or pdf from poisson
stats.poisson.cdf(12.5, mu) #observed value, mu
stats.poisson.ppf(.95, mu)  #percentile you want, mu
stats.poisson.sf(7, mu)     #chances of observing value greater than first arg



#NORMAL DISTRIBUTIONS

########################## draw poisson plot with mu as number of times something occurs over time
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
stats.norm(100,10).ppf(.95) #ppf at percentile
stats.norm.sf(110,100,10) #observation, mean, sd - output prob of value greater than first arg



#T DIST

##################################
stats.t.cdf(1.5,df=10) #cdf using observation and degrees of freedom of t dist
stats.t.ppf(0.95,df=10) #ppf defining cdf and df of t dist


#CHI2

##################################
stats.chi2.cdf(12.0,df=5)
stats.chi2.ppf(0.95, df=7)


#FITTER TO FIND THE PROPER DISTRIBUTION TO USE
data = stats.norm.rvs(loc=0, scale =1, size = 10000)

from fitter import Fitter
f = Fitter(data)
f.fit()
f.summary()




