# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:43:52 2018

@author: jd766
"""

def S_fun_filter(T,l,cam_resp,lambda_first,alpha):
    import numpy as np
    # Planck's constant in m^2 kg / s
    h = 6.62607004e-34
    
    # Boltzmann constant in m^2 kg / s^2 / K
    k = 1.38064852e-23;
    
    # Speed of light in m / s
    c = 299792458;

    i=(np.round(l*10-lambda_first).astype(int))
    
    # See Eq. 6 in:
    # Kempema and Long, Optics Letters 43 (2018) 1103-1106 Correction for self absorption
    S_i =  cam_resp[i] / ((l * 10**(-9))**(5+alpha)) / (np.exp(h*c/((l*10**(-9))*k*T))-1)
    return (S_i)