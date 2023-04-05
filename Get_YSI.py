# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:48:13 2022

@author: yjo
"""

def Get_YSI(f_ave,pixelmm,flame_width):
    import numpy as np
    import matplotlib.pyplot as plt
    
    #Find the middle
    n_middle=int(len(f_ave[0,:])/2)
    R=flame_width/2*pixelmm*1e-3
    F_z=np.zeros(len(f_ave))
    r=np.zeros(n_middle) 
    F_z_temp=0    
    #Sum over the whole axis till the flame tip (f_ave*2pi*pixelmm*radius_from_the_centre)*(1/pi/R^2)
    for i in range(0,len(f_ave)):
        
        for j in range(0,n_middle):
            r[j]=R-((j)*pixelmm*10**-3)  #n_middle is approximately the flame radius
            F_z[i]=abs(r[j]*f_ave[i,j]*(pixelmm*10**-3))+F_z_temp
            F_z_temp=F_z[i]   
        F_z_temp=0    
        F_z[i]=2/R**2*F_z[i]
    
    F_z_max=np.max(F_z)
    #plt.figure()
    #plt.plot(np.arange(len(f_ave)),F_z)
    #plt.show()
    return (F_z_max)