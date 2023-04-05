# -*- coding: utf-8 -*-
"""
This python code was developed to calculate soot temperatures and volume fractions of co-flow diffusion flames using colour ratio pyrometry. 
Details of the code can be found in the CoMo c4e preprint 217 (https://como.ceb.cam.ac.uk/preprints/217/) and in the 
manuscript Dreyer et al., Applied Optics 58 (2019) 2662-2670 (https://doi.org/10.1364/AO.58.002662). Please cite these sources when using this code.

The code uses a newly developed Abel inversion method for reconstructing flame cross-section intensities from their 2D projections recorded
by a camera. This method, called fitting the line-of-sight projection of a predefined intensity distribution (FLiPPID), was developed by Radomir Slavchov
and was implemented into Python by Angiras Menon. The rest of the code was written by Jochen Dreyer.

The Code was modified by: Prithviraj Phadnis
The modifications include contour detection for the flame edge detection and automation of the processing of the files 
in a batch process.

At: OWI Science4Fuels gGmbH
Info: https://www.owi-aachen.de/


"""
import matplotlib.pyplot as plt
import matplotlib as mpl 
import numpy as np
import pandas as pd
import sys
import skimage
from skimage import io
import time
import scipy as sc
from scipy import ndimage
from os.path import abspath, exists
from scipy.signal import savgol_filter
import tkinter
import tkinter.filedialog
import os
import os.path
import glob as glob
import abel
import cv2

# Most of the required Python packages should be installed by default. Two additional ones used by this code are PyAbel and OpenCV-Python:
# https://pyabel.readthedocs.io/en/latest/index.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/index.html
# Details and installation instructions can be found on the above websites. PyAbel us used to perfor the inverse Abel transform if methods other than 
# FLiPPID are used. OpenCV is a package for Computer Vision and Machine Learning and is used for the demosaicing of the Bayer raw images.



# General parameters
# Process new image or load old one? Available are 'Load' and 'Pre-process'.
#If new image is processed check parameters in corresponding case structure
load_image = 'Pre-process'

# Which method should be used for the inverse Abel transform? Some standard Abel inverse methods are available as implemented in the PyAbel package:
# https://pyabel.readthedocs.io/en/latest/index.html
# These currently are 'BASEX', '3-point', 'Dasch-Onion', and 'Hansen-Law'. It should be noted that in its current version, a regularization of the raw
# data is only possible with BASEX. 

# Also the FLiPPID method as reported in Dreyer et al., Applied Optics, 2019 is available. 
# Note that 'FLiPPID' will take around 35-60 min for the provided example image.
Abel = 'Hansen-Law'

# Get flame temperatures by comparing colour ratios to lookup table? Note that a temperature lookup table is 
# required. If this table is not found, the program calculates the table using a calibration curve and camera response data.
get_temperature = True

# Get soot volume fraction (only possible if get_temperature = True)?
get_volume_fraction= True

# Save the soot temperature, volume fraction, and, if FLiPPID was used, the fitting parameters?
save = False

# Filter used during the experiment and camera calibration. Options are 'nofilter', 'FGB7', and 'FGB39'. Others can be added in 
# the function Temperature_lookup.
filter = 'FGB7'
#------------------------------------------------------------------------------
# Select flame image files and parameters for image processing
#------------------------------------------------------------------------------

if load_image == 'Load':
    # Load a previously pre-processed flame image. Only the file path for the csv file containg the red colour channel has to be defined.
    ImPathRed = 'Photos//Processed//100H_BG7_exp1563_Red_Ave5.csv'
    ImPathGrn = ImPathRed.replace('Red','Grn')
    ImPathBlu = ImPathRed.replace('Red','Blu')
    
    ImRed = np.genfromtxt(ImPathRed, delimiter=',')
    ImGrn = np.genfromtxt(ImPathGrn, delimiter=',')
    ImBlu = np.genfromtxt(ImPathBlu, delimiter=',')
    
    # Define locations where the soot temperature and volume fractions are saved. Also a path for the FLiPPID optimised fitting parameters is defined.
    ImPathT = ImPathRed.replace('Red','T-ave')
    ImPathfv = ImPathRed.replace('Red','fv')
    ImPathfit = ImPathRed.replace('Red','fit_out')
            
elif load_image == 'Pre-process':
    # Set filename, how many frames to be processed (the example image contains 5 frames), and if the cropped raw image is plotted.
    #filename = 'N6_ISO200_SS141-5adhd-1_1010'#'N2ISO100_SS115'#RohF-SS1d15' #'DSC-195-199_s'#'0-5ss_s'
    average = 1
    exposure = 1/32*1e6 #1/50*1e6 #1/8*1e6 #'100H_BG7_exp1563'
    plot_raw =True
    
    # If save_single=False, only the averaged values of the pre-processed images are saved. If True, also each individual frame is saved.
    save_single = True
    
    # Parameters defining position of flame and cropped image size. Please use an uneven integer for flame_width.
    flame_height = 1700 #1600
    flame_width = 443 #401#411
    HAB0 = 1732 #1975 #1782 #1647  #2444 #'100H_BG7_exp1563'
    
    # Threshold for selecting non-tilted flames with similar height. Flames with a tip larger than +-thresh_tip are removed. Flames with a standard 
    # deviation of the centreline larger than thresh_std are removed. Note that an error will occur if non of the frames fulfills these conditions.
    thresh_tip = 20
    thresh_std = 100
    # Define locations where the soot temperature and volume fractions are saved. Also a path for the FLiPPID optimised fitting parameters is defined.
    # ImPathT = ('{0}{1}{2}'.format('Photos//Processed//', filename, 'T-ave.csv'))
    # ImPathfv = ('{0}{1}{2}'.format('Photos//Processed//', filename, 'fv.csv'))
    # ImPathfit = ('{0}{1}{2}'.format('Photos//Processed//', filename, 'fit_out.csv'))
    
    ######### DIALOG BOX ########################
    ''' Opens up a dialog box asking for the path of the folder which contains the TIF Images '''
    root=tkinter.Tk()
    root.withdraw()
    
    # filepath=tkinter.filedialog.askopenfilename() #
    filepath=tkinter.filedialog.askdirectory()
    print(filepath) #
        
    ################################################
    num_files=0
    for fname in glob.glob(os.path.join(filepath, '*.tif')):
        num_files= num_files+1
    print(num_files,' total files inside the folder')
    ################################################
    
    ########### CREATE EMPTY ARRAYS TO STORE VALUES ######################
    ''' Creates arrays to store the information such as the fv values and the filename '''
    fv_array = np.zeros(shape=(num_files))
    filename_list = []
    
    ## Counter variable 
    check_num = 0
    
    start = time.time()
    
    ''' Looping through each and every image with an extension .TIF inside the folder to process it '''
    for fname in glob.glob(os.path.join(filepath, '*.tif')):
        # print(fname)
        filename= os.path.splitext(os.path.basename(fname))[0]
        print('Filename: ', filename)
        from Get_flame_rev import Get_flame
        [ImRed, ImGrn, ImBlu, ImDevRed, ImDevGrn, ImDevBlu] = Get_flame(fname, filename, average, flame_height, HAB0, flame_width, plot_raw, thresh_tip, thresh_std, save_single)
    # else: 
    #     sys.exit("Selected image_load option does not exist. Programme stopped")
    # print(filepath)
    # fname=filepath
        
        filename_list.append(filename)
        # print(filename_list)
        # Set image background to 0 if below 200 counts.
        ImRed[ImRed <200] = 0 #<200
        ImGrn[ImGrn <200] = 0 #<200
        ImBlu[ImBlu <200] = 0 #<200

# Set bottom 20 pixel rows to 0 to remove reflections from burner exit
        ImRed[len(ImRed)-20:len(ImRed),:] = 0
        ImGrn[len(ImGrn)-20:len(ImGrn),:] = 0
        ImBlu[len(ImBlu)-20:len(ImBlu),:] = 0

# start = time.time()

# The first few options use the PyAbel package for the inverse Abel transform. For details see:
# https://pyabel.readthedocs.io/en/latest/index.html
# Just some of the available transforms are used but others can easly be added.

        ImRed_half = ((ImRed[:,round(len(ImRed[0])/2):len(ImRed[0])]+np.flip(ImRed[:,0:round(len(ImRed[0])/2)+1], axis=1)) / 2)
        ImGrn_half = ((ImGrn[:,round(len(ImGrn[0])/2):len(ImGrn[0])]+np.flip(ImGrn[:,0:round(len(ImGrn[0])/2)+1], axis=1)) / 2)
        ImBlu_half = ((ImBlu[:,round(len(ImBlu[0])/2):len(ImBlu[0])]+np.flip(ImBlu[:,0:round(len(ImBlu[0])/2)+1], axis=1)) / 2)

        if Abel == 'BASEX':
            # These are the BASEX regularisation parameters. 
            sig = 8 #8
            qx = 2.2 # 2.2
        
            # R_red_half = abel.basex.basex_transform(ImRed_half, sigma=sig, reg=qx, correction=True, direction=u'inverse')
            # R_grn_half = abel.basex.basex_transform(ImGrn_half, sigma=sig, reg=qx, correction=True, direction=u'inverse')
            # R_blu_half = abel.basex.basex_transform(ImBlu_half, sigma=sig, reg=qx, correction=True, direction=u'inverse')
            R_red_half = abel.basex.basex_transform(ImRed_half, sigma=sig, reg=qx, correction=True, basis_dir=u'./BASEX_matrices/', dr=1.0, verbose=True, direction=u'inverse')
            R_grn_half = abel.basex.basex_transform(ImGrn_half, sigma=sig, reg=qx, correction=True, basis_dir=u'./BASEX_matrices/', dr=1.0, verbose=True, direction=u'inverse')
            R_blu_half = abel.basex.basex_transform(ImBlu_half, sigma=sig, reg=qx, correction=True, basis_dir=u'./BASEX_matrices/', dr=1.0, verbose=True, direction=u'inverse')
            
        if Abel == '3-point':
            R_red_half = abel.dasch.three_point_transform(ImRed_half, basis_dir=u'./3-point_matrices', dr=1, direction=u'inverse', verbose=False)
            R_grn_half = abel.dasch.three_point_transform(ImGrn_half, basis_dir=u'./3-point_matrices', dr=1, direction=u'inverse', verbose=False)
            R_blu_half = abel.dasch.three_point_transform(ImBlu_half, basis_dir=u'./3-point_matrices', dr=1, direction=u'inverse', verbose=False)
            
        elif Abel == 'Dasch-Onion':
            R_red_half = abel.dasch.onion_peeling_transform(ImRed_half, basis_dir=u'./Dasch-Onion_matrices', dr=1, direction=u'inverse', verbose=False)
            R_grn_half = abel.dasch.onion_peeling_transform(ImGrn_half, basis_dir=u'./Dasch-Onion_matrices', dr=1, direction=u'inverse', verbose=False)
            R_blu_half = abel.dasch.onion_peeling_transform(ImBlu_half, basis_dir=u'./Dasch-Onion_matrices', dr=1, direction=u'inverse', verbose=False)
            
        elif Abel == 'Hansen-Law':
            R_red_half = abel.hansenlaw.hansenlaw_transform(ImRed_half, dr=1, direction=u'inverse', hold_order=1, sub_pixel_shift=0)
            R_grn_half = abel.hansenlaw.hansenlaw_transform(ImGrn_half, dr=1, direction=u'inverse', hold_order=1, sub_pixel_shift=0)
            R_blu_half = abel.hansenlaw.hansenlaw_transform(ImBlu_half, dr=1, direction=u'inverse', hold_order=1, sub_pixel_shift=0)
            
        elif Abel == 'FLiPPID':
            from FLiPPID import FLiPPID
            # For which z values should FLiPPID be executed? Allowed is a range (z_min, z_max) or string 'all'
            z_range = ('all')# 'all'
            
            # Select function for R to fit to the recorded data. Available are:
            # fun1: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^6)
            # fun2: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^8)
            # fun3: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^10)
            # fun4: a/(b*sqrt(pi)) * exp(c(r/b)^2-(r/b)^12)
            fit_fun = 'fun1'
            
            # Define range of integral lookup table. Nx is the range for x/b*delta_c, Nc is the range for c*delta_c. 
            delta_x = 0.01
            delta_c = 0.01
            if fit_fun == 'fun1':
                Nx = 200
                Nc = (-500, 2500)
                
            elif fit_fun == 'fun2':
                Nx = 200
                Nc = (-500, 2500)
        
            elif fit_fun == 'fun3':
                Nx = 170
                Nc = (-300, 2500)
        
            elif fit_fun == 'fun4':
                Nx = 170
                Nc = (-300, 2200)
                
            if z_range == 'all':
                del z_range
                z_range=(0,len(ImRed))
                
            [R_red, R_grn, R_blu, P_red, P_grn, P_blu, fit_out] = FLiPPID(ImRed, ImGrn, ImBlu, z_range, Nx, Nc, delta_x, delta_c, fit_fun) 
            
            # Find maximum root mean square error.
            rmse_red_max = [ n for n,p in enumerate(fit_out[:,3,0]) if p==max(fit_out[:,3,0]) ][0]
            rmse_grn_max = [ n for n,p in enumerate(fit_out[:,3,1]) if p==max(fit_out[:,3,1]) ][0]
            rmse_blu_max = [ n for n,p in enumerate(fit_out[:,3,2]) if p==max(fit_out[:,3,2]) ][0]
            
            rmse_max_red = float("{0:.2f}".format(fit_out[rmse_red_max,3,0])) 
            rmse_max_grn = float("{0:.2f}".format(fit_out[rmse_grn_max,3,1])) 
            rmse_max_blu = float("{0:.2f}".format(fit_out[rmse_blu_max,3,2])) 
            
            print('{0}{1}{2}{3}{4}{5}'.format('Max. standard deviation of the residuals, red=', str(rmse_max_red), ', green=', str(rmse_max_grn), ', blue=', str(rmse_max_blu)))
        
            # plt.figure()
            # plt.plot(ImRed[rmse_red_max,:], 'r')
            # plt.plot(ImGrn[rmse_grn_max,:], 'g')
            # plt.plot(ImBlu[rmse_blu_max,:], 'b')
            # plt.plot(P_red[rmse_red_max,:], '-r')
            # plt.plot(P_grn[rmse_grn_max,:], '-g')
            # plt.plot(P_blu[rmse_blu_max,:], '-b')
            # plt.title('The worst FLiPPID fits for the red, green, and blue channel')
            # plt.show() 
        
        # The transforms in PyAbel only process one half of the image. Because the code below uses a full image, two halfs of the image are concentrated.    
        if Abel != 'FLiPPID':
            R_red = np.concatenate((np.flip(R_red_half[:,1:len(R_red_half[0])], axis=1), R_red_half), axis=1)
            R_grn = np.concatenate((np.flip(R_grn_half[:,1:len(R_grn_half[0])], axis=1), R_grn_half), axis=1)
            R_blu = np.concatenate((np.flip(R_blu_half[:,1:len(R_blu_half[0])], axis=1), R_blu_half), axis=1)      
        end = time.time()
            
        #------------------------------------------------------------------------------
        # Calculate colour ratios
        #------------------------------------------------------------------------------
        # Define threshold above which colour ratio will be calculated
        threshold_ratio_red = 1.2
        threshold_ratio_grn = 1.2
        threshold_ratio_blu = 1.2
        GoodRG = (R_grn > threshold_ratio_grn) & (R_red > threshold_ratio_red)
        GoodRB = (R_blu > threshold_ratio_blu) & (R_red > threshold_ratio_red)
        GoodBG = (R_blu > threshold_ratio_blu) & (R_grn > threshold_ratio_grn)
        
        R_RG = np.zeros((len(R_red),len(R_red[0])))
        R_RB = np.zeros((len(R_red),len(R_red[0])))
        R_BG = np.zeros((len(R_red),len(R_red[0])))
        
        R_RG[GoodRG] = np.array(R_red[GoodRG] / R_grn[GoodRG])
        R_RB[GoodRB] = np.array(R_red[GoodRB] / R_blu[GoodRB])
        R_BG[GoodBG] = np.array(R_blu[GoodBG] / R_grn[GoodBG])
        

        
        #------------------------------------------------------------------------------
        # Get the temperature lookup table and calculate the soot temperature
        #------------------------------------------------------------------------------
        if get_temperature == True:
            # Define wavelength range in nm
            wavelength = [300, 700]
            lam = np.linspace(wavelength[0], wavelength[1], (wavelength[1]-wavelength[0])*10+1)
            # Define temperature range in K
            temperature = [1000,2200]
            T_calc = np.linspace(temperature[0], temperature[1], (temperature[1]-temperature[0])+1)
            # Fit camera response to measured values of colour ratios?
            fit =True 
            # What soot dispersion exponent (alpha) is used for soot is used? Possible are 'Chang' with lambda^-1.423 as derived from the refractive index 
            # measurements by:
            # Chang and Charalampopoulos, Proc. R. Soc. Lond. A 430 (1990) 577-591
            # or 'Kuhn' for lambda^-1.38 as reported by Kuhn et al. for the same raw data:
            # Kuhn et al., Proceedings of the Combustion Institute 33 (2011) 743-750
            soot_coef = 'Kuhn'
            
            # Provide location of the colour ratio calibration file. Column 0 has to contain the measure temperature in K and
            # column 1, 2, 3 the corresponding RG, RB, and BG colour ratios. The other columns in the example file are not
            # required but show the exposure times and counts of the red, green, and blue channel. Dividing column 6 (green counts)
            # by column 4 (exposure time) and plotting the result over column 0 (temperature) leads to the power law function
            # as required for the soot volume fractions (see below).
            if filter == 'FGB7':
                Thermo_calib = pd.read_csv('Temperature_tables/Measured_R_Thermo_FGB7_mod.csv', delimiter=',', header=None).values
            elif filter == 'FGB39':
                Thermo_calib = pd.read_csv('Temperature_tables/Measured_S_Thermo_FGB39.csv', delimiter=',', header=None).values
            else:
                sys.exit("No calibration data for the selected camera filter found. Program terminated.")
                
            """
            This function calculates the temperature lookup table, i.e., a table of the colour ratios expected to be recorded by the camera 
            as a function of the soot temperature. The function uses the theoretical camera, lens, and filter response to calculate colour 
            ratios of a hot thermocouple and compares it to experimentally measured values. The blue and red colour channels are scaled to 
            match the theoretical and observed colour ratios after which the ratios of hot soot are calculated. Further details can be found
            in the following publications:
            Ma and Long, Proceedings of the Combustion Institute 34 (2013) 3531-3539
            Kuhn et al., Proceedings of the Combustion Institute 33 (2011) 743-750
            """     
            from Temperature_lookup import Temperature_lookup
            Ratio_tables = Temperature_lookup(filter, lam, T_calc, fit, Thermo_calib, soot_coef)
            #filename_lookup=abspath( "H:\Python_Code\Cambridge_V2\Temperature_tables\Ratio_tables_0201223.csv")
            #Ratio_tables = pd.read_csv(filename_lookup, delimiter=';', header=None).values
            """
            This function uses the temperature lookup table to calculate the soot temperature profiles of the flame.  
            # """     
        
            from Get_flame_temperature import Get_flame_temperature
            from Get_flame_temperature import Get_flame_temperature
            [T_RG, T_RB, T_BG] = Get_flame_temperature(Ratio_tables, R_RB, R_RG, R_BG)
            
            # If one of the three light intensities is below pre-defined threshold, the temperature is set to zero.
            Good_T = (R_grn > threshold_ratio_grn) & (R_red > threshold_ratio_red) & (R_blu > threshold_ratio_blu)
            
            # The average temperature obtained from the three colour ratios is calculated. 
            #T_ave = (T_RG + T_RB+T_BG) / 3 * Good_T  #T_BG
            #T_ave=T_RB*Good_T
            T_RG = T_RG * GoodRG
            T_RB = T_RB * GoodRB
            T_BG = T_BG * GoodBG
            T_ave = (T_RB+T_RG+T_BG) / 3 * Good_T  #T_BG , 
            # plt.figure(figsize=(20,8),dpi=1200)

        
        #------------------------------------------------------------------------------
        # Get soot volume fraction
        #------------------------------------------------------------------------------
        """
        This function calculates the soot volume fraction using the previously calculated soot temperatures, the recorded light intensity, 
        the exposure time used while imaging the flame, and the camera calibration. Further details can be found in the following 
        publications:
        Ma and Long, Proceedings of the Combustion Institute 34 (2013) 3531-3539
        Kuhn et al., Proceedings of the Combustion Institute 33 (2011) 743-750  
        """    
        if get_volume_fraction == True and get_temperature == True:
            
            # mm for each pixel and exposure time while taking the flame images
            pixelmm = 0.026 #0.03 #1/50
        
            
            
            # Measured green singal emitted from a hot thermocouple devided by the expsoure time as a function of 
            # temperature. The equation is obtained by plotting the green counts / exposure time over temperature and 
            # fitting a power law function to it.
            if filter=='FGB7':
                def S_thermo(T):
                  #return (1.41928E-47*T**(1.50268E+01)) #original
                  return (1.41928E-47*T**(1.50268E+01))/2 #original
   
            elif filter=='FGB39':
                def S_thermo(T):
                    return 3.53219E-54*T**(1.72570E+01)
           
            from Get_soot_volume import Get_soot_volume
            #from Get_soot_volume_yale_mod import Get_soot_volume_yale
            f_ave = Get_soot_volume(filter, lam, pixelmm, exposure, T_ave, R_red, R_grn, R_blu, Ratio_tables, S_thermo)
            #f_ave[f_ave>100]=0
            
            print("f_v_max=%.10f"%np.max(f_ave))
            print(np.max(f_ave))
            # plt.figure(dpi=1200)
           
            # plt.subplot(121)
            # plt.imshow(T_ave,vmin=900, vmax=2500)
            # plt.colorbar()
            # plt.title('Average soot temperature [K]')
            # plt.subplot(122)
            # plt.imshow(f_ave, vmin=0, vmax=0.2) #vmin=0.05, vmax=20
            # plt.colorbar()
            # plt.title('Soot volume fraction [ppm]')
         
            # plt.show()
            
            test=f_ave
            test[test<np.percentile(test[test>0],90)]=0
        
            f_v_max_mean=np.mean(test[test>0])
            print("f_v_mean_max=%.10f"%f_v_max_mean)
            
            fv_array[check_num] = f_v_max_mean
            print(fv_array)
            check_num=check_num+1
            print(check_num, ': Iteration number')
            
            CpImGrn=ImGrn
            CpImGrn[CpImGrn<np.percentile(CpImGrn[CpImGrn>0],50)]=0
            print("ImGrn_mean=%.10f"%np.mean(CpImGrn[CpImGrn>0]))
            # plt.figure(dpi=1200)
            # plt.subplot(121)
            # plt.imshow(CpImGrn, cmap='twilight')
            # plt.colorbar()
            # plt.title('Green Intensity Values')
            # plt.subplot(122)
            # plt.imshow(test,vmin=0.01, vmax=0.05,cmap='twilight') #vmin=0.05, vmax=20
            # plt.colorbar()
            # plt.title('Soot volume fraction [ppm]')
         
            #plt.show()  #saves time if not showed
        # If True save the soot temperature and volume fraction. If FLiPPID was used the optimised fitting parameters are also saved.
        if save==True:    
            np.savetxt(ImPathT, T_ave ,delimiter=',')
            np.savetxt(ImPathfv, f_ave ,delimiter=',')
            
            if Abel == 'FLiPPID':
                fit_save = np.concatenate((fit_out[:,:,0], fit_out[:,:,1], fit_out[:,:,2]), axis=1)
                np.savetxt(ImPathfit, fit_save ,delimiter=',')
        
            
        Abel_time = float("{0:.2f}".format(end-start))    
        print('{0}{1}{2}'.format('Abel inversion required', str(Abel_time), 's'))
        
        from Get_YSI import Get_YSI 
        F_z_max= Get_YSI(f_ave,pixelmm,flame_width)
        
        print("F_z_max=%.10f"%F_z_max)
        # hf = plt.figure(dpi=1200)
        # ha = hf.add_subplot(111, projection='3d')
        # x=np.arange(0,flame_height)
        # y=np.arange(0,flame_width)
        # X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
        # from matplotlib import cm
        # ha.plot_surface(X.T, Y.T,f_ave,cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.show()
        # hh = plt.figure(dpi=1200)
        # hb = hh.add_subplot(111, projection='3d')
        # hb.plot_surface(X.T, Y.T,test,cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.show()
        # hh = plt.figure(dpi=1200)
        # hc = hh.add_subplot(111, projection='3d')
        # hc.plot_surface(X.T, Y.T,CpImGrn,cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.show()
        Results=np.transpose(np.array([F_z_max,f_v_max_mean,np.mean(CpImGrn[CpImGrn>0]),np.max(f_ave)]))

''' The extraction of the stored fv values and the filenames through the folder are converted into a Pandas Dataframe '''
###########################################################################
print(filename_list)
fv_mean_all = np.mean(fv_array[fv_array!=0])
fv_std=np.std(fv_array[fv_array!=0])
fv_relstd=fv_std*100/fv_mean_all
print(fv_mean_all)

filename_array_df = pd.DataFrame(filename_list, columns=['Filename'])
fv_array_df = pd.DataFrame(fv_array, columns=['FV_values'])
fv_mean_all_df = pd.DataFrame([fv_mean_all], columns=['Mean value'])
fv_std_all_df = pd.DataFrame([fv_std], columns=['Standard Abweichung'])
fv_relstd_all_df = pd.DataFrame([fv_relstd], columns=['Relative standard Abweichung'])

filename_array_df.reset_index(drop=True, inplace=True)
fv_array_df.reset_index(drop=True, inplace=True)
fv_mean_all_df.reset_index(drop=True, inplace=True)
fv_std_all_df.reset_index(drop=True, inplace=True)
fv_relstd_all_df.reset_index(drop=True, inplace=True)
###########################################################################

###########################################################################
''' Concatenation of the Pandas Dataframes into one datframe ''' 
dataframe = pd.concat([filename_array_df, fv_array_df, fv_mean_all_df, fv_std_all_df, fv_relstd_all_df], axis=1)
###########################################################################
# path = tkinter.filedialog.asksaveasfile(mode='w')

''' Opens a file dialog to ask the path, the name of the file and the file type '''
###########################################################################
path= tkinter.filedialog.asksaveasfilename(title = 'Save File', filetypes=(('Excel File', '.xlsx'),('Text File','.txt'),('All Files','*.*')))
print(path) 
path= path + '.xlsx'
###########################################################################

''' Converting to Excel '''
###########################################################################
dataframe.to_excel(path, sheet_name='Sheet 1')
###########################################################################