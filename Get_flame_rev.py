# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:24:52 2018

@author: jd766

The Code was modified by: Prithviraj Phadnis
The modifications include contour detection for the flame edge detection and automation of the processing of the files 
in a batch process.

At: OWI Science4Fuels gGmbH
Info: https://www.owi-aachen.de/


"""

def Get_flame(fname, filename, average, flame_height, HAB0, flame_width, plot_raw, thresh_tip, thresh_std, save_single):
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import io
    import os
    import tkinter as tk
    from os.path import abspath, exists
    from scipy import ndimage
    import cv2
    from scipy.signal import savgol_filter
    
    """
    If the automatic flame centre and edge detection fails check the below values, i.e., the thresholds for edge and tip detection, 
    the selection criteria for straight flames with similar height, and the number and positions where flame centres are searched
    """
    
    # Threshold for the tip detection
    thresh_grn  = 2000
    
    # It is possible to rotate slightly tilded flame images
    degree = 0
    
    # Max. intensity count of the camera
    scale = 65535
    
    # Read raw image in Bayer formate from
    #fname = abspath( '{0}{1}{2}'.format('Photos//Select//N6///', filename, '.tif') )
    # ImDatBayer_all=cv2.imread(fname, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    # ImDatBayer_all=np.array([ImDatBayer_all,ImDatBayer_all])
    ############# DIALOG BOXES#############################
    # filepath= tk.filedialog.askdirectory()
    # filepath= tk.filedialog.askopenfile()
    # print(filepath)
    # fname= str(filepath.name)
    # print(fname)
    
    ImDatBayer_all=io.imread(fname)
    ImDatBayer = np.zeros(shape=(average,len(ImDatBayer_all),len(ImDatBayer_all[0])), dtype=ImDatBayer_all.dtype)
    #print(ImDatBayer.shape)
    ImDat = np.zeros(shape= (len(ImDatBayer_all),len(ImDatBayer_all[0]), average, 3), dtype=ImDatBayer_all.dtype)
        #img = np.fromfile(fname, np.dtype('u2'), len(ImDatBayer[0])*len(ImDatBayer[0][0])).reshape((len(ImDatBayer[0]), len(ImDatBayer[0][0])))
        # Only take 'average' frames from the stack        
    ImDatBayer[0,:,:] = ImDatBayer_all[:,:]
    del ImDatBayer_all
        
    print('Finished loading image')
        
        # Demosaic the image. Different algorithms can be used: COLOR_BayerGB2RGB, COLOR_BayerGB2RGB_EA, COLOR_BayerGB2RGB_VNG. Note that 
        # COLOR_BayerRG2RGB_VNG requires a 8 bit image. In case the camera has a different Bayer pattern the 'BayerGR' has to be replaced by the appropiate
        # pattern. Actually, the BlackflyS used for the example image has a BayerRG pattern but the raw image was rotated by 90 degrees such that the flame is 
        # vertical.
    for i in range(0, average):
        # ImDat[:,:,i,:] = ndimage.rotate((cv2.cvtColor(ImDatBayer[i,:,:], cv2.COLOR_BayerRGGB2BGR_EA)), degree, reshape=False)
        ImDat[:,:,i,:] = cv2.cvtColor(ImDatBayer[i,:,:], cv2.COLOR_BayerRGGB2BGR_EA)

         
    del ImDatBayer  
       
    
    ImDatRed = ImDat[:,:,:,2]
    ImDatGrn = ImDat[:,:,:,1] 
    ImDatBlu = ImDat[:,:,:,0]  
          
    print('Finished demosaicing image')
    
    left = np.zeros([len(ImDatGrn[0][0]), len(ImDatGrn)], dtype=int)
    right = np.zeros([len(ImDatGrn[0][0]), len(ImDatGrn)], dtype=int)
    middle = np.zeros([len(ImDatGrn[0][0]), len(ImDatGrn)])
    middle_std = np.zeros([len(ImDatGrn[0][0])])

    #Changed Code start
    # Find left and right flame edge to calculate the flame centre at different pixel rows
    for j in range(0,len(ImDatGrn[0][0])):
        gray = cv2.cvtColor(np.uint8(ImDat[:,:,j,:]/scale*255), cv2.COLOR_BGR2GRAY)
        th_gray = cv2.adaptiveThreshold(gray,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,85,2)
    
        for i in range(0,len(ImDatGrn)):
            left[j,i] = np.argmax(th_gray[i,:]<0.5)
            right[j,i] = len(ImDatGrn[0]) - np.argmax(np.flip(th_gray[i,:],0)<0.5) - 1
            
        # Remove the noisy top and bottom part of the flame    
        left[j,0:np.argmax(left[j,:]>0)+20] = 0
        left[j,len(left[0]) - np.argmax(np.flip(left[j,:])>0)-50:-1] = 0
        
        middle[j,:] = (left[j,:] + right[j,:])/2 * (left[j,:]!=0) * (right[j,:]!=len(ImDatGrn[0])-1)
        middle_std[j] = np.std(middle[j,middle[j,:]!=0])
        

        # Remove the noisy top and bottom part of the flame    
        left[j,0:np.argmax(left[j,:]>0)+20] = 0
        left[j,len(left[0]) - np.argmax(np.flip(left[j,:])>0)-50:-1] = 0
        
        middle[j,:] = (left[j,:] + right[j,:])/2 * (left[j,:]!=0) * (right[j,:]!=len(ImDatGrn[0])-1)
        middle_std[j] = np.std(middle[j,middle[j,:]!=0])
    
    thresh_std=np.min(middle_std)+1

    
    tip = np.zeros(len(ImDatGrn[0][0]), dtype=int)
    
    # Find the flame tip at the flame centre
    for j in range(0,len(ImDatGrn[0][0])):
        tip[j] = np.argmax(ImDatGrn[:,int(round(np.mean(middle[j,np.nonzero(middle[j,:])]))),j]) #np.argmax(ImDatGrn[:,int(round(np.mean(middle[j,np.nonzero(middle[j,:])]))),j]>thresh_grn)
    
    print('Flame edge and tip detection finished')
    
    ave_tip = np.mean(tip)
    G_filter = 0
    ImDatRed_filter = np.array(np.zeros((len(ImDatRed),len(ImDatRed[0]),1)))
    ImDatGrn_filter = np.zeros((len(ImDatRed),len(ImDatRed[0]),1))
    ImDatBlu_filter = np.zeros((len(ImDatRed),len(ImDatRed[0]),1))
    middle_filter = np.zeros((1,len(middle[0])))
    
    # The non-tilted flames with similar height are selected.
    for j in range(0,len(ImDatGrn[0][0])):
        if (middle_std[j] < thresh_std) and (tip[j]+thresh_tip > ave_tip) and (tip[j]-thresh_tip < ave_tip):
            if G_filter == 0:
                ImDatRed_filter[:,:,G_filter] = ImDatRed[:,:,j]
                ImDatGrn_filter[:,:,G_filter] = ImDatGrn[:,:,j]
                ImDatBlu_filter[:,:,G_filter] = ImDatBlu[:,:,j]
                middle_filter[G_filter,:] = middle[j,:]
                
            else:    
                ImDatRed_filter = np.insert(ImDatRed_filter, 1, ImDatRed[:,:,j], axis=2)
                ImDatGrn_filter = np.insert(ImDatGrn_filter, 1, ImDatGrn[:,:,j], axis=2)
                ImDatBlu_filter = np.insert(ImDatBlu_filter, 1, ImDatBlu[:,:,j], axis=2)
                middle_filter = np.insert(middle_filter, 1, middle[j,:], axis=0)
                
            G_filter = G_filter+1
    
    print('{0}{1}{2}'.format('After the flame centre and edge detection, ', G_filter, ' frames remained and were averaged.'))
    ImRed_filter = np.mean(ImDatRed_filter, axis=2)
    ImGrn_filter = np.mean(ImDatGrn_filter, axis=2)
    ImBlu_filter = np.mean(ImDatBlu_filter, axis=2)
    
    ImDevRed_filter = np.std(ImDatRed_filter, axis=2)
    ImDevGrn_filter = np.std(ImDatGrn_filter, axis=2)
    ImDevBlu_filter = np.std(ImDatBlu_filter, axis=2)
     
    if len(ImDatGrn[0][0]) > 1:
        print('passed 1')
        # plt.figure()
        # plt.title('Standard deviation of detected centres in each frame and the deviation threshold')
        # plt.plot(middle_std, 'r')
        # plt.plot(np.ones(len(ImDatGrn[0][0]))*thresh_std, 'b')
        # #set_ylim([0,thresh_std*1.2])
        # plt.show()
        
        # plt.figure()
        # plt.title('Flame tip in each frame and range of tip threshold')
        # plt.plot(tip, 'xr')
        # plt.plot(np.ones(len(ImDatGrn[0][0]))*(ave_tip+thresh_tip),'g--')
        # plt.plot(np.ones(len(ImDatGrn[0][0]))*(ave_tip-thresh_tip),'g--')
        # plt.show()
    
    middle_crop = int(round(np.mean(np.mean(middle_filter[middle_filter!=0]))))
    
    for l in range(0,len(ImDatRed_filter[0][0])):
        # Calculate single and averaged flame images.
        ImRed_single = ImDatRed_filter[:,:,l]
        ImGrn_single = ImDatGrn_filter[:,:,l]
        ImBlu_single = ImDatBlu_filter[:,:,l]
        
        # Crop the flame images to the desired size.
        ImRed_single_crop = ImRed_single[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
        ImGrn_single_crop = ImGrn_single[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
        ImBlu_single_crop = ImBlu_single[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
        

    
    Cropped_ImRed = ImRed_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    Cropped_ImGrn = ImGrn_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    Cropped_ImBlu = ImBlu_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    
    Cropped_ImDevRed = ImDevRed_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    Cropped_ImDevGrn = ImDevGrn_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
    Cropped_ImDevBlu = ImDevBlu_filter[HAB0-flame_height:HAB0, int(middle_crop-np.round(flame_width/2)):int(middle_crop+np.round(flame_width/2))+1]
     
    ImRed = Cropped_ImRed
    ImGrn = Cropped_ImGrn
    ImBlu = Cropped_ImBlu
    
    ImDevRed = Cropped_ImDevRed
    ImDevGrn = Cropped_ImDevGrn
    ImDevBlu = Cropped_ImDevBlu
    
    middle_line = np.zeros(2)
    left_plot = (np.mean(left, axis=0)-(middle_crop-np.floor(flame_width/2)))*np.min(left!=0,axis=0)*np.min(right!=len(ImDatGrn[0])-1,axis=0)
    right_plot = (np.mean(right, axis=0)-(middle_crop-np.floor(flame_width/2)))*np.min(right!=len(ImDatGrn[0])-1,axis=0)*np.min(left!=0,axis=0)
    middle_plot = (np.mean(middle_filter, axis=0)-(middle_crop-np.floor(flame_width/2)))*np.min(right!=len(ImDatGrn[0])-1,axis=0)*np.min(left!=0,axis=0)
    
    row_plot = np.linspace(0,len(ImDatGrn),len(ImDatGrn))-(HAB0-flame_height)
    
    middle_line[0] = middle_crop-(middle_crop-(flame_width/2))
    middle_line[1] = middle_line[0]
    

    if plot_raw==True:
        print('passed 2')
        plt.figure()
        plt.subplot(131)
        plt.imshow(ImRed, vmin=0, vmax=scale)
        plt.title('Original red channel intensity')
        plt.subplot(132)
        plt.imshow(ImGrn, vmin=0, vmax=scale)
        plt.title('Original green channel intensity and detected centre line')
        plt.plot(left_plot[left_plot!=0],row_plot[left_plot!=0],'r')
        plt.plot(right_plot[left_plot!=0],row_plot[left_plot!=0],'r')
        plt.plot(middle_plot[left_plot!=0],row_plot[left_plot!=0],'xy')
        #plt.plot(middle_line,[row_plot[np.argmax(left_plot>0)]-50, row_plot[len(left_plot) - np.argmax(np.flip(left_plot)>0)]+50],'r')
        plt.subplot(133)
        plt.imshow(ImBlu, vmin=0, vmax=scale)
        plt.title('Original blue channel intensity')
        plt.show()
        
        plt.figure()
        plt.plot(Cropped_ImGrn[300,:], 'r')
        plt.plot(Cropped_ImGrn[450,:], 'r')
        plt.plot(Cropped_ImGrn[660,:], 'r')
        plt.plot(ImGrn_single_crop[300,:], 'g')
        plt.plot(ImGrn_single_crop[450,:], 'g')
        plt.plot(ImGrn_single_crop[660,:], 'g')
        plt.title('Comparision single frame to averaged frames')
        plt.show()

        

    return (ImRed, ImGrn, ImBlu, ImDevRed, ImDevGrn, ImDevBlu)