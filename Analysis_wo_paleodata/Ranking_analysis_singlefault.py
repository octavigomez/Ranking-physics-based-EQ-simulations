#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This code analyzes RSQSim models and their performance based on a series of objective functions..
The code has been developed during the postdoctoral project of Octavi Gómez-Novell

Authors: Octavi Gómez-Novell, Bruno Pace, Francesco Visini, José Antonio Álvarez-Gómez
Involved institutions: Università di Chieti-Pescara, Universitat de Barcelona, Istituto Nazionale di Geofisica e Vulcanologia, Universidad Complutense de Madrid

Location: Chieti, Italy
July-September 2023

"""
import os
import json
import math
import natsort
import numpy as np
import pandas as pd
#import re
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
import matplotlib as mpl
from scipy.stats import poisson
import sys

mpl.rcParams['agg.path.chunksize'] = 100000
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['mathtext.fontset'] = 'cm'

#Define path of fault model and read fault model and surface patches
path_now = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_now)
path = path_now + '/Inputs'

#Fault model
file = os.path.join(path, "Nodes_RSQSim.flt")

#Weights of the respective benchmarks in the objective function
w_AM = 1 #AM relationship benchmark
w_AD = 1 #AD relationship benchmark
w_MFD = 1 #MFD benchmark

# =============================================================================
# Start
# =============================================================================

input_file = pd.read_csv(file, delimiter =" ", header=None)
n=1

x1 = input_file.iloc[:,0]*n
x2 = input_file.iloc[:,3]*n
x3 = input_file.iloc[:,6]*n
y1 = input_file.iloc[:,1]*n
y2 = input_file.iloc[:,4]*n
y3 = input_file.iloc[:,7]*n
z1 = input_file.iloc[:,2]*n
z2 = input_file.iloc[:,5]*n
z3 = input_file.iloc[:,8]*n

file = np.column_stack((x1,x2,x3,y1,y2,y3,z1,z2,z3, np.array(range(1, len(input_file)+1))))

x_center = (file[:,0]+file[:,1]+file[:,2])/3
y_center = (file[:,3]+file[:,4]+file[:,5])/3
z_center = (file[:,6]+file[:,7]+file[:,8])/3

#Area of triangles

init_depth = 0
con = (file[:,6]>=init_depth) | (file[:,7]>=init_depth) | (file[:,8]>=init_depth)
con = np.where(con)[0]
max_depth = max(z_center[con])
min_depth = min(z_center[con])
max_z_center = np.where((z_center>=min_depth) & (z_center<=max_depth))[0]
max_depth = np.min(z_center[max_z_center])

#Find patches that are located between 0 and mean_depth km

z_1 = np.where(file[max_z_center,6:9]>max_depth)[0]
z_filter, reps = np.unique(z_1, return_index = False, return_counts = True)
z_2 = np.where(reps > 1)[0]
z_0 = np.where(reps == 1)[0]
z_filter = z_filter[z_2]
file_surf = file[z_filter,:]
tri_area = []
tri_height = []
dip_triangle = []
tri_depth = []
tri_base = []
AB_locs = []
for ar in np.unique(z_1):
    AB = [x2[ar]-x1[ar], y2[ar]-y1[ar], z2[ar]-z1[ar]]
    AC = [x3[ar]-x1[ar], y3[ar]-y1[ar], z3[ar]-z1[ar]]
    AA = [x3[ar]-x2[ar], y3[ar]-y2[ar], z3[ar]-z2[ar]]
    
    l1 = math.sqrt(AB[0]**2+AB[1]**2+AB[2]**2)
    l2 = math.sqrt(AC[0]**2+AC[1]**2+AC[2]**2)
    l3 = math.sqrt(AA[0]**2+AA[1]**2+AA[2]**2)
    
    # Calculate the area
    semi_perimeter = (l1+l2+l3)/2
    
    # Heron's formula
    area = math.sqrt(semi_perimeter*(semi_perimeter-l1)*(semi_perimeter-l2)*(semi_perimeter-l3))
    height = 2*area/l1
    tri_area.append(area)
    tri_base.append(l1)
    tri_height.append(height)

AB_locs = np.array(AB_locs)
locs_AB = np.where(AB_locs==0)[0]
tri_base = np.array([int(tr) for tr in tri_base])
z_filter_patches = max_z_center
surface_patches_SRL = file[z_filter_patches, -1]
surface_patches_effective = file[z_filter,-1]
x_surface_patch = x_center[z_filter_patches]
y_surface_patch = y_center[z_filter_patches]
z_surface_patch = z_center[z_filter_patches]
coords_patch = np.column_stack((x_surface_patch, y_surface_patch, z_surface_patch))
coords_surface_SRL = file[np.isin(file[:,-1], surface_patches_SRL)]
coords_sorted_SRL = file[file[:,0].argsort()]
SRL_all_tris = round(sum(tri_base[z_2]),0)

#Define path where the models that you want to evaluate are located

path =  path_now + '/Simulation_models/'
c = -1
it = -1
Sum_lik_model = []
A_param = []
B_param = []
B_A_param = []
Sigma_param = []
b_rating = []
M_surf_mean = []
Sum_lik_model_AD = []
Sum_lik_model_SRL = []
idx_empty = []
idx_full = []
Mc_list = []
b_values = []
Sum_lik_model_paleo = []
processed_folder = []

#start_folder = path

start_processing = False

for folder_name in natsort.natsorted(os.listdir(path)):
    c=c+1
    folder = os.path.join(path, folder_name)
    if not os.path.isdir(folder):
        continue
    print(folder)
    
    # if not start_processing:
    #     if folder == start_folder:
    #         start_processing = True
    #     else:
    #         continue
    
    processed_folder.append(folder)
    file = os.path.join(folder, "data.json")
    it = it+1
    with open(file, "r") as file:
        catalog = json.load(file)
        
    #Extract parameters from catalogue and define magnitude range. Exlude first 2000 events of catalog
    if len(list(catalog.values())[2])>2000:
        M0 = list(catalog.values())[2][2000:]
        M = list(catalog.values())[3][2000:]
        x = list(catalog.values())[4][2000:]
        y = list(catalog.values())[5][2000:]
        z = list(catalog.values())[6][2000:]
        A = list(catalog.values())[7][2000:]
        t0 = list(catalog.values())[-2][2000:]
        parameters = list(catalog.values())[0]
        A_1 = list(parameters.values())[1][0]
        B_1 = list(parameters.values())[3][0]
        sigma0_1 = list(parameters.values())[11][0]
        
# =============================================================================
#         #If A, B and sigma are depth/strike variable, activate these instead of previous ones
#         
#         sigma0_1 = parameters.get("initSigmaFname")[0]
#         A_1 = parameters.get("AFname")[0]
#         B_1 = parameters.get("BFname")[0]
#         A_1 = float(re.findall('\d+\.\d+', A_1)[0])
#         B_1 = float(re.findall('\d+\.\d+', B_1)[0])
#         sigma0_1 = float(re.findall(r'\d+', sigma0_1)[0])
# =============================================================================
        
        B_A = A_1-B_1
        A_param.append(A_1)
        B_param.append(B_1)
        B_A_param.append(B_A)
        Sigma_param.append(sigma0_1)
        M_range = list(np.arange(-18, 18, 0.01))     
        eList = catalog.get("eList")
        ev, ix = np.unique(eList, return_index=True)
        ix = ix[2000]
        eList = eList[ix:]
        pList = catalog.get("pList")[ix:]
        dList = catalog.get("dList")[ix:]
        num_events = list(range(2001, max(eList)+1))
        uniq_patches, p_idx, cts = np.unique(sorted(pList), return_index=True, return_counts = True)

        #Calculate completeness (best Combo) and filter catalog by the completeness threshold.
        
        mrange = np.arange(-2, 8.1, 0.1)
        mrange = np.round(mrange,2)
        hist, bins = np.histogram(M, bins = mrange)
        # idx = np.argmax(hist)
        binInterval = 0.1
        half_bin = binInterval/2
        
        def maxCurvature(M, mrange):
            vEvents, bins = np.histogram(M, mrange)
            idx = np.argmax(vEvents)
            Mc_ini = mrange[idx]
            return Mc_ini
        
        
        def McCalc(M, binInterval):
            Mc_ini = maxCurvature(M, mrange)
            range_Mc = np.arange(Mc_ini, round(max(M)), binInterval)

            b_all = []
            for i in range_Mc:
                McM_id = [x>i for x in M]
                M_Mc = np.array(M)[McM_id]
                mean_M = np.mean(M_Mc)
                b_all.append((1/(mean_M + binInterval*0.5-i)) * np.log10(np.exp(1)))
            
            #Calculate variation between consecutive numbers
            
            diff_all = []
            idx_Mcs = []
            for ii in range(len(b_all)-1):
                diff_perc = abs((b_all[ii+1]-b_all[ii]))/b_all[ii]
                diff_all.append(diff_perc)
                if diff_perc <= 0.05:
                    idx_Mcs.append(ii)
                if diff_perc <= 0.1:
                    idx_Mcs.append(ii)
                elif diff_perc <= 0.15:
                    idx_Mcs.append(ii)
                elif diff_perc <= 0.25:
                    idx_Mcs.append(ii)
                elif diff_perc <= 0.3:
                    idx_Mcs.append(ii)
                elif diff_perc <= 0.35:
                    idx_Mcs.append(ii)  
                elif diff_perc <= 0.4:
                    idx_Mcs.append(ii)  

            if len(idx_Mcs)>0:
                Mc =range_Mc[min(idx_Mcs)]
            else:
                Mc = Mc_ini #Stick to Max. Curvature
            
            return Mc
        
        Mc = maxCurvature(M, mrange)
        print("Mc=" + str(Mc))
        Mc_list.append(Mc)
       # plt.scatter(Mc, B_A, c="black")


          
        # Remove events below the Mc from the catalogue
        
        idx_Mc = [x>Mc for x in M]
        M0 = (np.array(M0))[idx_Mc]
        M = (np.array(M))[idx_Mc]
        x = (np.array(x))[idx_Mc]
        y = (np.array(y))[idx_Mc]
        z = (np.array(z))[idx_Mc]
        A = (np.array(A))[idx_Mc]
        num_events = (np.array(num_events))[idx_Mc]
        Moment_release = np.column_stack ((M, M0, A, num_events))        
        idx_Mc_all = np.where(np.isin(eList, num_events))[0]
        eList = np.float64((np.array(eList))[idx_Mc_all])
        pList = np.float64((np.array(pList))[idx_Mc_all])
        dList = np.float64((np.array(dList))[idx_Mc_all])
       # dtauList = np.float64((np.array(dtauList))[idx_Mc_all])
       # tList = np.float64((np.array(tList))[idx_Mc_all])       
        Event_list = np.column_stack((eList, pList, dList))#, dtauList, tList))

        if len(M)>0:
            idx_full.append(it)
            Mmin = min(M)
            Mmax = max(M)
            
            

           
# =============================================================================
#             # Rupture area and magnitude scaling relationship by Leonard (2010)
# =============================================================================
            
            # Leonard 2010 for dip-slip faults (DS).
            a = 1.5
            b = 6.1
            sd_b = 0.45 #b 1 sigma range is 5.69-6.60
            
            # Round values to 1 decimal to simplify further comparisons between scaling law and model.
            
            #M_round = [round(elem,2) for elem in M]
            Mmin = min(M)
            Mmax = max(M)
            M_range = list(np.arange(0, 100,0.01))
            M_range = [round(elem, 2) for elem in M_range]
            M_round = [round(np.log10(elem),2) for elem in M0]
    
            # Convert scaling laws into log likelihoods and find intersection with magnitudes computed by RSQSim.
            
            Mw = []
            sd_Mw = []
            M_fit = []
            Loglik_model = []
            c1=-1
            for i in A:
                c1 = c1+1
                logM0 = a*math.log10(i)+b
                sd_logM0 = sd_b
                Mg = logM0#(2/3)*logM0-6.07
                Mw.append(Mg)
                sd_Mg = sd_logM0#2/3*sd_logM0
                sd_Mw.append(sd_Mg)
                M_pdf = stats.norm.pdf(M_range, Mg, sd_Mg)
                #M_pdf = np.exp(M_pdf)
                Norm_pdf = M_pdf/max(M_pdf) #Normalization  
               #Norm_pdf = [np.float('{:.10000000e}'.format(nd)) for nd in Norm_pdf]
                loc = M_range.index(M_round[c1])              
                #Log_lik = np.log(Norm_pdf)             
                Lik_model = Norm_pdf[loc]
                Loglik_model.append(np.log(Lik_model))               
            Sum_lik = np.sum(Loglik_model) # Sumatory of all log-likelihoods 
            Sum_lik_model.append(Sum_lik/len(M0))
            
# =============================================================================
#             #B value
# =============================================================================

            #Theoretical     
        
            b_ref = 1.0
            locs_Mc = np.where(bins==np.round(Mc,1))[0]
            locs_max = np.where(bins == np.round(8,1))[0]
            num_eq = hist[locs_Mc[0]:locs_max[0]]
            a_value = np.log10(np.max(np.cumsum(num_eq)))+b_ref*Mc
            N_eq_GR = 10**(a_value-b_ref*bins[locs_Mc[0]:locs_max[0]+1]) 
           # N_eq_GR2 = np.concatenate((N_eq_GR, [N_max]))
            N_eq_noncum = abs(np.diff(N_eq_GR))
            N_eq_noncum = N_eq_noncum[0:len(num_eq)]
            N_eq_cum = np.flip(np.cumsum(N_eq_noncum[::-1]))
            n_eq_cum = np.flip(np.cumsum(num_eq[::-1]))
            #Norm_cat = [(a-min(n_eq_cum))/(max(n_eq_cum)-min(n_eq_cum)) for a in n_eq_cum]
           # Norm_b = [(a-min(N_eq_cum))/(max(N_eq_cum)-min(N_eq_cum)) for a in N_eq_cum]
            #Log_lik_mfd = [poisson.logpmf(round(cat*100), round(ref*100)) for cat, ref in zip(Norm_cat, Norm_b)]
            #Log_lik_mfd = [a/np.log(10) for a in Log_lik_mfd]
            Log_lik_mfd = [poisson.logpmf(round(cat,2), round(ref,2)) for cat, ref in zip(num_eq, N_eq_noncum)]
            Log_lik_mfd = [a/np.log(10) for a in Log_lik_mfd]
            #inf_vals = ~np.isinf(Log_lik_mfd)
            #Log_lik_mfd = Log_lik_mfd/min(Log_lik_mfd)
            Sum_lik_mfd = sum(np.array(Log_lik_mfd))/sum(num_eq)
            b_rating.append(Sum_lik_mfd)
            
            # print(Log_lik_mfd)
            # print(Sum_lik_mfd)
               
            def b_value(M):
                b = - (1 / (np.mean(M) + binInterval * 0.5 - np.min(M))) * np.log10(np.exp(1))
                return b
            b = b_value(M)

            
            # plt.figure(dpi=600).add_subplot()
            # plt.scatter(np.flip(mrange[locs_Mc[0]:]), np.log10(n_eq_cum))
            # plt.plot([Mc, Mc],[0, (np.log10(max(n_eq_cum)))], c='red')
            # plt.text(7,np.log10(max(n_eq_cum)), "bvalue="+str(round(b,2)))
            # plt.xlim(3, bins[-1])
            # plt.show()

# =============================================================================
#             # Average displacement and magnitude by Leonard (2010)
# =============================================================================

            #Find events in the catalogue that affect surface patches
            
            events_surface = Event_list[np.isin(Event_list[:,1], surface_patches_SRL)]
            events_surface[:,-1] /= (365*24*3600)       
            uni_ev, idx_uni = np.unique(events_surface[:,0], return_inverse =True)
            idx_M = np.where(np.isin(eList, uni_ev))[0]
            Surf = Moment_release[np.isin(Moment_release[:,-1], uni_ev)]
            M_surface = Surf[:,0]
            M_surf_mean.append(np.mean(M_surface))
            Slip_eve_surface = Surf[:,1]
            SRL_eq = []
            #fig = plt.figure()
            # ba = fig.add_subplot(111, projection='3d')
            c=-1
            for sfs in uni_ev:
                c=c+1
                time_of_event = t0[idx_M[c]]
                event_at_surface = events_surface[np.isin(events_surface[:,0], sfs)]
                slip_patch =  event_at_surface[:,2]
                patch_surf = event_at_surface[:,1]
                sf = surface_patches_SRL[z_0][np.isin(surface_patches_SRL[z_0], patch_surf)]
                sf_Ef = surface_patches_SRL[z_2][np.isin(surface_patches_SRL[z_2], patch_surf)]
                idx_sf = np.where(np.isin(surface_patches_SRL, sf))[0]
                idx_sf_Ef = np.where(np.isin(surface_patches_SRL, sf_Ef))[0]                
                all_idx = np.unique(np.concatenate((idx_sf,idx_sf_Ef)))              
                coords_ev = coords_patch[all_idx,:]
                sort_idx = np.argsort(coords_ev[:,0])
                coords_ev = coords_ev[sort_idx,:] 
                single_coord = coords_surface_SRL[all_idx]
                increment = [] 
                if (len(sf)==1) and (len(sf_Ef)==0):
                    SRLi = round(np.sum(tri_base[idx_sf]),0)
                elif len(sf_Ef)==0:
                    SRLi = round(np.sum(tri_base[idx_sf]))
                elif (len(sf)+len(sf_Ef))==2:
                    SRLi = round(np.sum(tri_base[all_idx]),0)
                else:
                    SRLi = round(np.sum(tri_base[idx_sf_Ef]),0)    
                SRL_eq.append(SRLi)
                #Debug
                
                if SRLi ==0:
                    print(c)
                    print(sf)
                    print(sf_Ef)
                    print(patch_surf)
                
                # if len(coords_ev)>1:
                #     for sfz in range(len(event_at_surface[:,1])-1):                    
                #         increment.append(math.sqrt((coords_ev[sfz,0] - coords_ev[sfz+1,0]) ** 2 + (coords_ev[sfz,1] - coords_ev[sfz+1,1]) ** 2 + (coords_ev[sfz,2] - coords_ev[sfz+1,2]) ** 2))                  
                # elif len(coords_ev)==1:
                #     min_coord = np.min(single_coord[:,0:3], axis=0)
                #     min_coord = np.min(min_coord)
                #     max_coord = np.max(single_coord[:,0:3], axis=0)
                #     max_coord = np.max(max_coord)
                #     increment.append(abs(max_coord-min_coord))    
                
                    
            #     ba.plot(coords_ev[:,0], coords_ev[:,1], slip_patch[sort_idx]+time_of_event)
            #     ba.scatter(coords_patch[:,0], coords_patch[:,1], np.zeros(len(coords_patch[:,1])))
            #     ba.view_init(elev=30., azim=-90)
            #     #ba.plot(coords_ev[:,0], slip_patch[sort_idx]+time_of_event)
            #     plt.xlabel("Lon")
            #     plt.ylabel ("Lat")
            # plt.show()
               
            event_dict = {}
            for row in Event_list:
                key = row[0]
                value = row[2]
                if key in event_dict:
                    event_dict[key].append(value)
                else:
                    event_dict[key] = [value] 
            events_AD = {key: round(np.mean(values), 2) for key, values in event_dict.items()}
            events_AD = list(events_AD.values())
            events_AD = [round(np.log10(ad),2) for ad in events_AD]
             
            a_AD = 0.5
            b_AD = -4.42
            sd_b_AD = (4.82-3.92)/2

            AD_range = list(np.arange(-10, 10, 0.01))
            AD_range = [round(ad,2) for ad in AD_range]
            Loglik_model_AD = []
            c1 = -1
            AD_all = []
            sd_all = []
            for i in A:
                c1 = c1+1
                logAD = a_AD*math.log10(i)+b_AD
                sd_logAD = sd_b_AD
                AD = logAD#round(10**(logAD),2)
                sd_AD = sd_logAD#round(AD*np.log(10)*sd_logAD,2)
                AD_all.append(AD)
                sd_all.append(sd_AD)
                AD_pdf = stats.norm.pdf(AD_range, AD, sd_AD)    
                #AD_pdf= np.exp(AD_pdf)
                Norm_pdf_AD = AD_pdf/max(AD_pdf) #Normalization   
                #Log_lik_AD = np.log(Norm_pdf_AD)
                loc_AD = AD_range.index(events_AD[c1])
                Lik_model_AD = Norm_pdf_AD[loc_AD]
                Loglik_model_AD.append(np.log(Lik_model_AD))
            Sum_lik_AD = np.sum(Loglik_model_AD) # Sumatory of all log-likelihoods 
            Sum_lik_model_AD.append(Sum_lik_AD/len(events_AD))
            
        else:
            print("This catalog is empty")
            idx_empty.append(it)
        
    else:
        print("This catalog is empty")
        idx_empty.append(it)
        
        
# =============================================================================
# #Prepare variables for the objective function
# =============================================================================


Sum_lik_model_perc = [((n-max(Sum_lik_model))/(min(Sum_lik_model)-max(Sum_lik_model)))/2 for n in Sum_lik_model]
Sum_lik_model_AD_perc = [((n-max(Sum_lik_model_AD))/(min(Sum_lik_model_AD)-max(Sum_lik_model_AD)))/2 for n in Sum_lik_model_AD]
b_rating_perc =[(n-max(b_rating))/(min(b_rating)-max(b_rating)) for n in b_rating]


Norm_lik_AM = Sum_lik_model_perc
Norm_b_rating = b_rating_perc 
Norm_lik_AD = Sum_lik_model_AD_perc
Combined_log_lik = [l1+l2 for l1, l2 in zip(Norm_lik_AM, Norm_lik_AD)]

#Objective function

obj_factor = [(fr*w_AM + sc*w_MFD + sa*w_AD) for fr, sc, sa in zip(Norm_lik_AM, Norm_b_rating, Norm_lik_AD)]#, Norm_lik_paleo)]
normalized_obj = [(obj-min(obj_factor))/(max(obj_factor)-min(obj_factor)) for obj in obj_factor]

#Tags to plot models + names side by side

tags = np.array(range(1, len(Sum_lik_model)+1))
tags = [str(s) for s in tags]
models = np.column_stack((tags, normalized_obj))
models_AM = np.column_stack((tags, Norm_lik_AM))

models_AD = np.column_stack((tags, Norm_lik_AD))
models_brating = np.column_stack((tags, Norm_b_rating))
models_Mc = np.column_stack((tags, Mc_list))

order_idx = np.argsort(models[:,1])
ordered_models = models[order_idx,0]
score_models = models[order_idx,1]

order_idx_AM = np.argsort(models_AM[:,1])
ordered_models_AM = models_AM[order_idx_AM,0]
score_models_AM = models_AM[order_idx_AM,1]

order_idx_AD = np.argsort(models_AD[:,1])
ordered_models_AD = models_AD[order_idx_AD,0]
score_models_AD = models_AD[order_idx_AD,1]

order_idx_brating = np.argsort(models_brating[:,1])
ordered_models_brating = models_brating[order_idx_brating,0]
score_models_brating = models_brating[order_idx_brating,1]

order_idx_Mc = np.argsort(models_Mc[:,1])
ordered_models_Mc = np.array(Mc_list)[order_idx_Mc]

print("The best models are (ordered from best to worst): " + str(ordered_models))

models_all = [str(Sigma_param[int(a)-1]) + "_" + str(A_param[int(a)-1]) + "_" + str(B_param[int(a)-1]) for a in ordered_models]
A_ordered = [A_param[int(a)-1] for a in ordered_models]
B_ordered = [B_param[int(a)-1] for a in ordered_models]
A_B_ordered = [A_param[int(a)-1]-B_param[int(a)-1] for a in ordered_models]

Sigma_ordered = [Sigma_param[int(a)-1] for a in ordered_models]
Sigma_ordered_b = [Sigma_param[int(a)-1] for a in ordered_models_brating]

A_ordered_AM = [A_param[int(a)-1] for a in ordered_models_AM]
B_ordered_AM = [B_param[int(a)-1] for a in ordered_models_AM]
A_B_ordered_AM = [A_param[int(a)-1]-B_param[int(a)-1] for a in ordered_models_AM]

A_ordered_AD = [A_param[int(a)-1] for a in ordered_models_AD]
B_ordered_AD = [B_param[int(a)-1] for a in ordered_models_AD]
A_B_ordered_AD = [A_param[int(a)-1]-B_param[int(a)-1] for a in ordered_models_AD]

A_ordered_brating = [A_param[int(a)-1] for a in ordered_models_brating]
B_ordered_brating = [B_param[int(a)-1] for a in ordered_models_brating]
A_B_ordered_brating = [A_param[int(a)-1]-B_param[int(a)-1] for a in ordered_models_brating]

# =============================================================================
# #Plots to visualize the best models
# =============================================================================

path = path_now +'/Ranking_results/'
path_2= path_now + '/Inputs/'
  
Sum_AD_SRL = [ad+srl for ad, srl in zip(Sum_lik_model_AD, Sum_lik_model_SRL)]

szs = (Norm_b_rating/min(Norm_b_rating))**1.2
szs_max = max(szs)
szs_min = min(szs)
legend_sizes = [szs_min, szs_min + (szs_max - szs_min) / 3, szs_min + 2 * (szs_max - szs_min) / 3, szs_max]
legend_labels = [f'Min Size: {szs_min:.2f}',
                 f'Middle Size 1: {(szs_min + (szs_max - szs_min) / 3):.2f}',
                 f'Middle Size 2: {(szs_min + 2 * (szs_max - szs_min) / 3):.2f}',
                 f'Max Size: {szs_max:.2f}']

legend_handles = [plt.Line2D([0], [0], marker='o', color="w", markeredgewidth = 0.3,
                             markersize=np.sqrt(size), markerfacecolor=[0.7, 0.7, 0.7], 
                             linestyle='None') for size in legend_sizes]

fig20=plt.figure(dpi=600)
aii = fig20.add_subplot()
objective2 = plt.scatter(Combined_log_lik, Norm_b_rating, c=normalized_obj, 
                         edgecolor = "w", cmap="copper", alpha= 1,
                         linewidths = 0.3, rasterized = True, s=10)
cbar = plt.colorbar(objective2, orientation="horizontal", shrink=0.50, pad=0.18)
plt.xlabel("Scaling relationship score (AM + AD)", fontsize=7.5)
plt.ylabel("MFD score", fontsize = 7.5)
plt.xticks(fontsize=7.5)
aii.tick_params(axis='both', which='both', width=0.3)
plt.yticks(np.linspace(round(min(Norm_b_rating), 3), round(max(Norm_b_rating),3),4),
           fontsize=7.5, rotation=90, va ="center")
cbar.set_label('Final score', rotation=0, labelpad=-3, fontsize = 7.5)
cbar.set_ticks([0, 1])
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(axis='both', which='both', width=0.3)
cbar.set_ticklabels(["Best", "Worst"], fontsize = 7.5, rotation=0,
                    ha="center", va="center")

#Activate if you want tags of model names at the side of plot
tags_int = [int(t)-1 for t in ordered_models]

for i in [order_idx[0], order_idx[-1]]:
    if i % 2 == 0:
        off = -0.00005
        plt.text(Combined_log_lik[i]+off, Norm_b_rating[i], "m-" +tags[i], fontsize = 6.5, va = "center", ha = "right") #, bbox=dict(boxstyle="round", pad=0.1,fc="white", ec="gray", lw=0.5))

    else:
        off = -0.0001
        plt.text(Combined_log_lik[i]+off, Norm_b_rating[i], "m-" + tags[i], fontsize = 6.5, va = "center", ha = "right") #, bbox=dict(boxstyle="round", pad=0.1,fc="white", ec="gray", lw=0.5))          

aii.spines['top'].set_linewidth(0.5)
aii.spines['right'].set_linewidth(0.5)
aii.spines['bottom'].set_linewidth(0.5)
aii.spines['left'].set_linewidth(0.5)
decimal_places = 3
aii.set_xticklabels([f"{tick:.{decimal_places}f}" for tick in aii.get_xticks()])
aii.set_yticklabels([f"{tick:.{decimal_places}f}" for tick in aii.get_yticks()])
fig20.savefig(path +'Final_ranking.pdf', format='pdf', dpi = 600)

sizes1 = 90/25.4
sizes2 = 160/25.4
gridspec = dict(hspace=0.2, height_ratios=[1.48,1.48,0.2,1.48,0.2,1.48,0.2,1.48])
fig25, axes =plt.subplots(8,1, figsize = (sizes1, sizes2), dpi = 600, gridspec_kw=gridspec)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
axes[2].set_visible(False)
axes[4].set_visible(False)
axes[6].set_visible(False)

axs0 = axes[0]
axs0.scatter(range(1, len(A_B_ordered)+1), A_B_ordered, c="black",  marker = 'x', s= 2, linewidths=0.3)
axs0.set_title("Final", fontsize = 7.5, pad=1)
axs0.set_ylabel("a-b", fontsize = 7.5)
axs0.set_xticks([])
axs0.tick_params(axis='y', labelsize=7.5, rotation = 0)
axs0.set_xlim(0, len(A_B_ordered)+2)
axs0.tick_params(axis='both', which='both', width=0.3)
slope, intercept = np.polyfit(range(1, len(A_B_ordered)+1), A_B_ordered, 1)
axs0.plot(range(1, len(A_B_ordered)+1), slope*(range(1, len(A_B_ordered)+1))+intercept, linestyle = "dashed", linewidth=0.3, c="red")
_, _, r_value, _, _ = stats.linregress(range(1, len(A_B_ordered)+1), A_B_ordered)
r_squared_final = round(r_value,2)
axs0.text(2, -0.005, "r=" + str(r_squared_final), fontsize = 7.5, horizontalalignment = "left", verticalalignment="bottom")
for label in axs0.get_yticklabels():
    label.set_va('center')
axs0.yaxis.set_label_position("left")
axs0.spines['top'].set_linewidth(0.5)
axs0.spines['right'].set_linewidth(0.5)
axs0.spines['bottom'].set_linewidth(0.5)
axs0.spines['left'].set_linewidth(0.5)

axs0.yaxis.tick_left()

axs1 = axes[3]
axs1.scatter(range(1, len(A_B_ordered)+1), A_B_ordered_AM, c="black", marker = 'x', s= 2, linewidths=0.3)
axs1.set_title("Area-Seismic moment (AM)", fontsize = 7.5, pad = 1)
axs1.set_ylabel("a-b", fontsize = 7.5)
axs1.set_xticks([])
axs1.tick_params(axis='y', labelsize=7.5, rotation = 0)
axs1.set_xlim(0, len(A_B_ordered_AM)+2)
axs1.tick_params(axis='both', which='both', width=0.3)
slope, intercept = np.polyfit(range(1, len(A_B_ordered)+1), A_B_ordered_AM, 1)
axs1.plot(range(1, len(A_B_ordered)+1), slope*(range(1, len(A_B_ordered)+1))+intercept, linestyle = "dashed", linewidth=0.3, c="red")
_, _, r_value, _, _ = stats.linregress(range(1, len(A_B_ordered)+1), A_B_ordered_AM)
r_squared_AM = round(r_value,2)
axs1.text(2, -0.016, "r=" + str(r_squared_AM), fontsize = 7.5, horizontalalignment = "left", verticalalignment = "top")
for label in axs1.get_yticklabels():
    label.set_va('center')
axs1.yaxis.set_label_position("left")
axs1.yaxis.tick_left()
axs1.spines['top'].set_linewidth(0.5)
axs1.spines['right'].set_linewidth(0.5)
axs1.spines['bottom'].set_linewidth(0.5)
axs1.spines['left'].set_linewidth(0.5)

axs3 = axes[1]
axs3.scatter(range(1, len(Sigma_ordered)+1), Sigma_ordered, c="black",marker = 'x', s= 2, linewidths=0.3)
axs3.set_ylabel("$σ_{0}$ \n (MPa)", fontsize = 7.5)
axs3.set_xticks([])
axs3.tick_params(axis='y', labelsize=7.5, rotation = 0)
axs3.set_yticks([100, 125, 150])
axs3.set_xlim(0, len(A_B_ordered)+2)
axs3.tick_params(axis='both', which='both', width=0.3)
for label in axs3.get_yticklabels():
    label.set_va('center')
axs3.yaxis.set_label_position("left")
axs3.yaxis.tick_left()
axs3.spines['top'].set_linewidth(0.5)
axs3.spines['right'].set_linewidth(0.5)
axs3.spines['bottom'].set_linewidth(0.5)
axs3.spines['left'].set_linewidth(0.5)

axs2 = axes[5]
axs2.scatter(range(1, len(A_B_ordered)+1), A_B_ordered_AD, c="black",marker = 'x', s= 2, linewidths=0.3)
axs2.set_title("Area-Average Displacement (AD)", fontsize = 7.5, pad=1)
axs2.set_ylabel("a-b", fontsize = 7.5)
axs2.set_xticks([])
axs2.set_xlim(0, len(A_B_ordered_AD)+2)
axs2.tick_params(axis='both', which='both', width=0.3)
slope, intercept = np.polyfit(range(1, len(A_B_ordered)+1), A_B_ordered_AD, 1)
axs2.plot(range(1, len(A_B_ordered)+1), slope*(range(1, len(A_B_ordered)+1))+intercept, linestyle = "dashed", linewidth=0.3, c="red")
_, _, r_value, _, _ = stats.linregress(range(1, len(A_B_ordered)+1), A_B_ordered_AD)
r_squared_AD = round(r_value,2)
axs2.text(2, -0.005, "r=" + str(r_squared_AD), fontsize = 7.5, horizontalalignment = "left", verticalalignment="bottom")
axs2.tick_params(axis='y', labelsize=7.5, rotation = 0)
for label in axs2.get_yticklabels():
    label.set_va('center')
axs2.yaxis.set_label_position("left")
axs2.yaxis.tick_left()
axs2.spines['top'].set_linewidth(0.5)
axs2.spines['right'].set_linewidth(0.5)
axs2.spines['bottom'].set_linewidth(0.5)
axs2.spines['left'].set_linewidth(0.5)


axs4 = axes[7]
axs4.scatter(range(1, len(Sigma_ordered)+1), A_B_ordered_brating, c="black", marker = 'x', s= 2, linewidths=0.3)
axs4.set_title("MFD", fontsize = 7.5, pad=1)
axs4.set_ylabel("a-b", fontsize = 7.5)
axs4.set_xticks([])
axs4.tick_params(axis='y', labelsize=7.5, rotation = 0)
axs4.set_xlim(0, len(A_B_ordered)+2)
axs4.tick_params(axis='both', which='both', width=0.3)

slope, intercept = np.polyfit(range(1, len(A_B_ordered)+1), A_B_ordered_brating, 1)
axs4.plot(range(1, len(A_B_ordered)+1), slope*(range(1, len(A_B_ordered)+1))+intercept, linestyle = "dashed", linewidth=0.3, c="red")
_, _, r_value, _, _ = stats.linregress(range(1, len(A_B_ordered)+1), A_B_ordered_brating)
r_squared_MFD = round(r_value,2)
axs4.text(2, -0.005, "r=" + str(r_squared_MFD), fontsize = 7.5, horizontalalignment = "left", verticalalignment="bottom")
axs4.set_xticks([1,  len(A_B_ordered)], labels = ["Best", "Worst"], fontsize = 7.5)
for label in axs3.get_yticklabels():
    label.set_va('center')
axs4.yaxis.set_label_position("left")
axs4.yaxis.tick_left()
axs4.spines['top'].set_linewidth(0.5)
axs4.spines['right'].set_linewidth(0.5)
axs4.spines['bottom'].set_linewidth(0.5)
axs4.spines['left'].set_linewidth(0.5)

fig25.savefig(path+'Parameter_sensitivity.pdf', format='pdf', dpi=600)

# =============================================================================
# Export files
# =============================================================================

tags_paper = ["m-"+ i for i in tags]
param_table = np.column_stack((tags_paper, A_param, B_param, B_A_param, Sigma_param))
headers = np.array([["Catalogue", "a", "b", "a-b", "Normal_stress"]])
headers_ranking = np.array([["Catalogue", "Normalized_final_score", "AM_score", "AD_score", "MFD_score"]])
Input_parameters = np.concatenate((headers, param_table))
rankings = np.column_stack((tags_paper, normalized_obj, Norm_lik_AM, Norm_lik_AD, Norm_b_rating))
Ranking_all = np.concatenate((headers_ranking, rankings))
   
exported_ranking = np.savetxt(path+"Ranking_results.txt", Ranking_all, fmt = "%6s", delimiter="\t")
exported_param = np.savetxt(path_2+"Input_Parameters.txt", Input_parameters, fmt = '%s', delimiter="\t")

