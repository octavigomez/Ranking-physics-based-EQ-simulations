#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This code analyzes RSQSim models and their performance based on a series of benchmarks.
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
import re
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
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
file = os.path.join(path, "EBSZ_model.csv")

#Paleosesismic data for the paleorate benchmark
file_2 = os.path.join(path, "coord_paleosites.csv") #Coordinates of paleosites
file_3 = os.path.join(path, "paleo_rates.csv") #Paleorates of each paleosite

#Weights of the respective benchmarks in the objective function
w_AM = 0.25 #AM relationship benchmark
w_AD = 0.25 #AD relationship benchmark
w_MFD = 0.25 #MFD benchmark
w_paleo = 0.25 #Paleorate benchmark

# =============================================================================
# Start
# =============================================================================


input_file = pd.read_csv(file, delimiter =";", header=None)
n=1000

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
SRL_all_tris = round(sum(tri_base[z_2]),0)#[idx_SRL])

#Define path where the models that you want to evaluate are located
path =  path_now + '/Simulation_models/'
c = -1
it = -1
Sum_lik_model = []
A_param = []
B_param = []
B_A_param = []
Sigma_param = []
Tau_param = []
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

# start_folder = path
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
        tau0_1 = list(parameters.values())[10][0]  
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
        Tau_param.append(tau0_1)
        M_range = list(np.arange(-18, 18, 0.01))     
        eList = catalog.get("eList")
        ev, ix = np.unique(eList, return_index=True)
        ix = ix[2000]
        eList = eList[ix:]
        pList = catalog.get("pList")[ix:]
        dList = catalog.get("dList")[ix:]
        #dtauList = catalog.get("dtauList")[ix:]
        #tList = catalog.get("tList")[ix:]
        num_events = list(range(2001, max(eList)+1))
                
        # Filter events generated at border (border effects)           
# =============================================================================
        # condit = (x<=0.1*max(coords_sorted_SRL[:, 0])) | ((max(coords_sorted_SRL[:, 0])*0.9 < x) & (x <= max(coords_sorted_SRL[:, 0])))
        # idx_border = np.where(condit)
        # idx_border_ev = [ib+2001 for ib in idx_border][0]
        # M0 = np.delete(M0, idx_border)
        # M = np.delete(M, idx_border)
        # x = np.delete(x, idx_border)
        # y = np.delete(y, idx_border)
        # z = np.delete(z, idx_border)
        # A = np.delete(A, idx_border)
        # num_events = np.delete(num_events, idx_border)       
        # idx_ev_border = list(np.where(np.isin(eList, num_events))[0])
        # eList = [eList[el] for el in idx_ev_border]
        # pList = [pList[pl] for pl in idx_ev_border]
        #dtauList = [dtauList[dt] for dt in idx_ev_border]
        #tList = [tList[tl] for tl in idx_ev_border]
# =============================================================================
        uniq_patches, p_idx, cts = np.unique(sorted(pList), return_index=True, return_counts = True)
        
        #Calculate completeness (best Combo) and filter catalog by the completeness threshold.
        
        mrange = np.arange(-2, 8.1, 0.1)
        mrange = np.round(mrange,2)
        hist, bins = np.histogram(M, bins = mrange)
        binInterval = 0.1
        half_bin = binInterval/2
        
        def maxCurvature(M, mrange):
            vEvents, bins = np.histogram(M, bins = mrange)
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
            
            # Leonard 2010 for strike-slip faults (DS).
            a = 1.5
            b = 6.09
            sd_b = 0.39 #b 1 sigma range is 5.69-6.47
            
            # Round values to 1 decimal to simplify further comparisons between scaling law and model.

            M_round = [round(elem,2) for elem in M]
            Mmin = min(M)
            Mmax = max(M)
            M_range = [round(elem, 2) for elem in M_range]
    
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
                Mg = (2/3)*logM0-6.07
                Mw.append(Mg)
                sd_Mg = 2/3*sd_logM0
                sd_Mw.append(sd_Mg)
                M_pdf = stats.norm.pdf(M_range, Mg, sd_Mg)
                M_pdf = np.exp(M_pdf)
                Norm_pdf = M_pdf/max(M_pdf) #Normalization  
               #Norm_pdf = [np.float('{:.10000000e}'.format(nd)) for nd in Norm_pdf]
                loc = M_range.index(M_round[c1])              
                Log_lik = np.log(Norm_pdf)             
                Lik_model = Norm_pdf[loc]
                Loglik_model.append(np.log(Lik_model))               
            Sum_lik = np.sum(Loglik_model) # Sumatory of all log-likelihoods 
            Sum_lik_model.append(Sum_lik/len(M))
            
# =============================================================================
#             #B value
# =============================================================================

            #Theoretical            

            b_ref = 1.0
            locs_Mc = np.where(bins==np.round(Mc,1))[0]
            locs_max = np.where(bins == np.round(max(M),1))[0]
            num_eq = hist[locs_Mc[0]:locs_max[0]]
            a_value = np.log10(sum(num_eq))+b_ref*Mc
            N_eq_GR = 10**(a_value-b_ref*bins[locs_Mc[0]:locs_max[0]]) 
            N_eq_GR2= np.concatenate((N_eq_GR, [0]))
            N_eq_noncum = abs(np.diff(N_eq_GR2))
            N_eq_noncum = N_eq_noncum[0:len(num_eq)]
            N_eq_cum = np.cumsum(N_eq_noncum[::-1])
            
            Log_lik_mfd = [poisson.logpmf(round(cat), ref) for cat, ref in zip(N_eq_noncum, num_eq)]
            inf_vals = ~np.isinf(Log_lik_mfd)
            Sum_lik_mfd = sum(np.array(Log_lik_mfd)[inf_vals])/sum(num_eq[inf_vals])
            b_rating.append(Sum_lik_mfd)

            
            # with PdfPages(path+'bvalue_fig.pdf') as pdf:
            fig0 =plt.figure(dpi=600)
            fig = plt.scatter(bins[locs_Mc[0]:locs_max[0]], N_eq_noncum, s=3)
            plt.scatter(bins[locs_Mc[0]:locs_max[0]],num_eq, c="red", s=3)
            plt.xlabel("Magnitude")
            plt.ylabel ("Cumulative earthquakes")
            plt.yscale("log")
        
            def b_value(M):
                b = - (1 / (np.mean(M) + binInterval * 0.5 - np.min(M))) * np.log10(np.exp(1))
                return b
            b = b_value(M)
            
            # plt.figure(dpi=600).add_subplot()
            # plt.scatter(np.flip(mrange[:-1]), np.log10(np.cumsum(hist[::-1])))
            # plt.scatter(np.flip(mrange[:-1]+0.05), np.log10(hist[::-1]))      
            # plt.plot([Mc, Mc],[0, (np.log10(max(hist)))], c='red')
            # plt.text(7,np.log10(max(hist)), "bvalue="+str(round(b,2)))
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
             
            a_AD = 0.5
            b_AD = -4.43
            sd_b_AD = (4.82-4.05)/2

            AD_range = list(np.arange(0, 20, 0.01))
            AD_range = [round(ad,4) for ad in AD_range]
            Loglik_model_AD = []
            c1 = -1
            AD_all = []
            sd_all = []
            for i in A:
                c1 = c1+1
                logAD = a_AD*math.log10(i)+b_AD
                sd_logAD = sd_b_AD
                AD = round(10**(logAD),2)
                sd_AD = round(AD*np.log(10)*sd_logAD,2)
                AD_all.append(AD)
                sd_all.append(sd_AD)
                AD_pdf = stats.norm.pdf(AD_range, AD, sd_AD)    
                AD_pdf= np.exp(AD_pdf)
                Norm_pdf_AD = (AD_pdf/max(AD_pdf)) #Normalization   
                Log_lik_AD = np.log(Norm_pdf_AD)
                loc_AD = AD_range.index(events_AD[c1])
                Lik_model_AD = Norm_pdf_AD[loc_AD]
                Loglik_model_AD.append(np.log(Lik_model_AD))
            Sum_lik_AD = np.sum(Loglik_model_AD) # Sumatory of all log-likelihoods 
            Sum_lik_model_AD.append(Sum_lik_AD/len(events_AD))
                        
# =============================================================================
#             # Just for the EBSZ - Paleoseismic constraint for Mw>6.0
# =============================================================================
            
            #Find the patches that belong to paleoseismic sites based on proximity to coordinates
            
            Coordinates_sites = pd.read_csv(file_2, delimiter =";")
            Recurrence_sites = pd.read_csv(file_3, delimiter =";")
            
            M_paleo = 5.0
            len_catalog = (max(t0)-min(t0))
            
            cat_patch = {}
            for i in surface_patches_SRL:
                cat_patch[str(i)]= events_surface[np.isin(events_surface[:,1], i)] #Seismic catalogs of each patch. Contains: eList, pList, dList

            N_eq_patch = []
            #Catalog_recurrence = []
            Catalog_rate = []
            for i in range(len(Coordinates_sites)):
                X_coord = Coordinates_sites.iloc[i, 1]
                Y_coord = Coordinates_sites.iloc[i, 2]
                all_distances = [(math.sqrt((cpx-X_coord)**2 + (cpy-Y_coord)**2)) for cpx, cpy in zip(coords_patch[:,0], coords_patch[:,1])] #distance between all points
                loc_min = all_distances.index(min(all_distances))               
                patch_site = coords_surface_SRL[loc_min, -1]                   
                catalog_affected_patch = cat_patch.get(str(patch_site))
                events_patch = np.where(np.isin(catalog_affected_patch[:,0], uni_ev))[0]
                magnitudes_patch = M_surface[events_patch]
                condition_paleo = [m>M_paleo for m in magnitudes_patch]
                N_eq_patch = sum(condition_paleo)              
                Catalog_rate.append(N_eq_patch/len_catalog)
                Paleo_rates = list(Recurrence_sites.iloc[:,2])
                
            Num_cat = [c*len_catalog for c in Catalog_rate]
            Num_paleo = [round(c*len_catalog) for c in Paleo_rates]
            Total_cat = sum(Num_cat)
            Total_paleo = sum(Num_paleo)

            #Calculate probability mass function between observed and catalog paleorates

            Sum_lik_paleo = poisson.logpmf(Total_paleo, Total_cat)
            Sum_lik_model_paleo.append(Sum_lik_paleo)
            
            
            
        else:
            print("This catalog is empty")
            idx_empty.append(it)
        
    else:
        print("This catalog is empty")
        idx_empty.append(it)
        
        
# =============================================================================
# #Prepare variables for the objective function
# =============================================================================


Sum_lik_model_perc = Sum_lik_model/sum(Sum_lik_model)
b_rating_perc = b_rating/sum(b_rating)
Sum_lik_model_AD_perc = Sum_lik_model_AD/sum(Sum_lik_model_AD)
Sum_lik_model_paleo_perc = Sum_lik_model_paleo/sum(Sum_lik_model_paleo)
Norm_lik_AM = Sum_lik_model_perc
Norm_b_rating = b_rating_perc 
Norm_lik_AD = Sum_lik_model_AD_perc
Norm_lik_paleo = Sum_lik_model_paleo_perc
Combined_log_lik = [l1+l2 for l1, l2 in zip(Norm_lik_AM, Norm_lik_AD)]

#Objective function

obj_factor = [(fr*w_AM + sc*w_MFD + sa*w_AD + pl*w_paleo) for fr, sc, sa, pl in zip(Norm_lik_AM, Norm_b_rating, Norm_lik_AD, Norm_lik_paleo)]
normalized_obj = [(obj-min(obj_factor))/(max(obj_factor)-min(obj_factor)) for obj in obj_factor]

#Tags to plot models + names side by side

tags = np.array(range(1, len(Sum_lik_model)+1))
tags = [str(s) for s in tags]
tags_name = [os.path.basename(pth) for pth in processed_folder]
models = np.column_stack((tags, normalized_obj, tags_name))
models_AM = np.column_stack((tags, Norm_lik_AM))
models_AD = np.column_stack((tags, Norm_lik_AD))
models_brating = np.column_stack((tags, Norm_b_rating))
models_Mc = np.column_stack((tags, Mc_list))
models_paleo = np.column_stack((tags, Norm_lik_paleo))


order_idx = np.argsort(models[:,1])
ordered_models = models[order_idx,0]
ordered_models_name = models[order_idx,-1]

order_idx_AM = np.argsort(models_AM[:,1])
ordered_models_AM = models_AM[order_idx_AM,0]
ordered_models_AM_paleo = models_AM[order_idx_AM,-1]

order_idx_AD = np.argsort(models_AD[:,1])
ordered_models_AD = models_AD[order_idx_AD,0]
ordered_models_AD_name = models_AD[order_idx_AD,-1]

order_idx_brating = np.argsort(models_brating[:,1])
ordered_models_brating = models_brating[order_idx_brating,0]
ordered_models_brating_name = models_AM[order_idx_brating,-1]

order_idx_paleo= np.argsort(models_paleo[:,1])
ordered_models_paleo = models_paleo[order_idx_paleo,0]
ordered_models_paleo_name = models_paleo[order_idx_paleo, -1]

print("The best models are (ordered from best to worst): " + str(ordered_models_name))

models_all = [str(Sigma_param[int(a)-1]) + "_" + str(A_param[int(a)-1]) + "_" + str(B_param[int(a)-1]) for a in ordered_models]
A_ordered = [A_param[int(a)-1] for a in ordered_models]
B_ordered = [B_param[int(a)-1] for a in ordered_models]
A_B_ordered = [A_param[int(a)-1]-B_param[int(a)-1] for a in ordered_models]

A_ordered_AM = [A_param[int(a)-1] for a in ordered_models_AM]
B_ordered_AM = [B_param[int(a)-1] for a in ordered_models_AM]
A_B_ordered_AM = [A_param[int(a)-1]-B_param[int(a)-1] for a in ordered_models_AM]

A_ordered_AD = [A_param[int(a)-1] for a in ordered_models_AD]
B_ordered_AD = [B_param[int(a)-1] for a in ordered_models_AD]
A_B_ordered_AD = [A_param[int(a)-1]-B_param[int(a)-1] for a in ordered_models_AD]

A_ordered_brating = [A_param[int(a)-1] for a in ordered_models_brating]
B_ordered_brating = [B_param[int(a)-1] for a in ordered_models_brating]
A_B_ordered_brating = [A_param[int(a)-1]-B_param[int(a)-1] for a in ordered_models_brating]

A_ordered_paleo = [A_param[int(a)-1] for a in ordered_models_paleo]
B_ordered_paleo = [B_param[int(a)-1] for a in ordered_models_paleo]
A_B_ordered_paleo = [A_param[int(a)-1]-B_param[int(a)-1] for a in ordered_models_paleo]

Sigma_ordered = [Sigma_param[int(a)-1] for a in ordered_models]
Sigma_ordered_paleo = [Sigma_param[int(a)-1] for a in ordered_models_paleo]

# =============================================================================
# #Plots to visualize the best models
# =============================================================================

path = path_now +'/Ranking_results/'
path_2 = path_now + '/Inputs/'
Sum_AD_SRL = [ad+srl for ad, srl in zip(Sum_lik_model_AD, Sum_lik_model_SRL)]
szs = ((Norm_lik_paleo/min(Norm_lik_paleo))**1.1)*5
szs_max = max(szs)
szs_min = min(szs)
legend_sizes = [szs_min, szs_min + (szs_max - szs_min) / 3, szs_min + 2 * (szs_max - szs_min) / 3, szs_max]
legend_labels = [f'Min Size: {szs_min:.2f}',
                  f'Middle Size 1: {(szs_min + (szs_max - szs_min) / 3):.2f}',
                  f'Middle Size 2: {(szs_min + 2 * (szs_max - szs_min) / 3):.2f}',
                  f'Max Size: {szs_max:.2f}']

legend_handles = [plt.Line2D([0], [0], marker='o', color="black", markeredgewidth = 0.3,
                              markersize=np.sqrt(size), markerfacecolor="None", 
                              linestyle='None') for size in legend_sizes]


fig20=plt.figure(dpi=600)
aii = fig20.add_subplot()
objective2 = plt.scatter(Combined_log_lik, Norm_b_rating, c=normalized_obj, 
                          s=szs, edgecolor = "w", cmap="copper", alpha= 1, 
                          linewidths = 0.01, rasterized = True)
cbar = plt.colorbar(objective2, orientation="horizontal", shrink=0.50, pad=0.18)
plt.xlabel("Scaling relationship score (AM + AD)", fontsize = 7.5)
plt.ylabel("MFD score", fontsize = 7.5)

plt.xticks(fontsize=7.5)
aii.yaxis.set_label_position("right")
aii.yaxis.tick_right()
aii.tick_params(axis='both', which='both', width=0.3)
plt.yticks(np.linspace(round(min(Norm_b_rating), 2), round(max(Norm_b_rating),2),4),
            fontsize=7.5, rotation=90, va ="center")
cbar.set_label('Final score', rotation=0, labelpad=-3, fontsize = 7.5)
cbar.set_ticks([0, 1])
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(axis='both', which='both', width=0.3)
cbar.set_ticklabels(["Best", "Worst"], fontsize = 7.5, rotation=0,
                    ha="center", va="center")

legend = plt.legend(handles=legend_handles, labels=legend_labels, 
                    title="Paleorate score", handleheight=0.5, borderpad=1)
legend.get_title().set_fontsize(7.5)

labels = [text.get_text() for text in legend.get_texts()]
labels_int = [int(re.findall(r'\d+', a)[0]) for a in labels]
labels_int = [round(((a/5)**(1/1.1))*min(Norm_lik_paleo),2) for a in labels_int]
legend.get_texts()[0].set_text(str(labels_int[0]) + " (Best)")
legend.get_texts()[1].set_text("")
legend.get_texts()[2].set_text("")
legend.get_texts()[-1].set_text(str(labels_int[-1]) + " (Worst)")
legend.get_frame().set_linewidth(0.3)
for text in legend.get_texts():
    text.set_fontsize(7.5)


#Activate if you want tags of model names at the side of plot
tags_int = [int(t)-1 for t in ordered_models]
for i in range(len(b_rating)):
    off = 0.001
    plt.text(Combined_log_lik[i]+off, Norm_b_rating[i], tags_name[i], fontsize = 6.5, va = "center", ha = "left")

aii.spines['top'].set_linewidth(0.5)
aii.spines['right'].set_linewidth(0.5)
aii.spines['bottom'].set_linewidth(0.5)
decimal_places = 2

aii.set_xticklabels([f"{tick:.{decimal_places}f}" for tick in aii.get_xticks()])
aii.set_yticklabels([f"{tick:.{decimal_places}f}" for tick in aii.get_yticks()])


fig20.savefig(path+'Final_ranking.pdf', format='pdf', dpi = 600)

gridspec = dict(hspace=0.4, height_ratios=[1.6,1.6,0.1,1.6,0.1,1.6,0.1,1.6, 0.1, 1.6])
fig25, axes =plt.subplots(10,1, dpi = 600, gridspec_kw=gridspec)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
axes[2].set_visible(False)
axes[4].set_visible(False)
axes[6].set_visible(False)
axes[8].set_visible(False)

axs0 = axes[0]
axs0.scatter(range(1, len(A_B_ordered)+1), A_B_ordered, c="black",  marker = 'x', s= 4, linewidths=0.3)
axs0.set_title("Final", fontsize = 7.5, pad=1)
axs0.set_ylabel("a-b", fontsize = 7.5)
axs0.set_xticks([])
axs0.tick_params(axis='y', labelsize=7.5, rotation = 0)
axs0.set_xlim(0, len(A_B_ordered)+1)
axs0.tick_params(axis='both', which='both', width=0.3)
slope, intercept = np.polyfit(range(1, len(A_B_ordered)+1), A_B_ordered, 1)
#axs0.plot(range(1, len(A_B_ordered)+1), slope*(range(1, len(A_B_ordered)+1))+intercept, linestyle = "dashed", linewidth=0.3, c="red")
_, _, r_value, _, _ = stats.linregress(range(1, len(A_B_ordered)+1), A_B_ordered)
r_squared_final = round(r_value,2)
#axs0.text(2, 0.005, "r=" + str(r_squared_final), fontsize = 6.5, horizontalalignment = "left", verticalalignment="top")
for label in axs0.get_yticklabels():
    label.set_va('center')
axs0.yaxis.set_label_position("right")
axs0.spines['top'].set_linewidth(0.5)
axs0.spines['right'].set_linewidth(0.5)
axs0.spines['bottom'].set_linewidth(0.5)
axs0.spines['left'].set_linewidth(0.5)

axs0.yaxis.tick_right()

axs1 = axes[3]
axs1.scatter(range(1, len(A_B_ordered)+1), A_B_ordered_AM, c="black", marker = 'x', s= 4, linewidths=0.3)
axs1.set_title("Area-Magnitude (AM)", fontsize = 7.5, pad = 1)
axs1.set_ylabel("a-b", fontsize = 7.5)
axs1.set_xticks([])
axs1.tick_params(axis='y', labelsize=7.5, rotation = 0)
axs1.set_xlim(0, len(A_B_ordered_AM)+1)
axs1.tick_params(axis='both', which='both', width=0.3)
slope, intercept = np.polyfit(range(1, len(A_B_ordered)+1), A_B_ordered_AM, 1)
axs1.plot(range(1, len(A_B_ordered)+1), slope*(range(1, len(A_B_ordered)+1))+intercept, linestyle = "dashed", linewidth=0.3, c="red")
_, _, r_value, _, _ = stats.linregress(range(1, len(A_B_ordered)+1), A_B_ordered_AM)
r_squared_AM = round(r_value,2)
axs1.text(0.5, -0.015, "r=" + str(r_squared_AM), fontsize = 6.5, horizontalalignment = "left", verticalalignment = "bottom")
for label in axs1.get_yticklabels():
    label.set_va('center')
axs1.yaxis.set_label_position("right")
axs1.yaxis.tick_right()
axs1.spines['top'].set_linewidth(0.5)
axs1.spines['right'].set_linewidth(0.5)
axs1.spines['bottom'].set_linewidth(0.5)
axs1.spines['left'].set_linewidth(0.5)

axs3 = axes[1]
axs3.scatter(range(1, len(Sigma_ordered)+1), Sigma_ordered, c="black",marker = 'x', s= 4, linewidths=0.3)
axs3.set_ylabel("$σ_{0}$ \n (MPa)", fontsize = 7.5)
axs3.set_xticks([])
axs3.tick_params(axis='y', labelsize=7.5, rotation = 0)
axs3.set_yticks([0, 70, 140])
axs3.set_xlim(0, len(A_B_ordered)+1)
axs3.tick_params(axis='both', which='both', width=0.3)
for label in axs3.get_yticklabels():
    label.set_va('center')
axs3.yaxis.set_label_position("right")
axs3.yaxis.tick_right()
axs3.spines['top'].set_linewidth(0.5)
axs3.spines['right'].set_linewidth(0.5)
axs3.spines['bottom'].set_linewidth(0.5)
axs3.spines['left'].set_linewidth(0.5)

axs2 = axes[5]
axs2.scatter(range(1, len(A_B_ordered)+1), A_B_ordered_AD, c="black",marker = 'x', s= 4, linewidths=0.3)
axs2.set_title("Area-Average Displacement (AD)", fontsize = 7.5, pad=1)
axs2.set_ylabel("a-b", fontsize = 7.5)
axs2.set_xticks([])
axs2.set_xlim(0, len(A_B_ordered_AD)+1)
axs2.tick_params(axis='both', which='both', width=0.3)
slope, intercept = np.polyfit(range(1, len(A_B_ordered)+1), A_B_ordered_AD, 1)
#axs2.plot(range(1, len(A_B_ordered)+1), slope*(range(1, len(A_B_ordered)+1))+intercept, linestyle = "dashed", linewidth=0.3, c="red")
_, _, r_value, _, _ = stats.linregress(range(1, len(A_B_ordered)+1), A_B_ordered_AD)
r_squared_AD = round(r_value,2)
#axs2.text(2, 0.005, "r=" + str(r_squared_AD), fontsize = 6.5, horizontalalignment = "left", verticalalignment="top")
axs2.tick_params(axis='y', labelsize=7.5, rotation = 0)
for label in axs2.get_yticklabels():
    label.set_va('center')
axs2.yaxis.set_label_position("right")
axs2.yaxis.tick_right()
axs2.spines['top'].set_linewidth(0.5)
axs2.spines['right'].set_linewidth(0.5)
axs2.spines['bottom'].set_linewidth(0.5)
axs2.spines['left'].set_linewidth(0.5)


axs4 = axes[7]
axs4.scatter(range(1, len(Sigma_ordered)+1), A_B_ordered_brating, c="black", marker = 'x', s= 4, linewidths=0.3)
axs4.set_title("MFD", fontsize = 7.5, pad=1)
axs4.set_ylabel("a-b", fontsize = 7.5)
axs4.set_xticks([])
axs4.tick_params(axis='y', labelsize=7.5, rotation = 0)
axs4.set_xlim(0, len(A_B_ordered)+1)
axs4.tick_params(axis='both', which='both', width=0.3)

slope, intercept = np.polyfit(range(1, len(A_B_ordered)+1), A_B_ordered_brating, 1)
#axs4.plot(range(1, len(A_B_ordered)+1), slope*(range(1, len(A_B_ordered)+1))+intercept, linestyle = "dashed", linewidth=0.3, c="red")
_, _, r_value, _, _ = stats.linregress(range(1, len(A_B_ordered)+1), A_B_ordered_brating)
r_squared_MFD = round(r_value,2)
#axs4.text(0.5, 0.015, "r=" + str(r_squared_MFD), fontsize = 6.5, horizontalalignment = "left", verticalalignment="top")
for label in axs3.get_yticklabels():
    label.set_va('center')
axs4.yaxis.set_label_position("right")
axs4.yaxis.tick_right()
axs4.spines['top'].set_linewidth(0.5)
axs4.spines['right'].set_linewidth(0.5)
axs4.spines['bottom'].set_linewidth(0.5)
axs4.spines['left'].set_linewidth(0.5)
  

axs5 = axes[9]
axs5.scatter(range(1, len(A_B_ordered)+1), A_B_ordered_paleo, c="black", marker = 'x', s= 4, linewidths=0.3)
axs5.set_title("Paleorate", fontsize = 7.5, pad=1)
axs5.set_ylabel("a-b", fontsize = 7.5)
axs5.set_xlabel("Score", fontsize = 7.5, labelpad = -5)
axs5.set_xticks([])
axs5.set_xlim(0, len(A_B_ordered_AD)+1)
axs5.tick_params(axis='both', which='both', width=0.3)
axs5.tick_params(axis='y', labelsize=7.5, rotation = 0)
slope, intercept = np.polyfit(range(1, len(A_B_ordered)+1), A_B_ordered_paleo, 1)
#axs5.plot(range(1, len(A_B_ordered)+1), slope*(range(1, len(A_B_ordered)+1))+intercept, linestyle = "dashed", linewidth=0.3, c="red")
_, _, r_value, _, _ = stats.linregress(range(1, len(A_B_ordered)+1), A_B_ordered_paleo)
r_squared_paleo = round(r_value,2)
#axs5.text(0.5, 0.005, "r=" + str(r_squared_paleo), fontsize = 6.5, horizontalalignment = "left", verticalalignment = "bottom")
for label in axs2.get_yticklabels():
    label.set_va('center')
axs5.yaxis.set_label_position("right")
axs5.yaxis.tick_right()
axs5.set_xticks([1,  len(A_B_ordered)], labels = ["Best", "Worst"], fontsize = 7.5)
axs5.spines['top'].set_linewidth(0.5)
axs5.spines['right'].set_linewidth(0.5)
axs5.spines['bottom'].set_linewidth(0.5)
axs5.spines['left'].set_linewidth(0.5)

fig25.savefig(path+'Parameter_sensitivity.pdf', format='pdf', dpi=600)

# =============================================================================
# Export files
# =============================================================================

param_table = np.column_stack((tags_name, A_param, B_param, B_A_param, Sigma_param))
headers = np.array([["Catalogue", "a", "b", "b-a", "Normal_stress"]])
headers_ranking = np.array([["Catalogue", "Normalized_final_score", "AM_score", "AD_score", "MFD_score", "Paleorate_score"]])
Input_parameters = np.concatenate((headers, param_table))
rankings = np.column_stack((tags_name, normalized_obj, Norm_lik_AM, Norm_lik_AD, Norm_b_rating, Norm_lik_paleo))
Ranking_all = np.concatenate((headers_ranking, rankings))
   
exported_ranking = np.savetxt(path+"Ranking_results.txt", Ranking_all, fmt = "%6s", delimiter="\t")
exported_param = np.savetxt(path_2+"Input_Parameters.txt", Input_parameters, fmt = '%s', delimiter="\t")
