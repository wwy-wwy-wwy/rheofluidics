import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp
import cv2
import scipy
from tqdm import tqdm

import skimage.io as io

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

def remove_outliers_from_lists(y_i, x_i):
	mask = np.logical_and(np.logical_and(np.isfinite(x_i), np.isfinite(y_i)), x_i > 0)

	x_i = x_i[mask]
	y_i = y_i[mask]
	#print(x_i)
	# Convert to numpy array for easy manipulation
	coords = np.array(list(zip(y_i, x_i)))

	# Compute the center of the points
	center = np.mean(coords, axis=0)
	#print(center)
	# Compute the radii (distance from center)
	radii = np.linalg.norm(coords - center, axis=1)
	#print(radii)

	# Compute mean and standard deviation of radii
	mean_radius = np.mean(radii)


	# Identify indices that are NOT outliers
	inlier_indices = (radii > mean_radius - 5) & (radii < mean_radius + 5)

	# Filter the coordinates based on the inlier indices
	filtered_coords = coords[inlier_indices]

	# Split the coordinates back into y and x lists
	filtered_y, filtered_x = filtered_coords[:, 0].tolist(), filtered_coords[:, 1].tolist()

	return filtered_y, filtered_x


def sigmoid_fit(x, A, x0, w,B):
	return A*(1/(1+np.exp(-(x-x0)/w))) + B


def detect_edge_sigmoid(img_array,roi,plot=False):
	xc = roi; yc = roi
	yy, xx = np.meshgrid(np.arange(img_array.shape[0]),np.arange(img_array.shape[1]))
	rr = np.sqrt((xx - xc)**2 + (yy-yc)**2)
	ttheta = np.arctan2(yy-yc, xx-xc)
	numberpoints=20
	d_ttheta = np.pi/numberpoints
	N_points = 2*numberpoints-1
	x_i = np.zeros(N_points); y_i = np.zeros(N_points)
	if plot is True:
		ncol = 10
		nrow = int(np.ceil(N_points/ncol))
		x_size = 2; y_size = 2
		ax = plt.subplots(nrow, ncol, figsize=(x_size*ncol, y_size*ncol))[1].flatten()
	for bin_idx in range(N_points):
		ttheta_min = bin_idx * d_ttheta-np.pi; ttheta_max = (bin_idx+1) * d_ttheta-np.pi
		mask = np.logical_and(ttheta<=ttheta_max, ttheta >ttheta_min)
		r_plot = rr[mask].flatten(); n_plot = img_array[mask].flatten()
		I = np.argsort(r_plot); r_plot = r_plot[I]; n_plot = n_plot[I]
		p = [n_plot.max()-n_plot.min(), 15, 1.5, n_plot.min()]
		if plot is True:
			ax[bin_idx].plot(r_plot,n_plot,'s')
		try:
			popt, pcov = scipy.optimize.curve_fit(sigmoid_fit, r_plot, n_plot,p0=p)
			#print(popt)
			r_boundary = popt[1]
			y_i[bin_idx] = yc + r_boundary*np.cos((ttheta_min+ttheta_max)/2)
			x_i[bin_idx] = xc + r_boundary*np.sin((ttheta_min+ttheta_max)/2)
          
			if plot is True:
				ax[bin_idx].plot(r_plot,sigmoid_fit(r_plot,*popt),'-')
          
		except:
			y_i[bin_idx] = np.nan
			x_i[bin_idx] = np.nan
         


	# # filter out outliers
	filtered_y, filtered_x = remove_outliers_from_lists(y_i, x_i)
	#return y_i, x_i
	return filtered_y, filtered_x

def poly_fit_theta(x, a,b,c,d,e):
    return 1 + a*np.cos(x)+b*np.cos(2*x) + c*np.cos(3*x) + d*np.cos(4*x) + e*np.cos(5*x)


def calculate_gamma_fit(y,x,directory,framenum): # the edges are before rotation
    # retrieve the mass center from image itself
    
#     _, binarized = cv2.threshold(bparts, (thres[0]+thres[1])/2, 255, cv2.THRESH_BINARY)
#     dark_coords = np.column_stack(np.where(binarized == 0))
    
#     # Compute the average of these coordinates
#     centroid = dark_coords.mean(axis=0)

    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    centered_x=x-centroid_x
    centered_y=y-centroid_y
    centroid=[centroid_y,centroid_x]

    # now you can calculate r and theta from the centered x and y values
    r = np.sqrt(centered_x**2 + centered_y**2)
    theta = np.arctan2(centered_y, centered_x)

    # calculate r_bar, which is the mean radius
    r_bar = np.mean(r)

    # normalize r by r_bar
    r_normalized = r / r_bar

    p = [0.1,0.1,0.1,0.1,0.1]
    p2=[0.1,0.1]
    popt, pcov = scipy.optimize.curve_fit(poly_fit_theta, theta, r_normalized,p0=p)

    # Number of points we want for the reconstructed droplet

    N_points = 1000
    # Array of theta values for the reconstructed droplet
    theta_reconstructed = np.linspace(-np.pi, np.pi, N_points)

    # Start with the average radius for the reconstructed droplet
    r_reconstructed = poly_fit_theta(theta_reconstructed,*popt)* r_bar

    # Convert back to Cartesian coordinates for plotting
    x_reconstructed = r_reconstructed * np.cos(theta_reconstructed)
    y_reconstructed = r_reconstructed * np.sin(theta_reconstructed)

    # Plot the reconstructed droplet shape
    plt.figure(figsize=(6, 6))
    plt.plot(centered_x, centered_y,'.')
    plt.plot(x_reconstructed, y_reconstructed)
    plt.axis('equal')  # ensure the x and y axis scales are equal
    plt.grid(True)
    plot_filename = os.path.join(directory, f'reconstruct_{framenum}.png')
    plt.savefig(plot_filename)
    plt.close() # Closes the current figure


    return popt, centroid #centroid[0] is y and centroid[1] is x

def process_frames(img_tif, ref, df, particle_num, directory):
    roi=25
    total_frames = df[df['particle'] == particle_num]['frame']
    gamma2=[]
    for framenum in total_frames:
        print('frame number is'+str(framenum))
        filtered_t1 = df[(df['particle'] == particle_num) & (df['frame'] == framenum)]
        xloc = int(filtered_t1['x'])
        yloc = int(filtered_t1['y'])

        parts=img_tif[framenum][yloc-roi:yloc+roi,xloc-roi:xloc+roi]-ref[yloc-roi:yloc+roi,xloc-roi:xloc+roi]+100
        blurred_parts=cv2.GaussianBlur(parts,(7,7),3)
        y_i,x_i=detect_edge_sigmoid(blurred_parts,roi)
        plt.imshow(parts)
        plt.plot(x_i,y_i,'o')
        plot_filename = os.path.join(directory, f'plot_{framenum}.png')
        plt.savefig(plot_filename)
        plt.close() # Closes the current figure
        if len(y_i)>7:
            popt,centroid=calculate_gamma_fit(y_i,x_i,directory,framenum)
            df.loc[(df['particle'] == particle_num) & (df['frame'] == framenum), 'x'] = xloc -roi + centroid[1]
            df.loc[(df['particle'] == particle_num) & (df['frame'] == framenum), 'y'] = yloc -roi + centroid[0]
            gamma2.append(popt[1])
        else:
            gamma2.append(np.nan)
    return gamma2

def convert_ms(ulh):
    ms=ulh*10**(-9)/3600 #m^3/s
    return ms

def convert_ulh(ms):
    ulh=ms/(10**(-9))*3600 #ul/h
    return ulh

def calculate_stress(max_v):
	sigma_0=18 #pa
	h=10*10**(-6) # 10um
	d=100*10**(-6) # 100 um
	Q=2*10**(-10)*2 #m^3/s, same as 720 ul/hr
	q=3*Q/4/h
	eta=0.0168 #pa*s
	sigma_rescaled=sigma_0/q/eta

	h_exp=73*10**(-6) #!!!
	L_exp=100*10**(-6) #m
	Q_exp=4*h_exp*L_exp*(max_v)/3
	print('Q is: '+ str(convert_ulh(Q_exp)) + ' ul/h')
	q_exp=3*Q_exp/4/h_exp
	eta_exp=0.0341 #pa*s #!!!
	sigma_exp=sigma_rescaled*q_exp*eta_exp
	return sigma_exp

def sine_func_freq(x, A, B, C, omega=0.1):  # Here, omega is given a default value of 1.0, but you can replace it with any other value
    return A * np.sin(omega * x + B) + C


def calculate_modulus(framerate,gamma,sigma):
	lower_bounds = [0, 0,-0.1]
	upper_bounds = [np.inf, np.pi,0.1]

	if (type(gamma) is list) == True:
		x=np.arange(0,len(gamma))/framerate
		non_nan_indices = ~np.isnan(gamma)
		x=x[non_nan_indices]
		gamma=np.array(gamma)[non_nan_indices]
		fixed_omega=2 * np.pi / (len(gamma)/framerate)

		# Fit the sine function with the constraints
		params, _ = curve_fit(lambda x, A, B, C: sine_func_freq(x, A, B, C, omega=fixed_omega), x, gamma, 
							  p0=[0.05,2,0], method='trf', 
							  bounds=(lower_bounds, upper_bounds))
		print(fixed_omega)

		storage_modulus=sigma/params[0]*np.cos(np.pi-params[1])
		#storage_modulus_list.append(storage_modulus)
		print('Storage modulus is:'+str(storage_modulus))
		loss_modulus=sigma/params[0]*np.sin(np.pi-params[1])
		#loss_modulus_list.append(loss_modulus)
		print('Loss modulus is:'+str(loss_modulus))

		y_fit = sine_func_freq(x, *params,omega=fixed_omega)

		y=gamma
		residuals = y - y_fit
		ss_res = np.sum(residuals ** 2)
		ss_tot = np.sum((y - np.mean(y)) ** 2)
		r_squared = 1 - (ss_res / ss_tot)
		print('R^2: ' + str(r_squared))

		print(params)
		plt.plot(x, y, 'o', label='Data')
		plt.plot(x, y_fit, label='Fit')
		plt.show()
	else:
		storage_modulus=np.nan
		loss_modulus=np.nan
		fixed_omega=np.nan
		r_squared=np.nan
	return storage_modulus,loss_modulus, fixed_omega, r_squared


def analyze_wavelength_gamma2(wavelength_data, wavenum, imgname, img_tif, ref):
    unique_particles = wavelength_data['particle'].unique()
    gamma_2_list=[]
    G_p_list=[]
    G_pp_list=[]
    current_directory = os.getcwd()
    for particle in unique_particles:
        print('particle number: '+str(particle))
        total_frames = wavelength_data[wavelength_data['particle'] == particle]['frame']

        gamma_2 = []
        # First, create a path for the 'v1' subdirectory
        v_directory = os.path.join(current_directory, imgname)
        w_directory = os.path.join(v_directory, str(wavenum))
        # Then, create a path for the 'particle' directory inside 'v1'
        result_directory = os.path.join(w_directory, str(particle))

        try:
            os.makedirs(result_directory, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {result_directory}: {e}")


        gamma2 = process_frames(img_tif,ref, wavelength_data,particle,result_directory)
        gamma_2_list.append(gamma2)
    return gamma_2_list

def analyze_wavelength_modulus(wavelength_data,gamma2_list,real_stress,framerate,imgname):
    G_p_list=[]
    G_pp_list=[]
    omega_list=[]
    r2_list=[]
    particle_list=(wavelength_data['particle'].unique()).tolist()
    for index,gamma2 in enumerate(gamma2_list):
        print('particle is:' + str(particle_list[index]))
        G_p,G_pp,omega,r2=calculate_modulus(framerate,gamma2,real_stress)
        G_p_list.append(G_p)
        G_pp_list.append(G_pp)
        omega_list.append(omega)
        r2_list.append(r2)

    data = {
        "Droplet Number": particle_list,
        "Storage Modulus": G_p_list,
        "Loss Modulus": G_pp_list,
        "Frequency": omega_list,
        "R^2": r2_list
    }
    # Create a DataFrame with the dictionary
    df = pd.DataFrame(data)

    # Assign constant values for frequency and video name
    df["Video Name"] = imgname

    df = df[df["R^2"] >= 0.75]

    return df


        