#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:31:50 2024

@author: pcorchoc
"""

import numpy as np
from matplotlib import pyplot as plt

def gauss2d(x, y, amplitude, mu_x, mu_y, var_x, var_y):
    z_x = (x - mu_x)**2 / var_x
    z_y = (y - mu_y)**2 / var_y

    g = amplitude * np.exp(- 0.5 * z_x - 0.5 * z_y)
    return g

x, y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
real_mu = 3, 5
real_var = 1, 2**2
real_amplitude = 10
p_real = np.array([real_amplitude, *real_mu, *real_var])

real_g = gauss2d(x.flatten(), y.flatten(), *p_real)

def em_2d_gauss(x, y, data, p0):
    g = gauss2d(x, y, *p0)
    new_g = data * g
    norm = np.sum(new_g)
    mean_x = np.sum(new_g * x) / norm
    mean_y = np.sum(new_g * y) / norm
    
    var_x = np.sum(new_g * (x - mean_x)**2) / norm
    var_x = (1/var_x - 1/p0[3])**-1    
    var_y = np.sum(new_g * (y - mean_y)**2) / norm
    var_y = (1/var_y - 1/p0[4])**-1    
    
    amplitude = np.nanmax(new_g)**0.5
    p = np.array([amplitude, mean_x, mean_y, var_x**0.5, var_y**0.5])
    delta_param = np.mean(np.abs(p0 - p) / (1 +  p0))
    print("delta: ", delta_param)
    while delta_param > 0.01:
        p0 = p.copy()

        g = gauss2d(x, y, *p)
        new_g = data * g
        norm = np.sum(new_g)
        mean_x = np.sum(new_g * x) / norm
        mean_y = np.sum(new_g * y) / norm
        
        var_x = np.sum(new_g * (x - mean_x)**2) / norm
        var_x = (1/var_x - 1/p[3])**-1    
        var_y = np.sum(new_g * (y - mean_y)**2) / norm
        var_y = (1/var_y - 1/p[4])**-1    
        
        amplitude = np.nanmax(new_g)**0.5
        p = np.array([amplitude, mean_x, mean_y, var_x, var_y])
        delta_param = np.mean(np.abs(p0 - p) / (1 +  p0))
        print(delta_param, p)
    
    return p

p0 = (1, 0, 0, 4, 4)

p = em_2d_gauss(x.flatten(), y.flatten(), real_g.flatten(), p0=np.array(p0))
print("Final value: ", p)
print("Input value: ", p_real)

model_g = gauss2d(x.flatten(), y.flatten(), *p)

plt.figure()
plt.subplot(121)
plt.imshow(real_g.reshape(x.shape))
plt.colorbar()
plt.subplot(122)
plt.imshow(model_g.reshape(x.shape))
plt.colorbar()


