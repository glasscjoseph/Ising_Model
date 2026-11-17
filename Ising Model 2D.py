#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 18:42:40 2024

@author: josephglass
"""

import numpy as np
import random
import math
import statistics as stat
import matplotlib.pyplot as plt

# Number of iterions to be used
N_iterations = 500

# Size of the lattice N x N
N = 50

# Parameters
J = 1
h = 0
kT = 0.5


# =============================================================================
# Create different intial lattices for study
# =============================================================================

latticeUp = np.ones((N, N))


lattice75 = np.zeros((N,N))
for y in range(N):
    for x in range(N):
        if np.random.random() <= 0.75:
            lattice75[x,y] = 1
        else:
            lattice75[x,y] = -1


lattice25 = np.zeros((N,N))
for y in range(N):
    for x in range(N):
        if np.random.random() <= 0.25:
            lattice25[x,y] = 1
        else:
            lattice25[x,y] = -1


latticeR = np.zeros((N,N))
for y in range(N):
    for x in range(N):
        if np.random.random() <= 0.5:
            latticeR[x,y] = 1
        else:
            latticeR[x,y] = -1
            


# =============================================================================
# Function to find the energy of a given lattice
# =============================================================================

def E_lattice(lattice):
    sumJ = 0
    termh = 0
    for y in range(N):
        for x in range(N):
            sumJ += lattice[x,y]*lattice[(x+1)%N,y]
            sumJ += lattice[x,y]*lattice[(x-1)%N,y]
            
            sumJ += lattice[x,y]*lattice[x,(y+1)%N]
            sumJ += lattice[x,y]*lattice[x,(y-1)%N]
            
            termh += h*lattice[x,y]
    energy = -J*sumJ - termh
    return energy



# =============================================================================
# Metropolis algorithm
# =============================================================================

def metropolis(lattice, iterations, temp, energy):
    lattice2 = lattice.copy()
    mag_Total = list()
    energy_Total = list()
    acceptance = list()
    
    for i in range(0,iterations):
        
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        
        spin_Old = lattice2[x,y]
        spin_Trial = spin_Old*-1 
        
        E_Old = 0
        E_Trial = 0
        
        E_Old += -J*spin_Old*lattice2[(x-1)%N][y]
        E_Trial += -J*spin_Trial*lattice2[(x-1)%N][y]
        
        E_Old += -J*spin_Old*lattice2[(x+1)%N][y]
        E_Trial += -J*spin_Trial*lattice2[(x+1)%N][y]
        
        E_Old += -J*spin_Old*lattice2[x][(y-1)%N]
        E_Trial += -J*spin_Trial*lattice2[x][(y-1)%N]
        
        E_Old += -J*spin_Old*lattice2[x][(y+1)%N]
        E_Trial += -J*spin_Trial*lattice2[x][(y+1)%N]
        
        energy_Change = E_Trial-E_Old
        
        if energy_Change <= 0:
            lattice2[x,y]=spin_Trial   
            energy += energy_Change
            acceptance.append(1)
            
        else:
            prob = math.exp(-energy_Change/temp)
            rnum = random.random()
            if rnum <= prob:
                lattice2[x,y]=spin_Trial
                energy += energy_Change
                acceptance.append(1)
            else:
                acceptance.append(0)
            
        mag_Total.append(lattice2.sum())
        energy_Total.append(energy)
    
    avgAcceptance = stat.mean(acceptance)
    print("Acceptance rate",avgAcceptance)
            
    return mag_Total, energy_Total, lattice2



# =============================================================================
# Equilibration
# =============================================================================

mag, energy, latticeEq = metropolis(lattice75, N_iterations, kT, E_lattice(lattice75))

MC_steps = [i/N**2 for i in range(N_iterations)]

for i in range(len(mag)):
    mag[i] = mag[i]/N**2

plt.figure(1)
plt.plot(MC_steps,mag)
plt.xlabel('Monte Carlo step', fontsize=12)
plt.ylabel(r'Average magnetisation, $\langle M \rangle$', fontsize=12)
plt.grid()
plt.show()

plt.figure(2)
plt.plot(MC_steps,energy)
plt.xlabel('Monte Carlo step', fontsize=12)
plt.ylabel('Energy, E', fontsize=12)
plt.grid()
plt.show()



# =============================================================================
# Temperature sweep
# =============================================================================

kT_range = np.arange(0.1, 8.0, 0.1)

def avg_Magnetisation(lattice, kT_range):
    mag_Means = list()
    for i, temp in enumerate(kT_range):
        mag, energy, latticeEq = metropolis(lattice, N_iterations, temp, E_lattice(lattice))
        mag_Means.append(stat.mean(mag[-300000:])/N**2)
    return mag_Means

mag_Means_75 = avg_Magnetisation(lattice75, kT_range)
mag_Means_25 = avg_Magnetisation(lattice25, kT_range)
mag_Means_R = avg_Magnetisation(latticeR, kT_range)

plt.figure(3)
plt.plot(kT_range, mag_Means_75, 'o--', label='Initial state of mag 75% aligned upward')
plt.plot(kT_range, mag_Means_R, 'o--', label='Initial state of mag random')
plt.plot(kT_range, mag_Means_25, 'o--', label='Initial state of mag 75% aligned downward')
plt.xlabel(r'Temperature, $k_{B}T$',fontsize=12)
plt.ylabel(r'Average Magnetisation, $\langle M \rangle$',fontsize=12)
plt.legend()
plt.grid()
plt.show()


# =============================================================================
# Exact solution
# =============================================================================

def exactMag(temp):
    M = (1-(math.sinh(2*(1/temp)*J)**(-4)))**(1/8)
    return M
    

kT_range = np.arange(0.1, 6.0, 0.0125)
exact_mags = list()

for i in range(0,168):
    exact_mags.append(exactMag(kT_range[i]))

for j in range(len(kT_range)-168):
    exact_mags.append(0)
    
plt.figure(4)
plt.plot(kT_range,exact_mags)
plt.xlabel(r'$k_{B}T$', fontsize=12)
plt.ylabel(r'$Magnetisation$',fontsize=12)
plt.grid()
plt.show()


# =============================================================================
# Showing the lattice
# =============================================================================

mag, energy, latticeEq = metropolis(lattice75, N_iterations, 0.1, E_lattice(lattice75))

plt.imshow(latticeEq)
