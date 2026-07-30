import os
import sys
import numpy as np
import pandas as pd
import time
from scipy.stats.qmc import LatinHypercube as lhc
import matplotlib.pyplot as plt

"""
np.random.seed(42)

cube = np.zeros((11, 2))

## Sampler LHS de base
sampler = lhc(2, strength = 1, seed = 32)
samples = sampler.random(n = 5)



samples[:,0] = samples[:,0]*8 + 4
samples[:,1] = samples[:,1]*60 - 30


fig, ax = plt.subplots()

X = np.linspace(0,1,11)
Y = np.linspace(0,1,11)

# Plot the vertical lines at specified positions
for x in X:
    ax.axvline(x=x, color='black', linestyle='--', linewidth = 0.5)

# Plot the horizontal lines at specified positions
for y in Y:
    ax.axhline(y=y, color='black', linestyle='--', linewidth = 0.5)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.scatter(samples[:,0], samples[:,1], color = 'blue')
plt.show()

# Colonne 1 --> dimension du TSR
# Colonne 2 --> dimension du Yaw

cube[:,0] = np.linspace(0,1,11)
cube[:,1] = np.linspace(0,1,11)

new_samples = np.zeros((10,2))
new_samples[0:5, :] = samples

## Identifier les cellules déjà occupée

valid_lines = []
is_val_line = True
for i in range(len(cube[:,1])-1) :
    for s in range(len(samples[:,1])) :
        if samples[s,1] < cube[i+1,1] and samples[s,1] > cube[i,1] :
            is_val_line = False
            break
    valid_lines.append(is_val_line)
    is_val_line = True


valid_cols = []
is_val_col = True
for j in range(len(cube[:,0])-1):
    for s in range(len(samples[:,0])):
        if samples[s,0] < cube[j+1,0] and samples[s,0] > cube[j,0] :
            is_val_col = False
            break
    valid_cols.append(is_val_col)
    is_val_col = True

new_samp = []
it = -1
for i in range(len(valid_lines)) :

    if valid_lines[i] != True :
        continue

    k = it+1
    bool = valid_cols[k]

    while bool != True :
        k+=1
        bool = valid_cols[k]

    new_x = cube[k,0] + (cube[k+1,0]-cube[k,0])*np.random.rand()
    new_y = cube[i,1] + (cube[i+1,1]-cube[i,1])*np.random.rand()
    it = k
    new_samp.append([new_x, new_y])

new_samp = np.array(new_samp)

new_samples[5:,:] = new_samp

fig, ax = plt.subplots()

X = np.linspace(0,1,11)
Y = np.linspace(0,1,11)

# Plot the vertical lines at specified positions
for x in X:
    ax.axvline(x=x, color='black', linestyle='--')

# Plot the horizontal lines at specified positions
for y in Y:
    ax.axhline(y=y, color='black', linestyle='--')

ax.set_xlim(0,1)
ax.set_ylim(0,1)

ax.scatter(samples[:,0],samples[:,1], color = 'blue')
ax.scatter(new_samp[:,0], new_samp[:,1], color = 'red')
plt.show()

x = 0
"""

def lhs_extenderV1(old_seed, n_samples) : 

    np.random.seed(old_seed)
    sampler = lhc(2, strength = 1, seed = old_seed)
    samples = sampler.random(n_samples)

    cube = np.zeros((n_samples*2 + 1, 2))
    cube[:,0] = np.linspace(0,1, n_samples*2 + 1)
    cube[:,1] = cube[:,0]

    valid_lines = []
    is_val_line = True
    for i in range(len(cube[:,1])-1) :
        for s in range(len(samples[:,1])) :
            if samples[s,1] < cube[i+1,1] and samples[s,1] > cube[i,1] :
                is_val_line = False
                break
        valid_lines.append(is_val_line)
        is_val_line = True


    valid_cols = []
    is_val_col = True
    for j in range(len(cube[:,0])-1):
        for s in range(len(samples[:,0])):
            if samples[s,0] < cube[j+1,0] and samples[s,0] > cube[j,0] :
                is_val_col = False
                break
        valid_cols.append(is_val_col)
        is_val_col = True

    new_samp = []
    it = -1
    for i in range(len(valid_lines)) :

        if valid_lines[i] != True :
            continue

        k = it+1
        bool = valid_cols[k]

        while bool != True :
            k+=1
            bool = valid_cols[k]

        new_x = cube[k,0] + (cube[k+1,0]-cube[k,0])*np.random.rand()
        new_y = cube[i,1] + (cube[i+1,1]-cube[i,1])*np.random.rand()
        it = k
        new_samp.append([new_x, new_y])

    new_samp = np.array(new_samp)

    new_samples = np.zeros((n_samples*2,2))
    new_samples[0:n_samples,:] = samples
    new_samples[n_samples:,:] = new_samp

    return new_samples

def rescale(samples) :
    samp = samples.copy()
    samp[:,0] = 8*samples[:,0] + 4
    samp[:,1] = 60*samples[:,1] - 30
    return samp


if __name__ == '__main__' :

    ## Sampler LHS de base
    sampler = lhc(2, strength = 1, seed = 42)
    samples = sampler.random(n = 100)


    new_samples = lhs_extenderV1(old_seed = 42, n_samples = 5)

    fig, ax = plt.subplots()

    X = np.linspace(0,1,11)
    Y = np.linspace(0,1,11)

    # Plot the vertical lines at specified positions
    for x in X:
        ax.axvline(x=x, color='black', linestyle='--', linewidth = 0.5)

    # Plot the horizontal lines at specified positions
    for y in Y:
        ax.axhline(y=y, color='black', linestyle='--', linewidth = 0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.scatter(new_samples[:,0], new_samples[:,1], color = 'blue')
    plt.show()

    new_samples = rescale(new_samples)    
    print(new_samples)

    for i in range(20,30,1) :
        print(i)