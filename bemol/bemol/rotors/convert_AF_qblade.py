import sys
import os
import pandas as pd
from pathlib import Path


os.makedirs("/home/arthur/Documents/GitHub/bemol/bemol/rotors/AF_QBLADE", exist_ok = True)
af = 'AF'

for i in range(36) : 
    AF = pd.read_csv(f'/home/arthur/Documents/GitHub/bemol/bemol/rotors/mexico_vortex/airfoils/AF{i}.foil', comment = '#', sep = '\t', names = ['AoA', 'Cl', 'Cd'])
    cl = AF['Cl']; cd = AF['Cd']
    af_n = af+str(i)
    AF = pd.DataFrame({
        'Cl': cl,
        'Cd': cd
    })
    out = f"/home/arthur/Documents/GitHub/bemol/bemol/rotors/AF_QBLADE/{af_n}.dat"
    with open(out, 'w') as f:
        f.write(af_n + '\n')
    with open(out, 'a') as f:
        for cl_val, cd_val in zip(AF['Cl'], AF['Cd']):
            f.write(f"{cl_val}     {cd_val}\n")

