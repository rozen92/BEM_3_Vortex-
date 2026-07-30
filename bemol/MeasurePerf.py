import bemol.DataMaker.src.SimEnv as sv 
import bemol as bem
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube as lhc
import pathlib as p
import pandas as pd
import numpy as np
import multiprocessing as mp
from time import perf_counter

sampler = lhc(2, strength = 1, seed = 42)
samples = sampler.random(n = 100)

tsrs = samples[0:20,0]*8 + 4
yaws = samples[0:20,1]*60 - 30

corrections = [
    bem.secondary.hubTipLoss.Prandtl,
    bem.secondary.skewAngle.Burton,
    bem.secondary.turbulentWakeState.Buhl,
    bem.secondary.yawModel.IFPEN
]

sim_test = sv.SimEnv(omega = 44.5163679, U = 12.520228472, yaw = 15, skew = 15)

nbr_az = 42

print(f"--- Mesures de performances réalisées pour une grille 36 * {nbr_az} et {yaws.shape[0]} couples TSR/yaw.\n")

print("--- Début des mesures de performances du solveur séquentiel.\n")

start = perf_counter()
seq  = sim_test.data_maker(yaws, tsrs, nbr_az, export = False)
stop = perf_counter()
time = np.round((stop-start)/60, decimals=2)

print(f"\n[PERF]    Temps d'exécution (seq) : {time} minutes.\n")

print("--- Début des mesures de performances du solveur parallèle.\n")

start = perf_counter()
para = sim_test.para_data_maker(yaws, tsrs, nbr_az)
stop = perf_counter()
time = time = np.round((stop-start)/60, decimals=2)

print(f"\n[INFO]    Nombre de coeur disponibles : {mp.cpu_count()}")
print(f"\n[PERF]    Temps d'exécution (para) : {time} minutes.\n")

