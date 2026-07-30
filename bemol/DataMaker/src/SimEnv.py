import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), '..')))

import bemol as bem
import numpy as np
import pandas as pd
import pathlib as p
import matplotlib.pyplot as plt
import multiprocessing as mp


## Définir des variables par défaut pour plus de lisibilité
rho = 1.191

rotor_dir_vortex = str(p.Path(__file__).resolve().parents[2] / 'bemol' / 'rotors' / 'mexico_vortex')
rotor_vortex = bem.rotor.Rotor(rotor_dir_vortex)
corrections = [
    bem.secondary.hubTipLoss.Prandtl,
    bem.secondary.skewAngle.Burton,
    bem.secondary.turbulentWakeState.Buhl,
    bem.secondary.yawModel.IFPEN
]
solver_yawOn = bem.ning.NingUncoupled(rotor_vortex, rho, corrections)


class SimEnv :

    ## Paramètres mécaniques et relatif à la géométrie
    precone : float
    tilt : float
    pitch : float
    rotor_path : str
    rotor : bem.rotor.Rotor
    omega : float

    ## Paramètres métérologiques (pour l'instant un seul, mais à enrichir)
    U : float
    rho : float
    yaw : float
    skew : float
    act_yaw : bool

    ## Paramètres de simulations
    nbr_rev : float
    tsr : float
    tStep : float

    ## Solveur BEM (seul NingUncoupled est supporté pour l'instant)
    solver : bem.ning.NingUncoupled
    corrections : list
    

    def __init__(self, 
                 omega, U, yaw, skew, corrections:list = corrections,
                 rotor_dir:str = rotor_dir_vortex,
                 rho = 1.191, Nbr_rev = 1.0, tStep = 0,precone = 0, tilt = 0,
                 ) :
        
        ## Charger les paramètres mécaniques
        self.precone = precone
        self.tilt = tilt
        self.rotor = bem.rotor.Rotor(rotor_dir)
        self.pitch = self.rotor.pitchRated 
        self.rotor_dir = rotor_dir
        self.omega = omega

        ## Charger les paramètres métérologiques
        self.U       = U
        self.rho     = rho
        self.yaw     = yaw
        self.skew    = skew 
        if bem.secondary.yawModel.IFPEN or bem.secondary.yawModel.PittAndPeters in corrections :
            self.act_yaw = True
        else :
            self.act_yaw = False

        ## Charger les paramètres de simulations
        self.nbr_rev = Nbr_rev
        self.tsr     = self.rotor.sections[-1].radius*self.omega/U
        self.tStep   = tStep

        ## Charger le solver
        self.corrections = corrections
        self.solver = bem.ning.NingUncoupled(self.rotor, self.rho, corrections)

    def print(self):

        print("#"*100)
        print("Affichage des paramètres de la simulation")
        print("#"*100)

        print("\n\n")

        print("#"*100)
        print('------- Paramètres mécaniques/géométriques -------\n')
        print(f"----> Precone : {self.precone}\n")
        print(f"----> Tilt : {self.tilt}\n")
        print(f"----> Vitesse de rotation (omega) : {self.omega}\n")
        print(f"----> Rotor (chemin) : {self.rotor_dir}\n")
        print(f"----> Pitch : {self.rotor.pitchRated}")
        print("#"*100)

        print("#"*100)
        print('------- Paramètres métérologiques -------\n')
        print(f"----> Vitesse du vent (U) : {self.U}\n")
        print(f"----> densité de l'air : {self.rho}\n")
        print(f"----> yaw : {self.yaw} degré\n")
        print(f"----> skew : {self.skew} degré\n")
        print("#"*100)

        print("#"*100)
        print('------- Paramètres de simulation -------\n')    
        print(f"----> Nombre de révolution : {self.nbr_rev}\n")
        print(f"----> Tsr : {self.tsr}\n")
        print(f"----> tStep : {self.tStep}")
        print("#"*100)        

        print("#"*100)
        print('------- Paramètres du solver -------\n')
        print(f"----> Géométrie utilisé : {self.rotor_dir}\n")
        print(f"----> Corrections : {self.corrections}\n")
        if self.act_yaw == True :
            print(f"----> Yaw activé, modèle {self.corrections[-1]}\n")
        else : 
            print("----> Yaw désactivé\n") 
        print("#"*100)

        return                                                       
    
    def set_wind(self, tsr) :
        self.U = self.omega*2.25/tsr
        return
    
    def data_maker(self, yaws:list, tsrs:list, nbr_az, export = True) :
        deg_azs = np.linspace(0,360, nbr_az, endpoint = False)
        rad_azs = np.radians(deg_azs)
        data_list = []

        for i in range(len(yaws)):
            yaw = np.radians(yaws[i])
            tsr = tsrs[i]
            self.set_wind(tsr)
            for i, az in enumerate(rad_azs) :
                for j in range(len(self.rotor.sections)) :
                    ## Calculer les efforts normaux et les inductions
                    velocities = bem.tools.calculateVelocity(wind = self.U, omega = self.omega, rad = self.rotor.sections[j].radius, azi = rad_azs[i],
                                                    yaw = yaw, tilt = self.tilt,
                                                    precone = self.precone)                
                    fn,ft,ai,at = self.solver.solve(self.rotor.sections[j], rad_azs[i], pitch = self.rotor.pitchRated,
                                                velocity = velocities, angles = [yaw, self.tilt])

                    angle = self.rotor.sections[j].twist + self.rotor.pitchRated
                    _, aoa = compute_inflow_aoa(self.solver, velocities[0], velocities[1], angle)
                    V_eff = np.sqrt((self.U*(1-ai))**2 + (self.omega*self.rotor.sections[j].radius*(1+at))**2)                             

                    data_list.append({
                            'yaw': np.degrees(yaw),
                            'TSR' : tsr,
                            'r': self.rotor.sections[j].radius,
                            'theta': deg_azs[i],  
                            'Fn': fn,
                            'Ft': ft,                    
                            'V_eff': V_eff,
                            'Alpha_deg': np.degrees(aoa)
                            })
        df = pd.DataFrame(data_list)

        ## Exporter les données calculées dans le dossier Data/
        if export == True :
            path_data = p.Path(os.path.dirname(__file__)) / 'exports' / 'Data'
            path_data.mkdir(parents=True, exist_ok=True)
            df.to_csv(path_data/'bem_data.csv', index = False, sep = ',')
            print(f"Données calculées pour yaws == [{yaws}] et tsrs == [{tsrs}].\nEnregistrées à l'adresse {path_data}")

        return df   
    
    def para_data_maker(self, yaws:list, tsrs:list, nbr_az, export:p.Path = None):
        deg_azs = np.linspace(0,360, nbr_az, endpoint = False)
        rad_azs = np.radians(deg_azs)
        n_sections = len(self.rotor.sections)
        data_list = []

        with mp.Pool(processes = mp.cpu_count()) as pool :
            for i in range (len(yaws)) :
                yaw = np.radians(yaws[i])
                tsr = tsrs[i]
                self.set_wind(tsr)                  
            
                args = [
                    (az, yaw, tsr, self, j)
                    for az in rad_azs
                    for j in range(n_sections)
                ]

                #data = [job(*a) for a in args]
                data = pool.starmap(job, args)
                data_list.extend(data)
        df = pd.DataFrame(data_list)

        if export != None :
            export.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(export, index = False, sep = ',')
            print(f"Données calculées pour yaws == [{yaws}] et tsrs == [{tsrs}].\nEnregistrées à l'adresse {export}")

        return df

## Utilitaires pour la classe SimEnv

def easy_plot(df, attr_x:str, attr_y:str, yaw_index:int = 10, indexs:list = [4, 17, 30], savefig = False) :
    print(f"\nAffichage de {attr_y} en fonction de {attr_x}\n")

    if attr_y not in ['Fn', 'Ft', 'V_eff', 'Alpha_deg'] :
        raise ValueError(f"L'attribut {attr_y} n'est pas supporté pour l'affichage en ordonnée. Choisissez un dans ['Fn', 'Ft', 'V_eff', 'Alpha_deg'].")

    fig_dir = p.Path(p.Path(os.path.join(os.path.dirname(__file__), 'figs'))).mkdir(exist_ok = True)
    
    nbr_r = df['r'].unique().shape[0]
    nbr_a = df['theta'].unique().shape[0]
    yaw_iter = nbr_r*nbr_a
    
    if attr_x == 'r' :
        """
        Dans ce cas, la liste indexs contient les indices des azimuts d'intérêts.
        Plot la distribution de la variable attr_y sur la pale. 
        """

        x = df[attr_x].unique()
        fig = plt.figure(figsize = (20,12))
        
        for i in range(len(indexs)) :
            plt.subplot(len(indexs), 1, i+1)

            y = df[attr_y][yaw_iter*yaw_index + i*x.shape[0]: yaw_iter*(yaw_index-1) + (i+1)*x.shape[0]].values
            plt.plot(x, y, lw = 2.5)
            plt.xlabel(attr_x)
            plt.ylabel(attr_y)
            plt.title(f"Azimut : {indexs[i]}°")
            plt.grid()
    
    elif attr_x == 'theta' :
        """
        Dans ce cas, la liste indexs contient l'indice des rayons d'intérêts. 
        Plot la varibale attr_y en fonction des azimuts (sur [0,360]).
        """
        
        x = df[attr_x].unique()
        nbr_az = x.shape[0]
        fig = plt.figure(figsize = (18,12))

        for i in range(len(indexs)) :
            plt.subplot(len(indexs),1, i+1)

            ## Récupérer les données au bon endroit dans le DataFrame
            y = np.zeros((nbr_az))
            for j in range(nbr_az) : 
                y[j] = df[attr_y].values[(yaw_index-1)*yaw_iter + indexs[i] + nbr_az*j]

            plt.plot(x,y, lw = 2.5)
            plt.xlabel(attr_x)
            plt.ylabel(attr_y)
            plt.title(f"Section de pale : {df['r'].to_numpy()[indexs[i]].round(2)} (d'indice {indexs[i]})")
            plt.grid()
    else : 
        raise ValueError(f"L'attribut {attr_x} n'est par supporté. Essayer avec un des attributs de ['r', 'theta'].")    

    if savefig == True : 
        fig_name = attr_x +'_'+attr_y
        for i in range(len(indexs)) : 
            fig_name += ('_'+str(indexs[i])+'_')
            fig.savefif(fig_dir/fig_name)
    plt.subplots_adjust(hspace = 0.3)
    return fig
def compute_inflow_aoa(solver, Ux, Uy, angle):
        uxRelative = Ux * (1.0 - solver._axial_induction)
        uthetaRelative = Uy * (1.0 + solver._tangential_induction)
        inflowAngle = np.arctan2(uxRelative, uthetaRelative)

        attackAngle = inflowAngle - angle

        return inflowAngle, attackAngle
def job(az, yaw, tsr, env:SimEnv, j) :
    velocities = bem.tools.calculateVelocity(wind = env.U, omega = env.omega, 
                                                         rad = env.rotor.sections[j].radius, azi = az,
                                                         yaw = yaw, tilt = env.tilt,
                                                         precone = env.precone) 
    fn,ft,ai,at = env.solver.solve(env.rotor.sections[j], az, pitch = env.rotor.pitchRated,
                                                velocity = velocities, angles = [yaw, env.tilt])
    angle = env.rotor.sections[j].twist + env.rotor.pitchRated
    _, aoa = compute_inflow_aoa(env.solver, velocities[0], velocities[1], angle)
    V_eff = np.sqrt((env.U*(1-ai))**2 + (env.omega*env.rotor.sections[j].radius*(1+at))**2)                             

    dict_data = {
                'yaw': np.degrees(yaw),
                'TSR' : tsr,
                'r': env.rotor.sections[j].radius,
                'theta': np.degrees(az),  
                'Fn': fn,
                'Ft': ft,                    
                'V_eff': V_eff,
                'Alpha_deg': np.degrees(aoa)
                }
    return dict_data

## Calcul de la puissance à partir des sorties BEM

def contribution(SimEnv:SimEnv, azimuth:float, pos:int) :
    """
    Implémentation de la fonction de calcul des contributions (à azimut fixé, intégrale sur la pale). 
    Se référer à https://who.rocq.inria.fr/Julien.Salomon/docs/Article_BEM.pdf , section 5, eq (5.1)
    pour plus de détails.

    Input : 
        - SimEnv : paramètres de la simulation
        - azimuth : azimut pour lequel on calcule la contribution
        - pos : postion de la section de pale sur laquelle on calcule la contribution
    Output : 
        - Calcul direct de la formule (5.1)
    """

    velocity = bem.tools.calculateVelocity(
        wind = SimEnv.U, omega = SimEnv.omega, rad = SimEnv.rotor.sections[pos].radius,
        yaw = SimEnv.yaw, tilt =  SimEnv.tilt, precone = SimEnv.precone
    )
    phi, ai, at = SimEnv.solver.inductions(SimEnv.solver.sections[pos], azimuth=azimuth, 
                              velocity = velocity, angles = [SimEnv.yaw, 0.0],
                              tStep = 0.0)
    
    hub_tip_loss = bem.secondary.hubTipLoss.Prandtl(SimEnv.rotor.sections[pos].radius, 
                                                    nBlades = 3, hubRadius = SimEnv.rotor[0].radius, tipRadius = SimEnv.rotor[-1].radius, 
                                                    inflow = phi)
    
    loc_tsr = SimEnv.rotor.sections[pos].radius*SimEnv.omega/SimEnv.U  
    TSR = SimEnv.rotor.sections[-1].radius*SimEnv.omega/SimEnv.U  


    loc_twist = SimEnv.rotor.sections[pos].twist

    Cl = SimEnv.solver._funLift
    Cd = SimEnv.solver._funDrag

    return ((8*hub_tip_loss*loc_tsr**3)/TSR) * at*(1-ai) * (1 - (Cd(phi - loc_twist)/Cl(phi-loc_twist))*1/np.tan(phi))
def compute_power(SimEnv:SimEnv, azimuth:float) :
    """
    Calcul de Cp (coefficient de puissance) en intégrant selon le rayon 
    entre le nez et le bout de pale (à azimut fixé).
    La méthode d'intégration mobilisée est celle des points milieux.

    Input : 
        - SimEnv : objet contenant les paramètres de la simulation
        - azimuth : angle azimutale sur lequel la puissance est calculée

    Output :
        - Coefficient de puissance sur la pale Cp

    """

    sections = SimEnv.rotor.sections
    integral = 0
    for i in range(len(sections)-1):
        mid_pow = 0.5*(contribution(SimEnv = SimEnv, azimuth = azimuth, pos = i+1) + contribution(SimEnv = SimEnv, azimuth = azimuth, pos = i))
        integral += (sections[i+1]-sections[i])*mid_pow

    return integral*SimEnv.omega/SimEnv.U    
def compute_total_power(SimEnv:SimEnv, azimuths:list) :
    """
    Fonction qui calcule la puissance pour tous les azimuths de la révolution de l'éolienne.
    Cette puissance totale est calculé comme la somme des puissances sur chaque azimuts.

    Input : 
        - SimEnv : objet contenant les paramètres de la simulation
        - azimuths : liste des angles azimutaux sur lesquels la puissance est calculée
    OutPut : 
        Puissance généré par une révolution
    """

    power = 0
    for az in azimuths : 
        power += compute_power(SimEnv, azimuth = az)

    return power

## TODO : Implémenter divers tests unitaires pour détecter des incohérences dans les données