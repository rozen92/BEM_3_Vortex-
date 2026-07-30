
import numpy as np




class Dummy:
    """Empty skewed wake correction model."""
    
    def __init__(self) -> None:
        return

    def __call__(self,axialInduction,*args,**kwargs):
        return axialInduction


class PittAndPeters:
    """Pitt & Peters skewed wake model.
    
    Pitt & Peters 1980, Theoretical prediction of dynamic-inflow derivatives.
    6th European Rotorcraft & Powered Lift Aircraft Forum 
    
    Snel & Schepers 1995, Joint investigation of dynamic inflow effects and
    implementation of an engineering method. Technical Report ECN-C–94-107,
    Energy Research Centre of the Netherlands (ECN),
    publications.ecn.nl/E/1995/ECN-C--94-107

    """

    def __init__(self,factor=15.*np.pi/64.) -> None:
        self.factor = factor
    
    def __call__(self,axialInduction,wakeSkewAngle,azimuthAngle,radius,hubRadius,tipRadius):
        return axialInduction*(
            1. + self.factor*np.tan(wakeSkewAngle/2.)*radius/tipRadius*np.sin(azimuthAngle)
            )


class IFPEN:
    """IFPEN skewed wake model.
    
    Blondel et al. 2017, Improving a BEM yaw model based on NewMexico
    experimental data and Vortex/CFD simulations.
    CFM 2017 - 23ème Congrès Français de Mécanique
    ifp.hal.science/hal-01663643 
    
    """

    def __init__(self,A0=0.35,Phi1=-np.pi/9,Phi2=np.pi) -> None:
        self.A0 = A0
        self.Phi1 = Phi1
        self.Phi2 = Phi2
    
    def __call__(self,axialInduction,wakeSkewAngle,azimuthAngle,radius,hubRadius,tipRadius):
        k1 = (1 - self.A0) + self.A0*(radius - hubRadius)/(tipRadius - hubRadius)
        k2 = 1 - self.A0*(radius - hubRadius)/(tipRadius - hubRadius)
        eta = radius/tipRadius
        return axialInduction*(
            1. \
            + k1*eta*np.tan(wakeSkewAngle/2.)*np.sin(azimuthAngle + self.Phi1) \
            + k2*(1 - eta)*np.tan(wakeSkewAngle/2.)*np.sin(azimuthAngle + self.Phi2)
            )
    