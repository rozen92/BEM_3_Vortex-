

import numpy as np
from .. import bem




class Dummy:
    """Empty high-induction correction model.
    
    Calls the base thrust coefficient calculation method.
    """
    
    def __init__(self) -> None:
        return

    def __call__(self,axialInduction,lossFactor,skewAngle,*args,**kwargs):
        return bem.BaseBEM.CT(axialInduction,lossFactor,skewAngle)


class Buhl:
    """Buhl's empirical high-induction correction model.
    
    Buhl, Jr, M L. "New Empirical Relationship between Thrust Coefficient and
    Induction Factor for the Turbulent Windmill State.", Aug. 2005.
    https://doi.org/10.2172/15016819
    
    """

    def __init__(self,beta:float=0.4,da:float=0.02) -> None:
        self.beta = beta
        self.da = da
    
    def __call__(self,axialInduction,lossFactor,skewAngle):
        f0 = bem.BaseBEM.CT(self.beta,lossFactor,skewAngle)
        fp0 = (
            bem.BaseBEM.CT(self.beta + self.da,lossFactor,skewAngle)
            - bem.BaseBEM.CT(self.beta - self.da,lossFactor,skewAngle)
            ) / (2.*self.da)
        f1 = 2.*np.cos(skewAngle)
        k2 = (f1-f0-fp0*(1.0 - self.beta)) / (1.0 - self.beta)**2.
        k1 = fp0 - 2.*k2*self.beta
        k0 = f1 - k1 - k2

        return k0 + k1*axialInduction + k2*axialInduction**2.