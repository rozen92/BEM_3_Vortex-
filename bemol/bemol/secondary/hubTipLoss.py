

import numpy as np

INV_TWO_PI = float(2.0/np.pi)



class Dummy:
    """Empty hub/tip loss correction model."""

    def __init__(self) -> None:
        return

    def __call__(self,*args,**kwargs):
        return 1.0


class Prandtl:
    """Prandt hub/tip loss correction model.
    
    Parameters
    ----------
    epsilon: float, optional
        minimal value of F to avoid errors close to/at the bounds of the
        blade. Default is 1e-12.
    
    """
    def __init__(self,epsilon=1e-12) -> None:
        self._epsilon = epsilon
        return

    def __call__(self,radius,nBlades,hubRadius,tipRadius,inflowAngle):

        fTip = nBlades/2.0*((tipRadius - radius) / (radius*np.abs(np.sin(inflowAngle))))
        fTip = INV_TWO_PI*np.arccos(np.exp(-fTip))

        fRoot = nBlades/2.0*((radius - hubRadius) / (hubRadius*np.abs(np.sin(inflowAngle))))
        fRoot = INV_TWO_PI*np.arccos(np.exp(-fRoot))

        return max(fTip*fRoot,self._epsilon)

