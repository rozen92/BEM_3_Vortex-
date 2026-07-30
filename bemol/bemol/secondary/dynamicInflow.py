


class Dummy:
    """Empty dynamic inflow model."""

    def __init__(self) -> None:
        return

    def __call__(self,axialInduction,*args,**kwargs):
        return axialInduction


class Knudsen:
    """Knudsen dynamic inflow model.
    
    T. Knudsen and T. Bak, "Simple model for describing and estimating wind
    turbine dynamic inflow," 2013 American Control Conference,
    Washington, DC, USA, 2013, pp. 640-646, doi: 10.1109/ACC.2013.6579909.
    
    """

    def __init__(self,alphaDynamic=0.3,tauScale=3.0) -> None:
        self._alpha0 = alphaDynamic
        self.alphaDynamic = alphaDynamic
        self.tauScale = tauScale

    def __call__(self,axialInduction,Ux,radius,tStep):
        if (tStep == 0.):
            raise ValueError(
                f'Using a dynamic inflow model with tStep = {tStep} does not make sense!'
                )
        # lower bound for wind velocity
        kappa = 1. / (self.tauScale * radius / max(1.0,Ux) )
        alphaDynamic = tStep*kappa*(axialInduction - self.alphaDynamic) + self.alphaDynamic
        self.alphaDynamic = alphaDynamic

        return alphaDynamic
    
    def restart(self):
        """Restart alpha to the starting value."""
        self.alphaDynamic = self._alpha0