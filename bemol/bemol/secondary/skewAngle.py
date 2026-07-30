

class Dummy:
    """Empty skewed wake model."""

    def __init__(self) -> None:
        return

    def __call__(self,axialInduction,*args,**kwargs):
        return axialInduction


class Burton:
    """Burton skewed wake model.
    
    Burton "Handbook of wind energy", 2001, page 105.

    """

    def __init__(self,) -> None:
        return

    def __call__(self,axialInduction,yawAngle):
        return (0.6 * axialInduction + 1.0) * yawAngle

