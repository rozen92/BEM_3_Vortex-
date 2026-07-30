
import inspect
from types import ModuleType

from . import secondary


class Corrections(object):
    """Class for storing corrections.

    Consider all the corrections available in the secondary
    models module. If not given by the user considers the base (empty)
    correction with the default parameters.

    Parameters
    ----------
    corrections : optional
        dictionary or list with the secondary corrections, either classes of
        instances - in case of custom parameters. If dictionary is
        given, the key must be the name of the correction as defined in
        secondary.py with a lower first letter.
    
    """
    def __init__(self,corrections:dict={}):

        for name_mod, mod in inspect.getmembers(secondary):
            if not isinstance(mod,ModuleType): continue
            effect_name = name_mod[0].lower() + name_mod[1:]
            for name_obj, obj in inspect.getmembers(mod):
                if inspect.isclass(obj):
                    if type(corrections) is dict:
                        if effect_name in corrections:
                            # instantiation of correction with default values
                            corr = corrections[effect_name]
                            corr = corr() if isinstance(corr,type) else corr
                            setattr(self,effect_name,corr)
                        else:
                            setattr(self,effect_name,mod.Dummy())
                    else:
                        setattr(self,effect_name,mod.Dummy())
                        # loop for all effects to check if any of the input
                        # corrections are classes of the available corrections
                        for corr in corrections:
                            # instantiation of correction with default values
                            # TODO: check name before instanciating!
                            corr = corr() if isinstance(corr,type) else corr
                            if effect_name in corr.__module__:
                                setattr(self,effect_name,corr)
                                break


    def __iter__(self):
        """Iterate corrections."""
        for value in self.__dict__.values():
            yield value
