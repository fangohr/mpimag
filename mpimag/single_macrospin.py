"""
=========================================================================
Single Macrospin Setup
=========================================================================

Setup of a for the single Macrospin.

The Macrospin object has the following properties associated with it:

=========================================================================
Property    | Description                       | Usage
=========================================================================
coord       | The coordinate of the Macrospin in| tuple (x,y)
            | space (2D cartesian coordinates)  |
-------------------------------------------------------------------------
            |                                   |
=========================================================================            


"""
import numpy as np


class Macrospin(object):
    """
    Single Macrospin setup class
    """
    def __init__(self):
        self._coord = None

    def _set_coord(self, coord):
        if type(coord) != tuple:
            raise ValueError('Expecting a 2D tuple')
        if np.shape(coord) != (2,):
            raise ValueError('Expecting a tuple of for (x,y)\
                              Tuple supplied is not of this\
                              form.')
        self._coord = coord

    def _get_coord(self):
        """
        Set the coordinate of the single Macrospin. Expects cartesian
        coordinates in the form of a 2D tuple, (x, y).
        """
        if self._coord is None:
            raise AttributeError('Coordinate not yet set')
        return self._coord

    coord = property(_get_coord, _set_coord)
