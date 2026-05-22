import numpy as np
import copy
from scipy.spatial.transform import Rotation


def quaternion_from_rotation(omega):
    """
    Convert an angular displacement vector omega to a scipy Rotation.
    Equivalent to a rotation of |omega| radians about omega/|omega|.
    """
    omega_norm = np.linalg.norm(omega)
    if omega_norm > 1.0e-10:
        return Rotation.from_rotvec(omega)
    return Rotation.identity()


class Body(object):
    '''
    Small class to handle a single rigid body.
    '''

    def __init__(self, location, orientation=None):
        '''
        Parameters
        ----------
        location    : array-like, shape (3,) — body centre position
        orientation : scipy Rotation, defaults to identity
        '''
        self.location     = np.array(location, dtype=np.float64)
        self.location_new = np.copy(self.location)
        self.location_old = np.copy(self.location)

        ori = orientation if orientation is not None else Rotation.identity()
        self.orientation     = copy.copy(ori)
        self.orientation_new = copy.copy(ori)
        self.orientation_old = copy.copy(ori)

    def update(self, velocity, omega_dt, target='current'):
        '''
        Update location and orientation by a translational step and an
        angular displacement vector omega_dt = omega * dt.

        Parameters
        ----------
        velocity  : array-like, shape (3,)
        omega_dt  : array-like, shape (3,)
        target    : 'current' | 'new' | 'old'
        '''
        dq = quaternion_from_rotation(omega_dt)
        if target == 'new':
            self.location_new    += velocity
            self.orientation_new  = dq * self.orientation_new
        elif target == 'old':
            self.location_old    += velocity
            self.orientation_old  = dq * self.orientation_old
        else:
            self.location    += velocity
            self.orientation  = dq * self.orientation