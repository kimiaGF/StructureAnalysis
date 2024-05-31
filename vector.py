#%%
import numpy as np
import logging 
from pyquaternion import Quaternion

# %% Define logging configurations
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(lineno)s:%(message)s')

file_handler = logging.FileHandler('vector.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

#%%

def get_angle(a,b,deg=True):
    a = np.array(a)
    b = np.array(b)
    mag_a = np.sqrt(np.sum(a**2))
    mag_b = np.sqrt(np.sum(b**2))
    theta = np.arccos(np.dot(a,b)/(mag_a*mag_b))
    if deg:
        return theta*180/np.pi
    else:
        return theta
def direction_cosines(vector):
    """
    Calculate the direction cosines of a vector.

    Parameters:
        vector (numpy array): The vector for which direction cosines are to be calculated.

    Returns:
        cosines (numpy array): An array containing the direction cosines of the vector.
    """
    # Normalize the vector to obtain a unit vector
    unit_vector = vector / np.linalg.norm(vector)
    
    # Calculate the direction cosines by taking the dot product with each coordinate axis
    cosines = np.array([np.dot(unit_vector, axis) for axis in np.eye(len(vector))])
    
    return cosines

def unit(a):
    mag_a = np.sqrt(np.sum(a**2))
    return a/mag_a

def rotate_vectors(vectors, axis, angle,deg=True):
    """
    Rotate a list of vectors around a specified axis by a given angle.

    Parameters:
        vectors (list of numpy arrays): List of vectors to be rotated.
        axis (numpy array): Axis of rotation. Must be a unit vector.
        angle (float): Angle of rotation in radians.

    Returns:
        rotated_vectors (list of numpy arrays): Rotated vectors.
    """
    if deg:
        angle_r = angle*np.pi/180
    else:
        angle_r = angle
    # Convert axis to unit vector if it's not already
    q = Quaternion(axis=axis,angle=angle_r)
    vector_prime = []
    for i in vectors:
        vector_prime.append(q.rotate(i))
    return vector_prime