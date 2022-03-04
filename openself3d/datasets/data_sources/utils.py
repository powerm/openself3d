from __future__ import print_function

import yaml 
import numpy as np 
import os 
import sys 
from PIL  import Image 

from .transformations   import  quaternion_matrix, quaternion_from_matrix

import mmcv



def getPaddedString(idx, width=6):
    return str(idx).zfill(width)

def load_rgb_image(rgb_filename):
    """
    Returns PIL.Image.Image
    :param rgb_filename:
    :type rgb_filename:
    :return:
    :rtype: PIL.Image.Image
    """
    return Image.open(rgb_filename).convert('RGB')

class CameraIntrinsics(object):
    """
    Useful class for wrapping camera intrinsics and loading them from a
    camera_info.yaml file
    """
    def __init__(self, cx, cy, fx, fy, width, height):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.width = width
        self.height = height

        self.K = self.get_camera_matrix()

    def get_camera_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0,0,1]])

    @staticmethod
    def from_yaml_file(filename):
        config = mmcv.load(filename)

        fx = config['camera_matrix']['data'][0]
        cx = config['camera_matrix']['data'][2]

        fy = config['camera_matrix']['data'][4]
        cy = config['camera_matrix']['data'][5]

        width = config['image_width']
        height = config['image_height']

        return CameraIntrinsics(cx, cy, fx, fy, width, height)



def getQuaternionFromDict(d):
    """
    Get the quaternion from a dict describing a transform. The dict entry could be
    one of orientation, rotation, quaternion depending on the convention
    """
    quat = None
    quatNames = ['orientation', 'rotation', 'quaternion']
    for name in quatNames:
        if name in d:
            quat = d[name]


    if quat is None:
        raise ValueError("Error when trying to extract quaternion from dict, your dict doesn't contain a key in ['orientation', 'rotation', 'quaternion']")

    return quat


def homogenous_transform_from_dict(d):
    """
    Returns a transform from a standard encoding in dict format
    :param d:
    :return:
    """
    pos = [0]*3
    pos[0] = d['translation']['x']
    pos[1] = d['translation']['y']
    pos[2] = d['translation']['z']

    quatDict = getQuaternionFromDict(d)
    quat = [0]*4
    quat[0] = quatDict['w']
    quat[1] = quatDict['x']
    quat[2] = quatDict['y']
    quat[3] = quatDict['z']

    transform_matrix = quaternion_matrix(quat)
    transform_matrix[0:3,3] = np.array(pos)

    return transform_matrix


def compute_distance_between_poses(pose_a, pose_b):
    """
    Computes the linear difference between pose_a and pose_b
    

    Args:
        pose_a (numpy): 4x4 homogeneous transform
        pose_b (numpy): 4x4 homogeneous transform
    return:
        type: 
            Distance between translation component of the poses
    """
    
    pos_a = pose_a[0:3, 3]
    pos_b = pose_b[0:3, 3]
    
    return np.linalg.norm(pos_a - pos_b)

def compute_angle_between_quaternions(q, r):
    """
    Computes the angle between two quaternions.
    
    theta = arccos(2* <q1, q2>^2 -1)

    Args:
        q ([type]): [description]
        r ([type]): [description]
    
    return: 
        angle beween the quaternions, in radians
    """
    
    theta = 2*np.arccos(2 * np.dot(q,r)**2 - 1)
    return theta


def compute_angle_between_poses(pose_a, pose_b):
    """
    Computes the angle distance in radians between two homogenous transforms
    :param pose_a: 4 x 4 homogeneous transform
    :type pose_a:
    :param pose_b:
    :type pose_b:
    :return: Angle between poses in radians
    :rtype:
    """

    quat_a = quaternion_from_matrix(pose_a)
    quat_b = quaternion_from_matrix(pose_b)

    return compute_angle_between_quaternions(quat_a, quat_b)