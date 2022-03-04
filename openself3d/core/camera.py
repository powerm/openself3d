import numpy  as np 
from enum import Enum


class PinholeCameraIntrinsicParameters(Enum):
    """enum PinholeCameraIntrinsicParameters
           Sets default camera intrinsic parameters for sensors.
    Args:
        Enum ([type]): [description]
    """
    # Default settings for PrimeSense camera sensor.
    PrimeSenseDefault=0
    # Default settings for Kinect2 depth camera.
    Kinect2DepthCameraDefault=1
    # Default settings for Kinect2 color camera.
    Kinect2ColorCameraDefault=2

class PinholeCameraIntrinsic(object):
    """PinholeCameraIntrinsic class stores intrinsic camera matrix, and image height and width.

    Args:
        object ([type]): [description]
    """
    def __init__(self):
        """Default Constructor.
        """
        super().__init__()
        self.__width_ = -1
        self.__height_ = -1
        self.__intrinsic_matrix = np.zeros((3,3), dtype= np.float32)
        
    def __init__(self, width, height,  fx, fy, cx, cy) -> None:
        """Parameterized Constructor.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            fx (float or double): focal length along the X-axis.
            fy (float or double): focal length along the Y-axis.
            cx (float or double): principal point of the X-axis.
            cy (float or double): principal point of the Y-axis.
        """
        super().__init__()
        self.set_Intrinsics(width, height, fx, fy, cx, cy)
        
    def __init__(self,  param):
        """Parameterized Constructor.

        Args:
            param (PinholeCameraIntrinsicParameters enum): Sets the camera parameters to the default settings of one of the sensors.
        """
        if param is PinholeCameraIntrinsicParameters.PrimeSenseDefault:
            self.set_Intrinsics (640, 480,  525.0,  525.0,  319.5,  239.5)
        elif param is PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault:
            self.set_Intrinsics(512, 424, 365.456, 365.456, 254.878, 205.395)
        elif param is PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault:
            self.set_Intrinsics(1920, 1080, 1059.9718, 1059.9718, 975.7193, 545.9533)
    
    def set_Intrinsics (self, width,  height,  fx,  fy,  cx,  cy):
        """Set camera intrinsic parameters.

        Args:
           width (int): Width of the image.
           height (int): Height of the image.
           fx (double): X-axis focal length
           fy (double): Y-axis focal length.
           cx (double): X-axis principle point.
           cy (double): Y-axis principle point.
        """
        self.__width =  width
        self.__height =  height
        self.__intrinsic_matrix = np.identity(3,dtype=np.float32)
        
        assert (fx>0.00000001 and fy>0.0000001 and cy>0.0000001)
        self.__intrinsic_matrix[0,0] = fx
        self.__intrinsic_matrix[1,1] = fy
        self.__intrinsic_matrix[0,2] = cx
        self.__intrinsic_matrix[1,2] = cy
      
    
    def get_focal_length(self):
        """Returns the focal length in a tuple of X-axis and Y-axis focal lengths.
        
        """
        return (self.__intrinsic_matrix[0, 0],  self.__intrinsic_matrix[1, 1])
    
    def  get_principal_point(self):
        """Returns the principle point in a tuple of X-axis and.Y-axis principle points
        """
        return (self.__intrinsic_matrix[0, 2] ,  self.__intrinsic_matrix[1, 2])
        
    def get_skew(self):
        """Returns the skew.
        
        """
        return self.__intrinsic_matrix[0, 1]
    
    def is_valid(self):
        """Returns True iff both the width and height are greater than 0.
        """
        return(self.__width>0 and self.__height>0)
    
    @property
    def height(self):
        return self.__height
    @height.setter
    def height(self, value):
        assert (isinstance(value,  int) and value>0)
        self.__height = value
       
    @property
    def width(self):
        return self.__width
    @width.setter
    def width(self, value):
        assert (isinstance(value,  int) and value>0)
        self.__width =  value
       
    @property
    def intrinsic_matrix(self):
        """Intrinsic camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            return type
                    3x3 numpy array
        """
        return self.__intrinsic_matrix
       
    @intrinsic_matrix.setter
    def  intrinsic_matrix(self, value):
        assert  isinstance(value, np.array) and value.shape == (3, 3)
        self.__intrinsic_matirx  = value
           

class PinholeCameraParameters(object):
    
    
    def __init__(self):
        """Default constructor
        """
        self.__intrinsic = PinholeCameraIntrinsic()
        self.__extrinsic = np.identity(4, dtype=np.float32)
        
    def __init__(self, param):
        """Copy constructor

        Args:
            arg0 (PinholeCameraParameters): [description]
        """
        assert  isinstance(param, PinholeCameraParameters)
        self.__intrinsic = param.__intrinsic
        self.__extrinsic = param.__extrinsic
    
    @property  
    def extrinsic(self):
        """Camera extrinsic parameters.
               4x4 numpy array
        """
        return self.__extrinsic
    
    @extrinsic.setter
    def extrinsic(self, value):
        assert( isinstance(value, np.array) and value.shape==(4, 4))
    
    @property
    def intrinsic(self):
        """PinholeCameraIntrinsic object.
        """
        return self.__intrinsic
        
    @intrinsic.setter
    def intrinsic(self, value):
        """PinholeCameraIntrinsic object.
        """
        assert  isinstance(value , PinholeCameraIntrinsic)
        
    
    
    class PinholeCameraTrajectory(object):
        
        def __init__(self) -> None:
            super().__init__()
            pass
        
        def __init__(self, trajectory) -> None:
            super().__init__()
            pass
        
        @property
        def parameters(self):
            pass
            
        
        @parameters.setter
        def parameters(self, params):
            pass
