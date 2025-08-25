# ArUcoROSpy
Detect a set of aruco markers to estimate a pose of an object. Transforms between markers can be found or provided in a form of a .npz file. The .npz file should contain a dictionary with a transforms (4x4 matrix) of each marker to the main marker

##Detecting 36 ArUco Markers at once
[![36arucomarkers](https://user-images.githubusercontent.com/69029439/145345616-5071700a-db01-4aee-aae0-50d06c755227.png)](https://www.youtube.com/watch?v=2GhBSx3AGbs)

### Architecture

![MarkerDetectionSchematic](https://user-images.githubusercontent.com/69029439/145345568-1a678c56-5e92-44ff-8517-d279841cc42a.jpg)
 

### Prerequisites (Installations and Dependencies)

- Intel Realsense
    
    [**ROS Releases**](https://github.com/IntelRealSense/realsense-ros)
    
    [**Realsense SDK - for realsense-viewer**](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
    
    
- Azure Kinect
    
    [**Azure Kinect ROS driver**](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#Installation)
    
      
[opencv-contrib-python] cv2 does not have an attribute "aruco"
        
[Python 3 cv2 Has No Attribute Aruco](https://winkdoubleguns.com/2021/02/13/python-3-cv2-has-no-attribute-aruco/)
        
[cv_bridge](https://wiki.ros.org/cv_bridge)
        
        

### Dependencies info

- [Intel Realsense ROS](http://wiki.ros.org/realsense_camera)
- [Azure Kinect ROS Topic](https://github.com/microsoft/Azure_Kinect_ROS_Driver/blob/melodic/docs/usage.md)
- [Opencv ArUco](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gab9159aa69250d8d3642593e508cb6baa)
- [cv_bridge](https://wiki.ros.org/cv_bridge)



        
### Running the Code
There are 2 ways this package may be used. 
Either as a service that calculates the position of the object only when asked or a node that continuously updates the position of the object in the tf tree.

Before running any of the versions above the transformation matrix between markers has to be "calibrated".

## Run the camera code in ROS (If using realsense or kinect)
- Intel Realsense ROS
    
    **Run the built-in Intel Realsense program to see the camera's current state**
    
    ```bash
    realsense-viewer
    ```
    
    **Run the launch files**
    
    ```bash
    roslaunch realsense2_camera rs_camera.launch
    ```
    
    
- Azure Kinect ROS
    
    **Run the built-in Azure Kinect program to see the camera's current state**
    
    ```bash
    k4aviewer
    ```
    
    **Run the launch files**
    
    ```bash
    roslaunch azure_kinect_ros_driver kinect_rgbd.launch
    ```
## 1. Determine Aruco marker transforms
First open "aruco_calibrate.launch" file and set the appropriate parameters.
Run the following command. For the next 60 seconds ros will try to calculate the transformations between markers. Try to capture all markers from as many angles as possible. At least two markers should be seen in each frame.
```bash
roslaunch aruco_detect aruco_calibrate.launch
```

## 2. Run service or node
In the launch file the following parameters may be set:
- aruco_type {str}: Type of the aruco marker (default: "DICT_6X6_100")
- aruco_size {double}: Size of the aruco markers in meters (default: "0.1")
- aruco_transforms {str}: A path to the .npz file containing the transforms between markers. If not provided calibration needs to be done at launch.
- aruco_update_rate {double}: A value between 0 and 1. Determines how much each frame updates the position of the object (running average)
- aruco_object_id {str}: The name the detected object will be given in the tf tree.
- aruco_main_marker_id {int}: The ID of the "0" aruco marker. All transforms are calculated to this marker. The object position is the location of this marker.
- camera_img_topic {str}: The name of the ros topic where camera images are posted.
- camera_info_topic {str}: The name of the ros topic where the information about the camera is posted (camera matrix, distortion matrix)
- camera_frame_id {str}: The id of the frame in which the Image is posted.

To run the node:
```bash
roslaunch aruco_detect arucode_node.launch
```
To run as a service:
```bash
roslaunch aruco_detect arucode_service.launch
```


    
    
    
