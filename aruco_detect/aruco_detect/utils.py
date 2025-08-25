from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_matrix(pose):
    """
    Converts a Pose message to a 4x4 numpy matrix.
    """
    r = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    rot_matrix = r.as_matrix()
    
    trans_matrix = np.identity(4)
    trans_matrix[:3, :3] = rot_matrix
    trans_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    
    return trans_matrix

def matrix_to_pose(matrix):
    """
    Converts a 4x4 numpy matrix to a Pose message.
    """
    r = R.from_matrix(matrix[:3, :3])
    quat = r.as_quat()
    
    pose = Pose()
    pose.position = Point(x=matrix[0, 3], y=matrix[1, 3], z=matrix[2, 3])
    pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    
    return pose

def matrix_to_quat_trans(matrix):
    """
    Convert a 4x4 numpy matrix to a quaternion and translation vector.
    """
    r = R.from_matrix(matrix[:3, :3])
    quat = r.as_quat()
    trans = matrix[:3, 3]
    return trans, quat

def quat_trans_to_matrix(trans, quat):
    """
    Convert a translation and quaternion vector to a matrix.
    """
    r = R.from_quat(quat)
    rot_matrix = r.as_matrix()
    
    matrix = np.identity(4)
    matrix[:3, :3] = rot_matrix
    matrix[:3, 3] = trans
    
    return matrix

def pose_to_quat_trans(pose):
    """
    Converts a Pose to a quaternion and translation vector
    """
    quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    trans = np.array([pose.position.x, pose.position.y, pose.position.z])
    return trans, quat

def quat_trans_to_pose(trans, quat):
    """
    Converts a quaternion and translation vector to a Pose.
    """
    pose = Pose()
    pose.position.x = float(trans[0])
    pose.position.y = float(trans[1])
    pose.position.z = float(trans[2])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose

def average_quaternions(quat_list, weights=None):
    """
    Average a list of quaternions.
    """
    if len(quat_list) == 0:
        return None
    if len(quat_list) == 1:
        return quat_list[0]
    
    # Using Slerp for weighted average, but a simple normalized linear interpolation (NLERP)
    # is often sufficient and more stable for small angle differences.
    # For a more robust solution, a proper mean on the SO(3) manifold should be calculated.
    # Here, we use a simplified approach by averaging and normalizing.
    
    if weights is None:
        weights = np.ones(len(quat_list))
    weights = np.array(weights) / np.sum(weights)

    avg_quat = np.zeros(4)
    for i, quat in enumerate(quat_list):
        # Ensure quaternions are in the same hemisphere
        if i > 0 and np.dot(quat_list[0], quat) < 0:
            quat = -quat
        avg_quat += weights[i] * quat
    
    avg_quat /= np.linalg.norm(avg_quat)
    return avg_quat
