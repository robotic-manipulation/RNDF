"""
@Email: yiting.chen@rice.edu
"""
import os
import sys
sys.path.append("../")
from urdfpy import URDF
import trimesh
from collections import OrderedDict
import copy
from omegaconf import OmegaConf
import numpy as np
import open3d as o3d


class robot_kinematic:
    def __init__(self, urdf_path=None):
        self.conf = OmegaConf.load(self._get_file_path('config/robot_info.yaml'))

        if urdf_path is None:
            urdf_path = self._get_file_path(self.conf.robot_urdf_path)

        self.robot = URDF.load(urdf_path)
        self._robot_joints = {}

        self._joint_upper_bound = np.array(self.conf.joint_upper_bound)
        self._joint_lower_bound = np.array(self.conf.joint_lower_bound)

        self.joint_names = []
        self.link_names = []

        self.num_joints = None
        self.num_links = None

        self.robot_links_mesh = OrderedDict()
        self.robot_links_convex_mesh = OrderedDict()
        self.init_robot_info()

    def _get_file_path(self, filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    def init_robot_info(self):
        for joint in self.robot.joints:
            print('{} connects {} to {}'.format(joint.name, joint.parent, joint.child))
            self._robot_joints[joint.name] = 0.
            if joint.parent not in self.link_names:
                self.link_names.append(joint.parent)
            if joint.child not in self.link_names:
                self.link_names.append(joint.child)

        self.joint_names = list(self._robot_joints.keys())
        self.num_joints = len(self.joint_names)
        self.num_links = len(self.link_names)
        meshes = self.robot.visual_trimesh_fk()
        for i in range(self.num_links):
            link_name = self.link_names[i]
            self.robot_links_mesh[link_name] = list(meshes.keys())[i].copy()
            self.robot_links_convex_mesh[link_name] = trimesh.convex.convex_hull(self.robot_links_mesh[link_name])

    def show_robot_meshes(self, convex=True, bounding_box=True):
        combined_meshes = self.get_combined_mesh(convex, bounding_box)
        combined_meshes.show()

    def get_combined_mesh(self, convex=False, bounding_box=False):
        if bool(self.robot_links_convex_mesh) is False:
            raise ValueError('Please init the robot first!')
        convex_meshes = []
        fk = self.robot.link_fk(cfg=self.robot_joints)

        if convex:
            robot_meshes = self.robot_links_convex_mesh
        else:
            robot_meshes = self.robot_links_mesh

        for i in range(self.num_links):
            name = self.link_names[i]

            # use deepcopy for not messing up the original mesh
            mesh = robot_meshes[name].copy()
            mesh = mesh.apply_transform(fk[self.robot.links[i]])
            if bounding_box:
                mesh = mesh.bounding_box_oriented
                # print(trimesh.bounds.corners(mesh.bounds))
            convex_meshes.append(mesh)
        combined_meshes = trimesh.util.concatenate(convex_meshes)
        return combined_meshes

    def get_link_mesh(self, link_name, joint_positions=None):
        assert link_name in self.link_names
        if joint_positions is None:
            fk = self.robot.visual_trimesh_fk(cfg=self.robot_joints, links=[link_name])
        else:
            robot_config = copy.deepcopy(self.robot_joints)
            for i, name in self.link_names:
                robot_config[name] = joint_positions[i]
            fk = self.robot.visual_trimesh_fk(cfg=self.robot_joints, links=[link_name])

        mesh = list(fk.keys())[0]
        mesh = mesh.copy()
        mesh = mesh.apply_transform(list(fk.values())[0])

        return mesh

    def self_collision_detected(self):
        collision_detector = trimesh.collision.CollisionManager()
        fk = self.robot.link_fk(cfg=self.robot_joints)
        robot_meshes = self.robot_links_mesh

        for i in range(self.num_links):
            name = self.link_names[i]
            link_mesh = robot_meshes[name].copy()
            collision_detector.add_object(name=name,
                                          mesh=link_mesh,
                                          transform=fk[self.robot.links[i]])

        _, collision_set = collision_detector.in_collision_internal(return_names=True)
        collided_pair = collision_set - self.link_connection_set
        if collided_pair:
            return True
        else:
            return False

    def set_robot_joints(self, positions):
        assert len(positions) == 7
        for i in range(self.num_joints):
            self._robot_joints[self.joint_names[i]] = positions[i]

    @staticmethod
    def trimesh2pcd(mesh, num_points, even_sample=False):
        if even_sample:
            sampled_points, _ = trimesh.sample.sample_surface_even(mesh=mesh,
                                                                   count=num_points)
        else:
            sampled_points, _ = trimesh.sample.sample_surface(mesh=mesh,
                                                              count=num_points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(sampled_points))
        pcd.remove_duplicated_points()
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        pcd.normalize_normals()
        pcd.orient_normals_towards_camera_location(pcd.get_center())

        return pcd

    def sample_random_robot_config(self):
        rand_jp = []

        for i in range(self.num_joints):
            rand_jp.append(np.random.uniform(self.joint_lower_bound[i],
                                             self.joint_upper_bound[i]))

        return np.asarray(rand_jp)

    @property
    def robot_joints(self) -> dict:
        return self._robot_joints

    @property
    def robot_q(self) -> list:
        q = []
        for name in self.joint_names:
            q.append(self.robot_joints[name])
        return q

    @property
    def joint_upper_bound(self):
        return self._joint_upper_bound

    @property
    def joint_lower_bound(self):
        return self._joint_lower_bound

    @property
    def link_connection_set(self):
        return {('lbr_iiwa_link_2', 'lbr_iiwa_link_3'),
                ('lbr_iiwa_link_4', 'lbr_iiwa_link_5'),
                ('lbr_iiwa_link_5', 'lbr_iiwa_link_6'),
                ('lbr_iiwa_link_3', 'lbr_iiwa_link_4'),
                ('lbr_iiwa_link_6', 'lbr_iiwa_link_7'),
                ('lbr_iiwa_link_0', 'lbr_iiwa_link_1'),
                ('lbr_iiwa_link_1', 'lbr_iiwa_link_2')}


if __name__ == "__main__":
    arm = robot_kinematic()
    q = arm.sample_random_robot_config()
    print(q)

    arm.set_robot_joints(q)
    mesh = arm.get_combined_mesh()
    mesh.show()
