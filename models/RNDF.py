"""
@Email: yiting.chen@rice.edu
"""
import os
import sys

sys.path.append('../')
import torch
import numpy as np
from utils import robot_kinematic
import trimesh
from torch.autograd.functional import jacobian
from .networks import RobotNDF


class iiwa_RNDF:
    def __init__(self,
                 use_GPU=False,
                 model_weights="weight/128_params.pth",
                 enable_gradient=True) -> None:

        # initialize model
        print("Loading neural network models ...")
        self.device = "cuda:0" if (torch.cuda.is_available() and use_GPU) else "cpu"
        self.model = RobotNDF(N=128)
        self.model.load_model(self._get_file_path(model_weights))
        self.enable_gradient = enable_gradient

        print("Initializing Robot Model ...")
        self.robot = robot_kinematic()

    def _get_file_path(self, filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    def set_requires_grad(self, requires_grad=True):
        assert self.model is not None, "Model initialization failed."
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def sample_random_robot_config(self) -> np.ndarray:
        return self.robot.sample_random_robot_config()

    def show_robot(self, convex=False, bounding_box=False) -> None:
        self.robot.show_robot_meshes(convex, bounding_box)

    def calculate_signed_distance(self, points: list, min_dist=False) -> np.ndarray:
        """ computation for batch input points
        :param min_dist: return minimum distance to the whole robot, corresponding to the maximum signed distance
        :param position: (n, 3) or (3, )
        :return: pred values (n, )
        """
        assert self.model is not None
        points = np.array(points)
        if points.shape == (3,):
            points = np.expand_dims(points, axis=0)

        data_len = points.shape[0]

        q = torch.from_numpy(np.array([self.robot.robot_q]))
        nq = q.repeat(data_len, 1)
        qp_tensor = torch.concat((nq, torch.from_numpy(points)), dim=1).type(torch.float32).to(self.device)

        pred = self.model(qp_tensor)
        pred = pred.cpu().detach().numpy()

        if not min_dist:
            return pred
        else:
            max_index = pred.argmin(axis=1)
            pred_min_dist = pred[np.arange(data_len), max_index]
            return pred_min_dist

    def calculate_gradient(self, points: list):
        """ get gradient wrt input of the implicit function for batch input joint configuration and positions
        :param position: (n, 3) or (3, )
        :return: pred values (n, 10)
        """
        assert self.model is not None
        points = np.array(points)
        if points.shape == (3,):
            points = np.expand_dims(points, axis=0)

        data_len = points.shape[0]

        q = torch.from_numpy(np.array([self.robot.robot_q])).to(self.device)
        nq = q.repeat(data_len, 1).to(self.device)

        qp_tensor = torch.concat((nq, torch.from_numpy(points)), dim=1).type(torch.float32)
        qp_tensor.requires_grad = True

        gradient = jacobian(self.model, qp_tensor)
        gradient = torch.sum(gradient, dim=2)

        return gradient

    def set_robot_joint_positions(self, q) -> None:
        assert len(q) == 7
        self.robot.set_robot_joints(q)

    def show_robot_with_points(self, points, point_radius=0.02) -> None:
        points = np.array(points)
        if points.shape == (3,):
            points = np.expand_dims(points, axis=0)

        robot_mesh = self.robot.get_combined_mesh(convex=False, bounding_box=False)
        scene = trimesh.Scene()
        for point in points:
            rand_color = np.random.uniform(size=4)
            rand_color[3] = 1
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=point_radius,
                                                color=rand_color)
            sphere.apply_translation(point)
            scene.add_geometry(sphere)

        scene.add_geometry(robot_mesh)
        scene.show()