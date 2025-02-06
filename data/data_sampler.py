"""
@Email: yiting.chen@rice.edu
"""
import sys
sys.path.append("../")
import time
import copy
import trimesh
import numpy as np
from utils import robot_kinematic


class data_sampler(robot_kinematic):
    def __init__(self, urdf_path=None, dataset_path=None):
        super().__init__(urdf_path)
        self.dataset_path = dataset_path

    def get_link_mesh(self, link_name, joint_positions=None):
        assert link_name in self.link_names
        if joint_positions is None:
            fk = self.robot.link_fk(cfg=self.robot_joints)
        else:
            robot_config = copy.deepcopy(self.robot_joints)
            for i, name in self.link_names:
                robot_config[name] = joint_positions[i]
            fk = self.robot.link_fk(cfg=self.robot_joints)

        robot_meshes = self.robot_links_mesh
        mesh = robot_meshes[link_name].copy()
        link_index = self.link_names.index(link_name)
        mesh = mesh.apply_transform(fk[self.robot.links[link_index]])
        return mesh

    def set_robot_joints(self, positions):
        assert len(positions) == self.num_joints
        for i in range(self.num_joints):
            assert (self.joint_lower_bound[i] < positions[i]) & (positions[i] < self.joint_upper_bound[i])

        for i in range(self.num_joints):
            self._robot_joints[self.joint_names[i]] = positions[i]

    def whole_arm_normal_sampling(self,
                                  base_num=10,
                                  offset_range=None,
                                  joint_positions=None):

        if offset_range is None:
            offset_range = [0, 0.03]

        if joint_positions is not None:
            self.set_robot_joints(joint_positions)

        sampled_points = []
        for i, name in enumerate(self.link_names):
            link_mesh = self.get_link_mesh(name)
            points = self.random_range_normal_sampling(link_mesh,
                                                       offset=offset_range,
                                                       num_points=base_num)
            sampled_points.append(points)
        return np.vstack(sampled_points)

    def whole_arm_inside_sampling(self,
                                  base_num=5,
                                  joint_positions=None,
                                  link_weights=None):
        # trimesh uses a rejection-based sampling method
        # mesh with intricate geometry needs to sample more
        if link_weights is None:
            link_weights = [1, 1, 1, 1, 1, 3, 1, 1]
        assert len(link_weights) == self.num_links

        if joint_positions is not None:
            self.set_robot_joints(joint_positions)
        sampled_points = []
        for i, name in enumerate(self.link_names):
            link_mesh = self.get_link_mesh(name)

            points = self.random_sample_inside_mesh(link_mesh, num_points=int(base_num * link_weights[i]))
            sampled_points.append(points)

        return np.vstack(sampled_points)


    def batch_calculate_signed_distance(self, sampled_points):
        signed_distance_links = []
        for name in self.link_names:
            link_mesh = self.get_link_mesh(name)
            signed_distance = trimesh.proximity.signed_distance(link_mesh, sampled_points)
            signed_distance_links.append(signed_distance)

        return np.asarray(signed_distance_links).transpose()

    def batch_sample_inside_mesh(self, batch_size, base_num, link_weights=None):
        iter_time = 0
        batch_data = []
        while iter_time < batch_size:
            rand_q = self.sample_random_robot_config()
            self.set_robot_joints(rand_q)
            if self.self_collision_detected():
                continue
            sample_points = self.whole_arm_inside_sampling(base_num=base_num, link_weights=link_weights)
            signed_dist = self.batch_calculate_signed_distance(sample_points)
            rand_q = np.asarray([rand_q for _ in range(len(sample_points))])

            iter_data = np.concatenate((rand_q, sample_points, signed_dist), axis=1)
            batch_data.append(iter_data)
            iter_time += 1
            if iter_time % 20 == 0:
                print("Data Generation {} with base_num: {} progress {}/{}".format(iter_data.shape,
                                                                                   base_num,
                                                                                   iter_time,
                                                                                   batch_size))
        # combined data and define data type
        batch_data = np.vstack(batch_data).astype(np.float32)
        print("Saving generated data with shape of {}".format(batch_data.shape))
        np.save(self.dataset_path + '/inside/Inside_{}_{}'.format(str(time.time())[-4:],
                                                                      batch_data.shape[0]),
                batch_data)

    def batch_sample_outside_mesh(self, batch_size, base_num, offset_range=None):

        if offset_range is None:
            offset_range = [0, 0.03]

        iter_time = 0
        batch_data = []
        while iter_time < batch_size:
            rand_q = self.sample_random_robot_config()
            self.set_robot_joints(rand_q)
            if self.self_collision_detected():
                continue
            sample_points = self.whole_arm_normal_sampling(base_num=base_num, offset_range=offset_range)
            signed_dist = self.batch_calculate_signed_distance(sample_points)
            rand_q = np.asarray([rand_q for _ in range(len(sample_points))])
            iter_data = np.concatenate((rand_q, sample_points, signed_dist), axis=1)
            batch_data.append(iter_data)
            iter_time += 1
            if iter_time % 50 == 0:
                print("Data Generation {} with base_num: {} progress {}/{}".format(iter_data.shape,
                                                                                   base_num,
                                                                                   iter_time,
                                                                                   batch_size))
        # combined data and define data type
        batch_data = np.vstack(batch_data).astype(np.float32)
        print("Saving generated data with shape of {}".format(batch_data.shape))
        np.save(self.dataset_path + '/outside/Outside_{}_{}_{}_{}'.format(str(time.time())[-4:],
                                                                              offset_range[0],
                                                                              offset_range[1],
                                                                              batch_data.shape[0]),
                batch_data)

    def random_sample_inside_mesh(self, sampled_mesh, num_points=200):
        """
        sample inside points w.r.t. a given mesh
        rejection-based sampling
        """
        sampled_points = trimesh.sample.volume_mesh(sampled_mesh, count=num_points)
        return np.asarray(sampled_points)

    def random_range_normal_sampling(self, sampled_mesh, offset=None, num_points=3000):
        """
        sample outside points w.r.t. a given mesh
        random distance * normal vector
        """
        if offset is None:
            offset = [0.2, 1]
        vertices, faces = sampled_mesh.sample(num_points, return_index=True)
        normals = sampled_mesh.face_normals[faces]
        rand_dist = np.random.uniform(low=offset[0], high=offset[1], size=(len(vertices), 1))
        rand_dist = np.repeat(rand_dist, 3, axis=1)
        sampled_points = vertices - normals * rand_dist
        return sampled_points

    def create_point_cloud_scene(self, sampled_points, mesh, point_radius=0.002):
        scene = trimesh.Scene()
        for point in sampled_points:
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=point_radius, face_colors=np.random.uniform(size=4))
            sphere.apply_translation(point)
            scene.add_geometry(sphere)

        scene.add_geometry(mesh)
        return scene


if __name__ == "__main__":
    np.random.seed(16)
    robo = data_sampler(dataset_path='../dataset/')

    # random joint configuration
    sampled_q = robo.sample_random_robot_config()
    robo.set_robot_joints(sampled_q)

    # visualize sampled joint configuration
    robo.show_robot_meshes(convex=False, bounding_box=False)
    robo.show_robot_meshes(convex=False, bounding_box=True)

    print(robo.link_names)
    print(robo.joint_names)
    print("self-collision detected: {}".format(robo.self_collision_detected()))

    combined_mesh = robo.get_combined_mesh(convex=False, bounding_box=False)

    # sampled points outside
    outside_points = robo.whole_arm_normal_sampling(offset_range=[0.4, 0.5], base_num=5)
    scene_outside = robo.create_point_cloud_scene(outside_points, combined_mesh, point_radius=0.02)
    scene_outside.show()
    print("signed distance:", robo.batch_calculate_signed_distance(outside_points))

    # sampled points inside (well you may not see it without zooming in)
    inside_points = robo.whole_arm_inside_sampling(base_num=10)
    scene_inside = robo.create_point_cloud_scene(inside_points, combined_mesh, point_radius=0.02)
    scene_inside.show()
    print("signed distance:", robo.batch_calculate_signed_distance(inside_points))

    # batch sample outside
    robo.batch_sample_outside_mesh(batch_size=4, base_num=20, offset_range=[0., 0.1])

    # batch sample inside
    robo.batch_sample_inside_mesh(batch_size=4, base_num=20, link_weights=[1, 1, 1, 1, 1, 3, 1, 1])
