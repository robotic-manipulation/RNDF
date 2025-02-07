import torch
from models import RobotNDF
from data import DataSampler

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # feature size of 128
    model = RobotNDF(N=128).to(device)
    model.load_model("models/weight/128_params.pth")

    # feature size of 64
    # model = RobotNDF(N=64).to(device)
    # model.load_model("models/weight/64_params.pth")

    # (3, 10), 3 (joint state + position) pairs
    random_qp = torch.tensor([[-1.64202379,  0.09702638,  0.3008685,  -1.9033781,  -0.82645173, -1.15995584, 1.15286252,
                              -0.1, 0.2, -0.3],
                              [-1.64202379, 0.09702638, 0.3008685, -1.9033781, -0.82645173, -1.15995584, 1.15286252,
                               0.1, -0.2, 0.3],
                              [-1.64202379, 0.09702638, 0.3008685, -1.9033781, -0.82645173, -1.15995584, 1.15286252,
                               -0.1, -0.2, 0.3]
                              ]).to(device)

    print(model(random_qp))

    # Ground Truth from Trimesh
    robo = DataSampler()
    sampled_q = [-1.64202379,  0.09702638,  0.3008685,  -1.9033781,  -0.82645173, -1.15995584, 1.15286252]
    robo.set_robot_joints(sampled_q)

    points = [[-0.1, 0.2, -0.3], [0.1, -0.2, 0.3], [-0.1, -0.2, 0.3]]
    print(robo.batch_calculate_signed_distance(points)*-1)