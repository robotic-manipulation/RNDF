import torch
from models import RobotNDF

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = RobotNDF(N=128).to(device)
    model.load_model("models/weight/128_params.pth")

    # (3, 10), 3 (joint state + position) pairs
    random_qp = torch.tensor([[-1.64202379,  0.09702638,  0.3008685,  -1.9033781,  -0.82645173, -1.15995584, 1.15286252,
                              0.1, 0.2, 0.3],
                              [-1.64202379, 0.09702638, 0.3008685, -1.9033781, -0.82645173, -1.15995584, 1.15286252,
                               0.1, 0.2, 0.3],
                              [-1.64202379, 0.09702638, 0.3008685, -1.9033781, -0.82645173, -1.15995584, 1.15286252,
                               0.1, 0.2, 0.3]
                              ]).to(device)

    print(model(random_qp))
