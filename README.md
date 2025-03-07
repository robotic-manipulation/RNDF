# Robot Neural Distance Function

Implementation for paper "Implicit Articulated Robot Morphology Modeling with Configuration Space Neural Signed Distance Functions" [**ICRA 2025**] 

![Screenshot from 2025-02-05 17-33-40](./media/examples.png)



### Dependencies

- Python 3.8 (tested)
- PyTorch 2.4 (tested)
- numpy 1.24.4 (tested)
- urdfpy 0.0.22 (tested, for data generation and visualization)
- Trimesh 4.4.7 (tested, for data generation and visualization)
- Open3D 0.16.0 (tested, for visualization) 



### Neural SDF in the Robot Joint Space

**Lightweight** (size of 64 ~ 421.9 KB; size of 128 ~1 MB), **Differentiable**, **Parallelable**, and **Accurate** 

```python
# see inference.py, we use KUKA iiwa 7 as an example
import torch
from models import RobotNDF

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = RobotNDF(N=128).to(device)     # feature size of 128
    model.load_model("models/weight/128_params.pth")
	
    # feature size of 64
    # model = RobotNDF(N=64).to(device)
    # model.load_model("models/weight/64_params.pth")
    
    ####################
    # Model Prediction #
    ####################
    random_qp = torch.tensor([[-1.64202379,  0.09702638,  0.3008685,  -1.9033781,  -0.82645173, -1.15995584, 1.15286252,
                              -0.1, 0.2, -0.3],
                              [-1.64202379, 0.09702638, 0.3008685, -1.9033781, -0.82645173, -1.15995584, 1.15286252,
                               0.1, -0.2, 0.3],
                              [-1.64202379, 0.09702638, 0.3008685, -1.9033781, -0.82645173, -1.15995584, 1.15286252,
                               -0.1, -0.2, 0.3]
                              ]).to(device)     # input: (3, 10), 3 X (joint state,7 + query point,3)
	
    # print output: signed distance value conditioned on the robot configuraiton for each query point
    print(model(random_qp))
    """
	Prediction: 
	tensor(
	[[0.3167, 0.4775, 0.6122, 0.8805, 1.0217, 1.0173, 1.0143, 1.1295],
     [0.1992, 0.1643, 0.1357, 0.2923, 0.3463, 0.3233, 0.2841, 0.3846],
     [0.1997, 0.1383, 0.1554, 0.2913, 0.3602, 0.3454, 0.3196, 0.4654]],
       device='cuda:0', grad_fn=<MulBackward0>)
    """
    
    #############################
    # Ground Truth from Trimesh #
    #############################
    robo = DataSampler()
    sampled_q = [-1.64202379,  0.09702638,  0.3008685,  -1.9033781,  -0.82645173, -1.15995584, 1.15286252] # same joint state
    robo.set_robot_joints(sampled_q)

    points = [[-0.1, 0.2, -0.3], [0.1, -0.2, 0.3], [-0.1, -0.2, 0.3]] # same query points
    print(robo.batch_calculate_signed_distance(points)*-1) # positive as outside
    """
    Ground Truth: 
    [[0.3183 0.4778 0.6124 0.8822 1.0209 1.0205 1.0150 1.1316]
     [0.1985 0.1626 0.1362 0.2930 0.3467 0.3257 0.2840 0.3839]
     [0.1984 0.1382 0.1558 0.2921 0.3606 0.3467 0.3197 0.4665]]
    """

    
```



### Examples

```python
# see example.py
import numpy as np
from models import iiwa_RNDF

if __name__ == '__main__':
    np.random.seed(16)

    # initialize RNDF for KUKA iiwa 7
    iiwa_model = iiwa_RNDF()

    # sample random robot configuration
    random_q = iiwa_model.sample_random_robot_config()
    iiwa_model.set_robot_joint_positions(random_q) # set robot configuration
    iiwa_model.show_robot()
    # print(random_q)

    # sample random point in the robot frame
    point = np.array([0.1, 0.2, 0.3])
    iiwa_model.show_robot_with_points(point)

    # Signed Distance Computation, +1: Outside -1: Inside
    link_wise_dist = iiwa_model.calculate_signed_distance(point, min_dist=False)
    print("Signed Distance to Each Link: {}".format(link_wise_dist[0]))

    minimum_dist = iiwa_model.calculate_signed_distance(point, min_dist=True)
    print("Signed Distance to the Robot: {}".format(minimum_dist[0]))
```



### Dataset Generation

**For more details please refer to data/data_sampler.py**, we use KUKA iiwa 7 as an example.

```python
# see data_generation.py
from data import DataSampler


if __name__ == "__main__":
    # path to save the generated data, (n, 18), [:10] as input and [10:] as sdf
    robo = DataSampler(dataset_path='../dataset/')
    
    # batch sample outside; offset range is the sampled distance range for each link
    robo.batch_sample_outside_mesh(batch_size=1024, base_num=20, offset_range=[0., 0.1])

    # batch sample inside; a rejection-based sampling method is used
    robo.batch_sample_inside_mesh(batch_size=1024, base_num=20, link_weights=[1, 1, 1, 1, 1, 2, 1, 1])
```



### Build RNDF for Your Own Robot

TBD





### Roadmap

- [x] IIWA Model Release

- [x] Dataset Generation

- [ ] Model Training 

- [ ] Visualization

- [ ] More Robots (Franka Emika Panda, Allegro Hand, ...)

  

