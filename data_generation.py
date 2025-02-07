from data import DataSampler


if __name__ == "__main__":
    robo = DataSampler(dataset_path="dataset/")

    # batch sample outside
    robo.batch_sample_outside_mesh(batch_size=4, base_num=20, offset_range=[0., 0.1])

    # batch sample inside
    robo.batch_sample_inside_mesh(batch_size=4, base_num=20, link_weights=[1, 1, 1, 1, 1, 2, 1, 1])