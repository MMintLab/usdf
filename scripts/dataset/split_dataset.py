import argparse
import os.path

import mmint_utils
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into train/test.")
    parser.add_argument("dataset_dir", type=str, help="Dataset directory.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    splits_dir = os.path.join(dataset_dir, "splits")
    mmint_utils.make_dir(splits_dir)

    mesh_fns = [f for f in os.listdir(dataset_dir) if ".obj" in f]
    np.random.shuffle(mesh_fns)

    num_train = int(0.8 * len(mesh_fns))
    train_fns = mesh_fns[:num_train]
    test_fns = mesh_fns[num_train:]

    with open(os.path.join(splits_dir, "train.txt"), "w") as f:
        for fn in train_fns:
            f.write(fn + "\n")

    with open(os.path.join(splits_dir, "test.txt"), "w") as f:
        for fn in test_fns:
            f.write(fn + "\n")
