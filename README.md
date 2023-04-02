# Uncertainty SDF

## Setup

```bash
git clone --recursive git@github.com:MMintLab/usdf.git
cd usdf
pip install -e .
```

*Optional:* Build the Manifold repo:

```bash
cd 3rd/Manifold
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

## Make Mesh Watertight

We use the [Manifold](https://github.com/hjwdzh/Manifold) repo to make sure meshes are watertight. Make sure you
followed the instructions above to build the manifold repo.

1. Make single mesh watertight:

```
python scripts/make_mesh_watertight.py <path_to_input_mesh> <path_to_output_mesh>
```

2. Make a category of ShapeNet watertight:

```
python scripts/make_watertight_shapenet.py <path_to_input_category> <output_dir>
```

## TODO:

- Fix the way splits are generated/loaded to be separated from the datasets.