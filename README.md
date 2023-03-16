# Uncertainty SDF

## Setup

```bash
git clone --recursive git@github.com:MMintLab/usdf.git
```

## Make Mesh Watertight

We use the [Manifold](https://github.com/hjwdzh/Manifold) repo to make sure meshes are watertight.

Build the Manifold repo:

```bash
cd 3rd/Manifold
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```