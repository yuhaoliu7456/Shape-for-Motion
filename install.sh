

# Install nvdiffrast
pip install git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
pip install git+https://github.com/NVlabs/nvdiffrast/

# Install pytorch3d
export FORCE_CUDA=1
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git"


# Install submodules
pip install src/s1/submodules/diff-gaussian-rasterization
pip install src/s1/submodules/simple-knn

# Install other dependencies
pip install -r requirements.txt
