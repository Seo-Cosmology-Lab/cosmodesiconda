# Install pip packages.
echo Installing pip packages at $(date)
PYTHON=$(which python)

$PYTHON -m pip install "numpy<2.0"

MPICC=$MPICC $PYTHON -m pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
MPICC=$MPICCPFFT CFLAGS="-Wno-error=int-conversion" $PYTHON -m pip install --no-cache-dir git+https://github.com/MP-Gadget/pfft-python
MPICC=$MPICCPFFT $PYTHON -m pip install --no-cache-dir --no-binary=pmesh git+https://github.com/MP-Gadget/pmesh

# BASE ASTRONOMY & COSMOLOGY
$PYTHON -m pip install 'ipywidgets==8.0.4'
$PYTHON -m pip install --no-cache-dir getdist
$PYTHON -m pip install --no-cache-dir --no-deps healpy camb isitgr emcee dynesty zeus-mcmc schwimmbad dill corner iminuit Py-BOBYQA bigfile hankl chainconsumer pydantic 
$PYTHON -m pip install --no-cache-dir pocomc
$PYTHON -m pip install --no-cache-dir --no-deps git+https://github.com/adematti/MGCAMB
$PYTHON -m pip install --no-cache-dir 'blosc>=1.9.2'
$PYTHON -m pip install --no-cache-dir --no-deps parallel_numpy_rng 
$PYTHON -m pip install pyccl
$PYTHON -m pip install SciencePlots

# DEEP LEARNING (TORCH / TENSORFLOW)
$PYTHON -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
$PYTHON -m pip install "tensorflow==2.11.0"
$PYTHON -m pip install pytorch-lightning
$PYTHON -m pip install tensorflow gast

# JAX
$PYTHON -m pip install "flax==0.8.2"
$PYTHON -m pip install "chex==0.1.85"
$PYTHON -m pip install "optax==0.1.9"
$PYTHON -m pip install "equinox==0.11.3"
$PYTHON -m pip install --no-deps interpax jaxtyping blackjax fastprogress typeguard

# Force legacy JAX to match CUDA 11 runtime
$PYTHON -m pip install "jax==0.4.23" "jaxlib==0.4.23+cuda11.cudnn86" --find-links=https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# PINNED ADAPTIVE JAX PACKAGES
$PYTHON -m pip install "numpyro==0.13.2"
$PYTHON -m pip install "diffrax==0.5.0"
$PYTHON -m pip install "distrax==0.1.5"
$PYTHON -m pip install "jaxdecomp==0.2.0"

# PARALLEL HDF5 STORAGE 
conda uninstall hdf5 --yes
HDF5_MPI=ON CC=cc $PYTHON -m pip install -v --force-reinstall --no-cache-dir --no-binary=h5py --no-build-isolation --no-deps h5py
$PYTHON -m pip install hdf5plugin
$PYTHON -m pip install tables

#UTILITIES & NOTEBOOKS
$PYTHON -m pip install sphinx sphinx-rtd-theme
$PYTHON -m pip install ipympl

if [ $? != 0 ]; then
    echo "ERROR installing pip packages; exiting"
    exit 1
fi

echo Current time $(date)
Done installing pip packages
