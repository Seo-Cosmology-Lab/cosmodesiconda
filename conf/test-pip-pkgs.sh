# Dependency conflicts identified and resolved with assistance from Claude
# Install pip packages.
# see https://docs.nersc.gov/development/languages/python/parallel-python/
# also https://docs.nersc.gov/development/languages/python/using-python-perlmutter/
echo Installing pip packages at $(date)
PYTHON=$(which python)

# ============================================================
# MPI / PARALLEL INFRASTRUCTURE
# ============================================================
MPICC=$MPICC $PYTHON -m pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
MPICC=$MPICCPFFT CFLAGS="-Wno-error=int-conversion" $PYTHON -m pip install --no-cache-dir git+https://github.com/MP-Gadget/pfft-python
MPICC=$MPICCPFFT $PYTHON -m pip install --no-cache-dir --no-binary=pmesh git+https://github.com/MP-Gadget/pmesh

# ============================================================
# BASE ASTRONOMY & COSMOLOGY
# ============================================================
$PYTHON -m pip install 'ipywidgets==8.0.4'
$PYTHON -m pip install --no-cache-dir getdist
# install healpy with pip, as sometimes conda yields WARNING: version mismatch between CFITSIO header (as it reinstalls cfitsio)
$PYTHON -m pip install --no-cache-dir --no-deps healpy camb isitgr emcee dynesty zeus-mcmc schwimmbad dill corner iminuit Py-BOBYQA bigfile hankl chainconsumer pydantic  # pydantic for chainconsumer
$PYTHON -m pip install --no-cache-dir pocomc
$PYTHON -m pip install --no-cache-dir --no-deps git+https://github.com/adematti/MGCAMB
$PYTHON -m pip install --no-cache-dir 'blosc>=1.9.2' # for some reason, ImportError when conda install
$PYTHON -m pip install parallel_numpy_rng

# ============================================================
# DEEP LEARNING: JAX (CUDA 12)
# jax[cuda12] pulls jaxlib, jax-cuda12-plugin, jax-cuda12-pjrt
# and all NVIDIA CUDA wheels automatically.
# Pinned to 0.9.1 for reproducibility.
# ============================================================
$PYTHON -m pip install --upgrade "jax[cuda12]==0.9.1"

# ============================================================
# DEEP LEARNING: PYTORCH (CUDA 12.9)
# ============================================================
$PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
$PYTHON -m pip install pytorch-lightning

# ============================================================
# DEEP LEARNING: TENSORFLOW
# Bare install (no version pin) pulls the latest stable TF.
# TF 2.18+ is compiled with numpy 2.0 support.
# gast is a TF graph-transformation dependency.
# Note: TF 2.18.0 pins ml-dtypes<0.5 which conflicts with
# JAX 0.9.1 (requires ml-dtypes>=0.5); TF 2.18.1+ loosened
# this to <1.0. If JAX import breaks with a ml-dtypes version
# error after this install, upgrade TF to >=2.18.1.
# ============================================================
$PYTHON -m pip install tensorflow gast
$PYTHON -m pip install pyccl

# ============================================================
# PARALLEL HDF5 STORAGE
# conda uninstall hdf5 needed because of astropy conflict
# (astropy's conda hdf5 is serial-only and blocks MPI builds).
# h5py is force-reinstalled from source with HDF5_MPI=ON so
# it links against the parallel HDF5 provided by cray-hdf5.
# --no-build-isolation uses the already-installed numpy for
# the build rather than pulling a fresh one into isolation.
# --no-deps prevents pip from reinstalling a serial h5py on
# top of the parallel build.
# NOTE: PyTables (installed later) links its own HDF5 build.
# Test `import tables; import h5py` in the same process before
# deploying — they can segfault depending on libhdf5.so load order.
# Collecting h5py<3.15.0,>=3.11.0 (from tensorflow)
# ============================================================
conda uninstall hdf5 --yes
HDF5_MPI=ON CC=cc $PYTHON -m pip install -v --force-reinstall --no-cache-dir --no-binary=h5py --no-build-isolation --no-deps h5py
$PYTHON -m pip install hdf5plugin

# ============================================================
# JAX ECOSYSTEM
# Installed after JAX itself so pip resolves all jax version
# constraints against the already-locked 0.9.1 install.
# nvidia-cublas-cu12 pin is a workaround for a JAX/XLA bug
# where certain cublas versions cause NaN results on A100s;
# see https://github.com/jax-ml/jax/issues/29042
# --no-deps used for interpax/equinox/jaxtyping to avoid
# pip re-resolving the jax version upward.
# ============================================================
$PYTHON -m pip install flax lineax
$PYTHON -m pip install blackjax
$PYTHON -m pip install --no-deps interpax equinox jaxtyping
# https://github.com/jax-ml/jax/issues/29042
$PYTHON -m pip install nvidia-cublas-cu12==12.9.0.13
$PYTHON -m pip install optax
$PYTHON -m pip install SciencePlots
$PYTHON -m pip install numpyro
$PYTHON -m pip install diffrax distrax
$PYTHON -m pip install jaxdecomp

# ============================================================
# UTILITIES & NOTEBOOKS
# tables (PyTables): HDF5 interface used by pandas HDFStore.
# sphinx + sphinx-rtd-theme: documentation only.
# ipympl: interactive matplotlib backend for Jupyter.
# ============================================================
# For h5 with pandas
$PYTHON -m pip install tables
# Just for docs
$PYTHON -m pip install sphinx sphinx-rtd-theme
$PYTHON -m pip install ipympl
$PYTHON -m pip install cupy-cuda12X  # for Roger

$PYTHON -m pip install --upgrade "astropy>=7.0" # force astropy>=7.0

if [ $? != 0 ]; then
    echo "ERROR installing pip packages; exiting"
    exit 1
fi

echo Current time $(date)
echo Done installing pip packages
