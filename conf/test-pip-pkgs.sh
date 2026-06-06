# Dependency conflicts identified and resolved with assistance from Claude
# Install pip packages.
echo Installing pip packages at $(date)
PYTHON=$(which python)

# ============================================================
# STEP 1: NUMPY — pin before anything else.
# numpy>=1.26 satisfies JAX 0.9.x, PyTorch cu129, and TF 2.17.
# Cap at <2.0: healpy, camb, and astronomy C-extensions have
# runtime ABI issues with numpy 2.x despite claiming support.
# ============================================================
$PYTHON -m pip install "numpy>=1.26,<2.0"

# ============================================================
# STEP 2: MPI / PARALLEL INFRASTRUCTURE
# Must be compiled against system MPI (MPICC) before any
# scientific stack is installed.
# ============================================================
MPICC=$MPICC $PYTHON -m pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
MPICC=$MPICCPFFT CFLAGS="-Wno-error=int-conversion" $PYTHON -m pip install --no-cache-dir git+https://github.com/MP-Gadget/pfft-python
MPICC=$MPICCPFFT $PYTHON -m pip install --no-cache-dir --no-binary=pmesh git+https://github.com/MP-Gadget/pmesh

# ============================================================
# STEP 3: JAX RUNTIME + ENTIRE JAX ECOSYSTEM
# Pin jax first, then install every jax-dependent package
# immediately. Installing PyTorch or TF in between risks pip
# re-resolving jax upward/downward to satisfy their deps.
#
# nvidia-cublas-cu12 pin: workaround for jax/issues/29042.
# Placed here (after jax, before PyTorch) so it satisfies JAX
# without downgrading PyTorch's bundled cublas post-install.
# NOTE: test `torch.cuda.is_available()` and `jnp.ones(1)` in
# the same process — the two frameworks share CUDA libraries
# and the cublas pin may need adjustment per driver version.
# ============================================================
$PYTHON -m pip install --upgrade "jax[cuda12]==0.9.1"
$PYTHON -m pip install nvidia-cublas-cu12==12.9.0.13  # JAX cublas workaround; must precede PyTorch

# JAX ecosystem — install all at once while jax version is locked
$PYTHON -m pip install flax lineax
$PYTHON -m pip install optax
$PYTHON -m pip install blackjax
$PYTHON -m pip install numpyro
$PYTHON -m pip install diffrax distrax
$PYTHON -m pip install jaxdecomp
$PYTHON -m pip install --no-deps interpax equinox jaxtyping

# ============================================================
# STEP 4: PYTORCH (CUDA 12.9)
# Installed after JAX so the cublas pin above is already in
# place. PyTorch cu129 bundles its own nvidia-* libs; pip will
# not re-download cublas since the pinned version is already
# present.
# ============================================================
$PYTHON -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

$PYTHON -m pip install pytorch-lightning

# ============================================================
# STEP 5: TENSORFLOW
# Pin explicitly so pip cannot resolve numpy>=2.0 via TF's
# lower bound. TF 2.16+ dropped gast; do not install separately.
# Requires h5py>=3.11,<3.15 — installed later with MPI support.
# ============================================================
$PYTHON -m pip install "tensorflow==2.17.0"
# NOTE: do NOT add a bare `pip install tensorflow` or
# `pip install tensorflow gast` anywhere after this line —
# it would overwrite the pin and re-resolve numpy freely.

# ============================================================
# STEP 6: BASE ASTRONOMY & COSMOLOGY
# install healpy with pip — conda can cause CFITSIO header
# version mismatch warnings on reinstall.
# pydantic>=2.0 installed explicitly first so chainconsumer
# (which requires pydantic v2) can validate its own deps.
# ============================================================
$PYTHON -m pip install 'ipywidgets>=8.1'   # 8.0.4 causes silent degradation with ipympl 0.9.4+
$PYTHON -m pip install --no-cache-dir getdist
$PYTHON -m pip install "pydantic>=2.0"     # chainconsumer 1.x requires pydantic v2
$PYTHON -m pip install --no-cache-dir --no-deps \
    healpy camb isitgr emcee dynesty zeus-mcmc schwimmbad dill corner iminuit \
    Py-BOBYQA bigfile hankl
$PYTHON -m pip install --no-cache-dir chainconsumer   # now has pydantic v2 available
$PYTHON -m pip install --no-cache-dir pocomc
$PYTHON -m pip install --no-cache-dir --no-deps git+https://github.com/adematti/MGCAMB
$PYTHON -m pip install --no-cache-dir 'blosc>=1.9.2'  # ImportError when conda-installed
$PYTHON -m pip install --no-cache-dir --no-deps parallel_numpy_rng  # --no-deps prevents mpi4py re-resolve
$PYTHON -m pip install pyccl
$PYTHON -m pip install SciencePlots

# ============================================================
# STEP 7: PARALLEL HDF5 STORAGE
# conda uninstall hdf5 needed because of astropy conflict.
# NOTE: PyTables links its own HDF5 build. Test
# `import tables; import h5py` in the same process before
# deploying — parallel-HDF5 h5py and PyTables can segfault
# depending on which libhdf5.so is loaded first.
# ============================================================
conda uninstall hdf5 --yes
HDF5_MPI=ON CC=cc $PYTHON -m pip install \
    -v --force-reinstall --no-cache-dir --no-binary=h5py \
    --no-build-isolation --no-deps h5py
$PYTHON -m pip install hdf5plugin
$PYTHON -m pip install tables  # PyTables — see runtime note above

# ============================================================
# STEP 8: UTILITIES, DOCS & NOTEBOOKS
# ============================================================
$PYTHON -m pip install sphinx sphinx-rtd-theme
$PYTHON -m pip install ipympl
$PYTHON -m pip install cupy-cuda12X  # for Roger

if [ $? != 0 ]; then
    echo "ERROR installing pip packages; exiting"
    exit 1
fi

echo Current time $(date)
echo Done installing pip packages