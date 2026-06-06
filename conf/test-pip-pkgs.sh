# Dependency conflicts identified and resolved with assistance from Claude
# Install pip packages.
# see https://docs.nersc.gov/development/languages/python/parallel-python/
# also https://docs.nersc.gov/development/languages/python/using-python-perlmutter/
echo Installing pip packages at $(date)
PYTHON=$(which python)

# ============================================================
# STEP 1: CORE BUILD TOOLS
# setuptools must come first on Python 3.12 — distutils was
# removed from the stdlib and any source build that still
# references it (pfft-python, pmesh) will fail without this.
# ============================================================
$PYTHON -m pip install --upgrade setuptools

# ============================================================
# STEP 2: NUMPY
# numpy>=2.0 is required by jax==0.9.1 (hard lower bound).
# Cap at <2.3: jax 0.9.1 was released before numpy 2.3 and
# the combination is untested; raise the cap once jax 0.9.2+
# is deployed.
# NOTE: PolyBin3D must be installed from a patched local clone
# (pyproject.toml numpy pin relaxed) — see separate script.
# ============================================================
$PYTHON -m pip install "numpy>=2.0,<2.3"

# ============================================================
# STEP 3: MPI / PARALLEL INFRASTRUCTURE
# Built against system MPI (MPICC) before scientific stack.
# CFLAGS appends -Wno-error=int-conversion to the global
# CFLAGS rather than replacing it (uses ${CFLAGS} expansion).
# ============================================================
MPICC=$MPICC $PYTHON -m pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
MPICC=$MPICCPFFT CFLAGS="${CFLAGS} -Wno-error=int-conversion" $PYTHON -m pip install --no-cache-dir git+https://github.com/MP-Gadget/pfft-python
MPICC=$MPICCPFFT $PYTHON -m pip install --no-cache-dir --no-binary=pmesh git+https://github.com/MP-Gadget/pmesh

# ============================================================
# STEP 4: JAX RUNTIME + FULL JAX ECOSYSTEM
# JAX 0.9.1 requires numpy>=2.0 (installed above).
# All JAX-dependent packages are installed immediately after
# locking the JAX version so pip cannot re-resolve it later.
#
# nvidia-cublas-cu12 pin: workaround for jax/issues/29042.
# Placed here (after jax, before PyTorch) so it satisfies JAX
# without downgrading PyTorch's bundled cublas post-install.
# NOTE: test `torch.cuda.is_available()` and `jnp.ones(1)` in
# the same process — both frameworks share CUDA libs and the
# cublas pin may need adjustment per driver version.
# ============================================================
$PYTHON -m pip install --upgrade "jax[cuda12]==0.9.1"
$PYTHON -m pip install nvidia-cublas-cu12==12.9.0.13  # JAX cublas workaround; before PyTorch

# JAX ecosystem — install all while jax version is locked
$PYTHON -m pip install flax lineax
$PYTHON -m pip install optax
$PYTHON -m pip install blackjax
$PYTHON -m pip install numpyro
$PYTHON -m pip install diffrax distrax
$PYTHON -m pip install jaxdecomp
$PYTHON -m pip install --no-deps interpax equinox jaxtyping

# ============================================================
# STEP 5: PYTORCH (CUDA 12.9)
# Installed after JAX so the cublas pin above is already set.
# ============================================================
$PYTHON -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

$PYTHON -m pip install pytorch-lightning

# ============================================================
# STEP 6: TENSORFLOW
# TF 2.17 does NOT natively support numpy 2.x (wheels were
# compiled against numpy 1.x). TF 2.18 is the first release
# compiled with numpy 2.0. Pin to 2.18 to avoid the ABI
# mismatch that causes "compiled using NumPy 1.x" ImportError
# at runtime with numpy 2.x installed.
# gast is not used by TF 2.16+; omit it.
# ============================================================
$PYTHON -m pip install "tensorflow==2.18.0"
# NOTE: never run a bare `pip install tensorflow` after this —
# it would overwrite the pin and re-resolve numpy freely.

# ============================================================
# STEP 7: BASE ASTRONOMY & COSMOLOGY
# pydantic>=2.0 installed explicitly before chainconsumer
# (chainconsumer 1.x requires pydantic v2).
# parallel_numpy_rng gets --no-deps to prevent mpi4py
# re-resolve outside the MPICC-compiled build from step 3.
# ============================================================
$PYTHON -m pip install 'ipywidgets>=8.1'  # 8.0.4 degrades with ipympl 0.9.4+
$PYTHON -m pip install --no-cache-dir getdist
$PYTHON -m pip install "pydantic>=2.0"    # required by chainconsumer 1.x
# install healpy with pip: conda can cause CFITSIO header version mismatch
$PYTHON -m pip install --no-cache-dir --no-deps \
    healpy camb isitgr emcee dynesty zeus-mcmc schwimmbad dill corner iminuit \
    Py-BOBYQA bigfile hankl
$PYTHON -m pip install --no-cache-dir chainconsumer  # pydantic v2 now available
$PYTHON -m pip install --no-cache-dir pocomc
$PYTHON -m pip install --no-cache-dir --no-deps git+https://github.com/adematti/MGCAMB
$PYTHON -m pip install --no-cache-dir 'blosc>=1.9.2'  # ImportError when conda-installed
$PYTHON -m pip install --no-cache-dir --no-deps parallel_numpy_rng
$PYTHON -m pip install pyccl
$PYTHON -m pip install SciencePlots

# ============================================================
# STEP 9: PARALLEL HDF5 STORAGE
# conda uninstall hdf5 needed because of astropy conflict.
# NOTE: PyTables links its own HDF5; test `import tables;
# import h5py` in the same process before deploying —
# parallel-HDF5 h5py and PyTables can segfault depending on
# which libhdf5.so is loaded first.
# ============================================================
conda uninstall hdf5 --yes
HDF5_MPI=ON CC=cc $PYTHON -m pip install \
    -v --force-reinstall --no-cache-dir --no-binary=h5py \
    --no-build-isolation --no-deps h5py
$PYTHON -m pip install hdf5plugin
$PYTHON -m pip install tables  # PyTables — see runtime note above

# ============================================================
# STEP 10: UTILITIES, DOCS & NOTEBOOKS
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
