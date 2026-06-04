# Install pip packages.
echo Installing pip packages at $(date)

PYTHON=$(which python)
$PYTHON -m pip install "numpy<2.0" # set numpy<2.0
# see https://docs.nersc.gov/development/languages/python/parallel-python/
# also https://docs.nersc.gov/development/languages/python/using-python-perlmutter/
MPICC=$MPICC $PYTHON -m pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

MPICC=$MPICCPFFT CFLAGS="-Wno-error=int-conversion" $PYTHON -m pip install --no-cache-dir git+https://github.com/MP-Gadget/pfft-python
MPICC=$MPICCPFFT $PYTHON -m pip install --no-cache-dir --no-binary=pmesh git+https://github.com/MP-Gadget/pmesh
$PYTHON -m pip install 'ipywidgets==8.0.4'
$PYTHON -m pip install --no-cache-dir getdist
# $PYTHON -m pip install --no-cache-dir --no-deps healpy camb schwimmbad dill bigfile hankl
$PYTHON -m pip install --no-cache-dir --no-deps healpy camb isitgr emcee dynesty zeus-mcmc schwimmbad dill corner iminuit Py-BOBYQA bigfile hankl chainconsumer pydantic  # pydantic for chainconsumer
$PYTHON -m pip install --no-cache-dir pocomc
$PYTHON -m pip install --no-cache-dir --no-deps git+https://github.com/adematti/MGCAMB
# for abacusutils
$PYTHON -m pip install --no-cache-dir 'blosc>=1.9.2' # for some reason, ImportError when conda install
$PYTHON -m pip install --no-cache-dir --no-deps parallel_numpy_rng # avoid numpy>2.0 requirement 

# ML
# $PYTHON -m pip install pytorch-lightning
# $PYTHON -m pip install tensorflow gast
$PYTHON -m pip install pyccl

$PYTHON -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
$PYTHON -m pip install "tensorflow==2.11.0"
$PYTHON -m pip install "flax==0.8.2"
$PYTHON -m pip install "chex==0.1.86"
$PYTHON -m pip install "optax==0.2.2"
$PYTHON -m pip install "equinox==0.11.4"
$PYTHON -m pip install "jaxopt<=0.8.3" #
$PYTHON -m pip install "lineax<=0.1.0" # 
$PYTHON -m pip install --no-deps interpax jaxtyping blackjax fastprogress typeguard
$PYTHON -m pip freeze | grep "nvidia*cu11" | xargs pip uninstall -y
$PYTHON -m pip freeze | grep "jaxlib" | xargs pip uninstall -y
$PYTHON -m pip install -U "jax[cuda11_pip]==0.4.25" --find-links=https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
$PYTHON -m pip install "nvidia-cudnn-cu11==8.9.4.25"

$PYTHON -m pip install SciencePlots
$PYTHON -m pip install numpyro
$PYTHON -m pip install diffrax distrax
$PYTHON -m pip install jaxdecomp

$PYTHON -m pip install pytorch-lightning
$PYTHON -m pip install tensorflow gast

# Collecting h5py<3.15.0,>=3.11.0 (from tensorflow)
# conda uninstall hdf5 needed because of astropy, I believe
conda uninstall hdf5 --yes
HDF5_MPI=ON CC=cc $PYTHON -m pip install -v --force-reinstall --no-cache-dir --no-binary=h5py --no-build-isolation --no-deps h5py
$PYTHON -m pip install hdf5plugin

# For h5 with pandas
$PYTHON -m pip install tables

# Just for docs
$PYTHON -m pip install sphinx sphinx-rtd-theme
$PYTHON -m pip install ipympl

if [ $? != 0 ]; then
    echo "ERROR installing pip packages; exiting"
    exit 1
fi

echo Current time $(date) Done installing pip packages
