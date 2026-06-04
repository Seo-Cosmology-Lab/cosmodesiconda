# Install pip packages.
echo Installing pip packages at $(date)

PYTHON=$(which python)
# see https://docs.nersc.gov/development/languages/python/parallel-python/
# also https://docs.nersc.gov/development/languages/python/using-python-perlmutter/
MPICC=$MPICC $PYTHON -m pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

MPICC=$MPICCPFFT CFLAGS="-Wno-error=int-conversion" $PYTHON -m pip install --no-cache-dir git+https://github.com/MP-Gadget/pfft-python
MPICC=$MPICCPFFT $PYTHON -m pip install --no-cache-dir --no-binary=pmesh git+https://github.com/MP-Gadget/pmesh
$PYTHON -m pip install 'ipywidgets==8.0.4'
$PYTHON -m pip install --no-cache-dir --no-deps healpy camb schwimmbad dill bigfile hankl

#$PYTHON -m pip install --no-cache-dir --no-deps git+https://github.com/minaskar/pocomc.git
# for abacusutils
$PYTHON -m pip install --no-cache-dir 'blosc>=1.9.2' # for some reason, ImportError when conda install
$PYTHON -m pip install parallel_numpy_rng
# ML
$PYTHON -m pip install --upgrade "jax[cuda12]==0.9.1"
$PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
$PYTHON -m pip install pytorch-lightning
$PYTHON -m pip install tensorflow gast
$PYTHON -m pip install pyccl
# Collecting h5py<3.15.0,>=3.11.0 (from tensorflow)
# conda uninstall hdf5 needed because of astropy, I believe
conda uninstall hdf5 --yes
HDF5_MPI=ON CC=cc $PYTHON -m pip install -v --force-reinstall --no-cache-dir --no-binary=h5py --no-build-isolation --no-deps h5py
$PYTHON -m pip install hdf5plugin

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
# For h5 with pandas
$PYTHON -m pip install tables
# Just for docs
$PYTHON -m pip install sphinx sphinx-rtd-theme
$PYTHON -m pip install ipympl
$PYTHON -m pip install cupy-cuda12X  # for Roger


if [ $? != 0 ]; then
    echo "ERROR installing pip packages; exiting"
    exit 1
fi

echo Current time $(date) Done installing pip packages
