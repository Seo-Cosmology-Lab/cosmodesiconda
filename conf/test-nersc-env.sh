#export MINICONDA=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# miniforge solves fast and works well with conda-forge
export MINICONDA=https://github.com/conda-forge/miniforge/releases/download/25.3.0-1/Miniforge3-$(uname)-$(uname -m).sh # for python 3.12
# export MINICONDA=https://github.com/conda-forge/miniforge/releases/download/24.1.2-0/Miniforge3-$(uname)-$(uname -m).sh  # for python 3.10
# export MINICONDA=https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Miniforge3-$(uname)-$(uname -m).sh # for python 3.8
export GRP=desi
export PRGENVS="PrgEnv-gnu PrgEnv-intel PrgEnv-cray PrgEnv-nvidia"
export CONDAPRGENV=PrgEnv-gnu
export COSMOINSTALL="test-pip-pkgs.sh camb-pkgs.sh classy-pkgs.sh planck-pkgs.sh nersc-cosmosis-pkgs.sh cobaya-pkgs.sh desilike-pkgs.sh desipipe-pkgs.sh nersc-scripts.sh"
#export COSMOINSTALL="planck-pkgs.sh"
# darshan not necessary and suspected to generate overhead
# altd not necessary and suspected to cause random job hangs
# craype-hugepages2M https://docs.nersc.gov/development/languages/python/faq-troubleshooting
export UNLOADMODULES="cray-hdf5"
#export LOADMODULES="cudatoolkit/12.2 cudnn/8.9.3_cuda12 nccl/2.17.1-ofi cray-hdf5"
export LOADMODULES="gcc-native/14 cray-hdf5-parallel cudatoolkit/12.9 craype-accel-nvidia80 cray-mpich"
export HOSTVARIABLE=NERSC_HOST

export CC="gcc"
export FC="gfortran"
export CFLAGS="-O3 -fPIC -pthread"
export FCFLAGS="-O3 -fPIC -pthread -fexceptions"
export NTMAKE=8

if [ "${NERSC_HOST}" == "cori" ] ; then
  export UNLOADMODULES="darshan altd craype-hugepages2M"
  # See https://docs.nersc.gov/development/languages/python/parallel-python/
  export LOADMODULES="${LOADMODULES} texlive/2019"
  export ENVVARIABLES="HDF5_USE_FILE_LOCKING FALSE"
  export MPICC="cc -shared"
elif [ "${NERSC_HOST}" == "perlmutter" ] ; then
  # See https://docs.nersc.gov/development/languages/python/using-python-perlmutter
  export LOADMODULES="${LOADMODULES} texlive"
  # See https://github.com/jax-ml/jax/issues/29031
  export ENVVARIABLES='MPI4PY_RC_RECV_MPROBE FALSE MPICH_GPU_SUPPORT_ENABLED FALSE CXI_FORK_SAFE 1 CXI_FORK_SAFE_HP 1 HDF5_USE_FILE_LOCKING FALSE XLA_FLAGS "--xla_gpu_enable_cublaslt=true --xla_gpu_cublas_fallback=true --xla_gpu_enable_command_buffer="'
  export MPICC="cc -shared"
fi
export MPICCPFFT="mpicc"
export ENVVARIABLES="${ENVVARIABLES} TF_CPP_MIN_LOG_LEVEL 2"
export ENVVARIABLES_LOGIN="JAX_PLATFORMS cpu"
export ENVVARIABLES_NODES="JAX_PLATFORMS """
