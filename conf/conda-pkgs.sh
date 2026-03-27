# conda packages
echo Current time $(date) Installing conda packages
echo condadir is $CONDADIR
# matplotlib: apparently, bug in inset_axis with 3.7.2
# cython: https://github.com/gevent/gevent/issues/1899
# libblas with openblas: yields warnings with NUM_THREADS
conda install --copy --yes -c conda-forge \
    future \
    'libblas=*=*mkl' \
    'numpy>=2.0.0' \
    'scipy>=1.5.0' \
    matplotlib \
    cfitsio \
    fitsio \
    cython \
    pyfftw \
    sympy \
    numexpr \
    'astropy>=7.0.0' \
    ipython \
    jupyterlab \
    wurlitzer \
    'numba>=0.56' \
    'asdf>=2.13' \
    pandas \
    dask \
    pytest \
    scikit-learn \
    tabulate \
    pyccl \
    namaster \
&& mplrc="$CONDADIR/lib/python$PYVERSION/site-packages/matplotlib/mpl-data/matplotlibrc"; \
    cat ${mplrc} | sed -e "s#^backend.*#backend : TkAgg#" > ${mplrc}.tmp; \
    mv ${mplrc}.tmp ${mplrc} \
&& conda clean --yes --all #&& conda uninstall hdf5 --yes
if [ $? != 0 ]; then
    echo "ERROR installing conda packages; exiting"
    exit 1
fi

conda list --export | grep -v conda > "$CONDADIR/pkg_list.txt"
echo Current time $(date) Done installing conda packages
