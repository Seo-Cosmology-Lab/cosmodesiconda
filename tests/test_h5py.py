"""To run with multiple processes on a CPU node."""

import os
from pathlib import Path

import numpy as np


def test_h5py():
    # From https://docs.nersc.gov/development/languages/python/parallel-python/
    from mpi4py import MPI
    import h5py
    print(h5py.__file__)

    mpicomm = MPI.COMM_WORLD

    fn = dirname / 'test.h5'
    with h5py.File(fn, 'w', driver='mpio', comm=mpicomm) as f:
      dset = f.create_dataset('test', (4,), dtype='i')
      dset[mpicomm.rank] = mpicomm.rank

    mpicomm.Barrier()
    if mpicomm.rank == 0:
        with h5py.File(fn, 'r') as f:
            print(f['test'][...])
        os.remove(fn)


def test_catalog():
    from mpytools import Catalog
    from mpytools.random import MPIRandomState

    rng = MPIRandomState(size=1000, seed=42)
    catalog = Catalog(data={'RA': rng.uniform(0., 1.), 'DEC': rng.uniform(0., 1.), 'Z': rng.uniform(0., 1.), 'Position': rng.uniform(0., 1., itemshape=3)})
    fn = dirname / 'test.h5'
    catalog.write(fn)
    mpicomm = catalog.mpicomm
    catalog2 = Catalog.read(fn)
    assert np.allclose(catalog2['Position'], catalog['Position'])
    mpicomm.Barrier()
    if mpicomm.rank == 0:
        os.remove(fn)


if __name__ == '__main__':

    dirname = Path('_tests')
    dirname.mkdir(exist_ok=True)
    test_h5py()
    test_catalog()
