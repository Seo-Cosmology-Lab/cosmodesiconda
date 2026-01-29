import os


def test_h5py(fn='test.h5'):
    # From https://docs.nersc.gov/development/languages/python/parallel-python/
    from mpi4py import MPI
    import h5py
    print(h5py.__file__)

    mpicomm = MPI.COMM_WORLD

    with h5py.File(fn, 'w', driver='mpio', comm=mpicomm) as f:
      dset = f.create_dataset('test', (4,), dtype='i')
      dset[mpicomm.rank] = mpicomm.rank

    mpicomm.Barrier()
    if mpicomm.rank == 0:
        with h5py.File(fn, 'r') as f:
            print(f['test'][...])
        os.remove(fn)


if __name__ == '__main__':

    test_h5py()
