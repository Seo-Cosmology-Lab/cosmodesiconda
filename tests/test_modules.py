import os
import copy
import numpy as np


def generate_catalogs(size=100, boxsize=(1000.,) * 3, boxcenter=(1000.,) * 3, n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    positions = np.column_stack([c + rng.uniform(-0.5, 0.5, size) * s for c, s in zip(boxcenter, boxsize)])
    weights = [rng.randint(0, 0xffffffff, size, dtype='i8') for i in range(n_bitwise_weights)]
    weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
    if len(weights) == 1: weights = weights[0]
    return positions, weights


def test_cosmoprimo():

    # cosmoprimo, pyclass, camb
    from cosmoprimo.fiducial import DESI

    DESI(engine='class').get_fourier().pk_interpolator()
    DESI(engine='camb').get_fourier().pk_interpolator()


def test_pycorr():

    data_positions, data_weights = generate_catalogs()
    randoms_positions, randoms_weights = generate_catalogs()

    from pycorr import TwoPointCorrelationFunction
    TwoPointCorrelationFunction(mode='smu', edges=(np.linspace(0., 50., 51), np.linspace(-1., 1., 51)), data_positions1=data_positions, data_weights1=data_weights, randoms_positions1=randoms_positions, randoms_weights1=randoms_weights, position_type='pos', nthreads=1)
    TwoPointCorrelationFunction(mode='smu', edges=(np.linspace(0., 50., 51), np.linspace(-1., 1., 51)), data_positions1=data_positions, data_weights1=data_weights, randoms_positions1=randoms_positions, randoms_weights1=randoms_weights, position_type='pos', nthreads=1, gpu=True)
    TwoPointCorrelationFunction(mode='smu', edges=(np.linspace(0., 50., 51), np.linspace(-1., 1., 51)), data_positions1=data_positions, data_weights1=data_weights, randoms_positions1=randoms_positions, randoms_weights1=randoms_weights, position_type='pos', nthreads=1, engine='cucount')


def test_pypower():
    from pypower import CatalogFFTPower
    size = 10000
    data_positions, data_weights = generate_catalogs(size=size)
    randoms_positions, randoms_weights = generate_catalogs(size=size)
    CatalogFFTPower(edges={'step': 0.01}, data_positions1=data_positions, data_weights1=data_weights, randoms_positions1=randoms_positions, randoms_weights1=randoms_weights, position_type='pos', nmesh=64)


def test_pyrecon():
    size = 10000
    data_positions, data_weights = generate_catalogs(size=size)
    randoms_positions, randoms_weights = generate_catalogs(size=size)
    from pyrecon import MultiGridReconstruction
    recon = MultiGridReconstruction(f=0.8, bias=2.0, los=None, nmesh=32, boxsize=2000., boxcenter=1000.)
    recon.assign_data(data_positions, data_weights)
    recon.assign_randoms(randoms_positions, randoms_weights)
    recon.set_density_contrast()
    recon.run()
    data_positions_rec = recon.read_shifted_positions(data_positions)


def test_cucount():
    size = 10000
    positions1, weights1 = generate_catalogs(size=size)
    positions2, weights2 = generate_catalogs(size=size)

    def run(**kwargs):
        particles1 = Particles(positions1, weights1)
        particles2 = Particles(positions2, weights2)
        battrs = BinAttrs(s=np.linspace(1., 201, 201), mu=(np.linspace(-1., 1., 201), 'midpoint'))
        return count2(particles1, particles2, battrs=battrs, **kwargs)

    from cucount.numpy import count2, Particles, BinAttrs
    counts = run(nthreads=2)

    from jax import config
    config.update("jax_enable_x64", True)  # Currently only double precision is supported
    from cucount.jax import count2, Particles, BinAttrs
    counts = run()


def test_jaxpower():
    import jax
    from jaxpower import (
        get_mesh_attrs,
        compute_mesh2_spectrum,
        ParticleField,
        FKPField,
        create_sharding_mesh,
        BinMesh2SpectrumPoles,
        compute_fkp2_normalization,
        compute_fkp2_shotnoise
    )

    size = 10000
    data_positions, data_weights = generate_catalogs(size=size)
    randoms_positions, randoms_weights = generate_catalogs(size=size)

    # Create MeshAttrs from positions (assumed already sharded across processes)
    mattrs = get_mesh_attrs(data_positions, randoms_positions, boxpad=2., meshsize=128)
    data = ParticleField(data_positions, data_weights, attrs=mattrs, exchange=True)
    randoms = ParticleField(randoms_positions, randoms_weights, attrs=mattrs, exchange=True)
    fkp = FKPField(data, randoms)
    # Define k-bin edges and multipoles
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=(0, 2, 4))

    # Compute normalization and shot noise terms
    norm = compute_fkp2_normalization(fkp, bin=bin)
    num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)

    # Paint FKP field to mesh
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    del fkp  # cleanup

    # JIT the power spectrum function
    compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])

    # Compute P(k)
    spectrum = compute_mesh2_spectrum(mesh, bin=bin, los='firstpoint')
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)


def test_jaxrecon():
    size = 10000
    data_positions, data_weights = generate_catalogs(size=size)
    randoms_positions, randoms_weights = generate_catalogs(size=size)

    import jax.numpy as jnp
    from jaxpower import ParticleField, FKPField, get_mesh_attrs, create_sharding_mesh
    from jaxrecon.zeldovich import IterativeFFTReconstruction

    # Define FKP field = data - randoms
    mattrs = get_mesh_attrs(data_positions, randoms_positions, boxpad=1.2, cellsize=10.)
    data = ParticleField(data_positions, data_weights, attrs=mattrs, exchange=True, return_inverse=True)
    randoms = ParticleField(randoms_positions, randoms_weights, attrs=mattrs, exchange=True, return_inverse=True)
    fkp = FKPField(data, randoms)
    recon = IterativeFFTReconstruction(fkp, growth_rate=0.8, bias=2.0, los=None, smoothing_radius=15., halo_add=0)
    data_positions_rec = recon.read_shifted_positions(data.positions)
    # RecSym = remove large scale RSD from randoms
    randoms_positions_rec = recon.read_shifted_positions(randoms.positions)


def test_abacusutils():
    # abacusutils
    import abacusnbody.data.compaso_halo_catalog
    from abacusnbody.data import read_abacus
    fn = '/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.800/halo_rv_A/halo_rv_A_000.asdf'
    read_abacus.read_asdf(fn, load=['pos'])['pos']


def test_mockfactory():
    # mockfactory
    from mockfactory import Catalog


def test_desilike():
    # desilike
    from desilike.theories.galaxy_clustering import EFTLikeKaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix, BoxFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = EFTLikeKaiserTracerPowerSpectrumMultipoles(template=template)
    for param in theory.init.params.select(basename=['ct*', 'sn*']): param.update(fixed=True)
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={'ct0_2': 1., 'sn0': 1000.},
                                                         theory=theory)
    covariance = ObservablesCovarianceMatrix(observables=observable, footprints=BoxFootprint(volume=1e10, nbar=1e-2))
    observable.init.update(covariance=covariance().value())
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood.params['LRG.loglikelihood'] = likelihood.params['LRG.logprior'] = {}

    for param in likelihood.all_params.select(basename=['df', 'dm']): param.update(fixed=True)
    profiler = MinuitProfiler(likelihood, rescale=True)
    profiles = profiler.maximize(niterations=1)
    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles
    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.))
    theory()
    # theory behaves like any function for jax
    import jax
    from jax import numpy as jnp
    jac = jax.jacfwd(theory)  # jacobian function
    jac = jax.jit(jac)  # just-in-time compilation
    dpoles = jac(dict(dm=0., qpar=1., qper=1.))  # return jacobian
    vtheory = jax.vmap(lambda p: theory(p, return_derived=True))  # vectorized theory, also returning derived parameters
    poles, derived = vtheory(dict(dm=jnp.linspace(0., 0.01, 10)))
    print(poles)


def test_inference():
    # cosmosis, montepython
    assert os.getenv('COSMOSIS_STD_DIR')
    assert os.getenv('CLASS_STD_DIR')
    assert os.getenv('PLANCK_SRC_DIR')

    # cobaya
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()

    # No magic here, this is all Cobaya stuff
    params = {'Omega_m': {'prior': {'min': 0.1, 'max': 1.},
                          'ref': {'dist': 'norm', 'loc': 0.3, 'scale': 0.01},
                          'latex': r'\Omega_{m}'},
              **{name: float(cosmo[name]) for name in ['omega_b', 'H0', 'A_s', 'n_s', 'tau_reio']}}

    info_ref = {'params': params,
            'likelihood': {'planck_2018_highl_plik.TTTEEE': None, 'sn.pantheon': None},
            'theory': {'classy': {'extra_args': {'m_ncdm': float(cosmo['m_ncdm'][0]), 'N_ncdm': int(cosmo['N_ncdm']), 'N_ur': int(cosmo['N_ur'])}}}}

    info_sampler = {'evaluate': {}}

    from cobaya.model import get_model
    from cobaya.sampler import get_sampler

    info = copy.deepcopy(info_ref)
    model = get_model(info)
    get_sampler(info_sampler, model=model).run()

    info = copy.deepcopy(info_ref)
    info['likelihood'] = {'planck_2018_highl_CamSpec2021.TTTEEE': None, 'planckpr4lensing.PlanckPR4Lensing': None}
    model = get_model(info)
    get_sampler(info_sampler, model=model).run()

    info = copy.deepcopy(info_ref)
    info['likelihood'] = {'planck_NPIPE_highl_CamSpec.TTTEEE': None, 'planckpr4lensing.PlanckPR4LensingMarged': None}
    model = get_model(info)
    get_sampler(info_sampler, model=model).run()

    info = copy.deepcopy(info_ref)
    info['likelihood'] = {'planck_NPIPE_highl_CamSpec.TEEE': None, 'planckpr4lensing.PlanckPR4LensingMarged': None}
    model = get_model(info)
    get_sampler(info_sampler, model=model).run()

    info = copy.deepcopy(info_ref)
    info['likelihood'] = {'planck_2020_hillipop.TTTEEE': None, 'planckpr4lensing.PlanckPR4LensingMarged': None}
    model = get_model(info)
    get_sampler(info_sampler, model=model).run()

    info = copy.deepcopy(info_ref)
    info['likelihood'] = {'act_dr6_lenslike.ACTDR6LensLike': {'lens_only': False, 'stop_at_error': True, 'lmax': 4000, 'variant': 'act_baseline'}}
    info['theory']['classy']['extra_args'].update({'modes': 's', 'output': 'tCl, pCl, lCl'})
    info['debug'] = True
    model = get_model(info)
    get_sampler(info_sampler, model=model).run()

    info = copy.deepcopy(info_ref)
    #info['params']['yp2'] = {'prior': {'min': 0.5, 'max': 1.5}}  # for ACT
    info['likelihood'] = {'wmaplike.WMAPLike': None, 'spt3g_2020.TEEE': None}  # , 'pyactlike.ACTPol_lite_DR4': None theory deprecated
    model = get_model(info)
    get_sampler(info_sampler, model=model).run()

    info = copy.deepcopy(info_ref)
    #info['params']['yp2'] = {'prior': {'min': 0.5, 'max': 1.5}}  # for ACT
    info['likelihood'] = {'wmaplike.WMAPLike': None, 'spt3g_2022.TTTEEE': None}
    model = get_model(info)
    get_sampler(info_sampler, model=model).run()

    info = copy.deepcopy(info_ref)
    info['likelihood'] = {'act_dr6_spt_lenslike.ACTDR6LensLike': {'lens_only': False, 'stop_at_error': True, 'lmax': 4000, 'variant': 'act_baseline'}}
    info['theory']['classy']['extra_args'].update({'modes': 's', 'output': 'tCl, pCl, lCl'})
    info['debug'] = True
    model = get_model(info)
    get_sampler(info_sampler, model=model).run()

    info = copy.deepcopy(info_ref)
    rename = {'ombh2': 'omega_b', 'As': 'A_s', 'ns': 'n_s', 'tau': 'tau_reio'}
    info['params'] = {'omch2': {'prior': {'min': 0.01, 'max': 0.3},
                          'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.01},
                          'latex': r'\omega_{cdm}'},
              **{name: float(cosmo[rename.get(name, name)]) for name in ['ombh2', 'H0', 'As', 'ns', 'tau']}}
    info['likelihood'] = {'elica': {}}
    info['theory'] = {'camb': {'extra_args': {'lens_potential_accuracy': 1, 'nnu': 3.044, 'num_massive_neutrinos': 1}}}
    info['debug'] = True
    model = get_model(info)
    get_sampler(info_sampler, model=model).run()

    from pypolychord import settings


def test_desihub():
    from fiberassign.hardware import load_hardware, get_default_exclusion_margins
    from fiberassign._internal import Hardware
    from fiberassign.tiles import load_tiles
    from fiberassign.targets import Targets, TargetsAvailable, LocationsAvailable, create_tagalong, load_target_file, targets_in_tiles
    from fiberassign.assign import Assignment
    from fiberassign.utils import Logger
    from desitarget.io import read_targets_in_tiles
    import desimodel.focalplane
    import desimodel.footprint
    import desimeter


if __name__ == '__main__':
    import os
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.2'

    from mockfactory import setup_logging
    setup_logging()

    test_inference()
    test_cosmoprimo()
    test_pycorr()
    test_pypower()
    test_pyrecon()
    test_cucount()
    test_jaxpower()
    test_jaxrecon()
    test_mockfactory()
    test_desilike()
    test_inference()
    test_abacusutils()
    test_desihub()
