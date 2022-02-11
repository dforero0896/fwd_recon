from nbodykit.lab import *
import numpy as np

z_init = 10.
box_size = 2500
n_bins = 512


nbk_cosmo = cosmology.Planck15
nbk_plin = cosmology.LinearPower(nbk_cosmo, z_init, transfer='CLASS')
nbk_cat = LogNormalCatalog(Plin=nbk_plin, nbar=3e-3, BoxSize=box_size, Nmesh=n_bins, bias=1., seed=42)
cat = np.c_[nbk_cat['Position'].compute(), nbk_cat['Velocity'].compute()]
print(cat)
np.save("data/data/lognormal_ic_tracer.npy", cat)

