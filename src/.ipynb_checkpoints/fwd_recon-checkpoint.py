import numpy as np
import time
import os
import jax
import jax.numpy as jnp
from jax.experimental import loops
from jax.example_libraries import optimizers
import sys
#sys.path.insert(0, "/home/astro/dforero/projects/jax_cosmo") #Use my local jax_cosmo with correlations module
sys.path.insert(0, "/global/u1/d/dforero/projects/jax-powspec") #Use my local jax_cosmo with correlations module
sys.path.insert(0, "/global/u1/d/dforero/projects/jax_cosmo") #Use my local jax_cosmo with correlations module
import jax_cosmo as jc
from jax_cosmo.correlations import xicalc_trapz
import numpy as np
import pandas as pd

import proplot as pplt

from jax_powspec.mas import cic_mas, cic_mas_vec
from jax_powspec.correlations import powspec, powspec_vec, xi_vec, powspec_vec_fundamental, xi_vec_fundamental, bispec, compute_all_correlations
from src import displacements

import MAS_library as MASL
import Pk_library as PKL
#jax.config.update("jax_debug_nans", True)

box_size = 2000.
data_fn = "/pscratch/sd/d/dforero/projects/fwd-recon/data/CATALPTCICz0.466G960S1005638091.dat"
fig, ax = pplt.subplots(nrows=4, ncols=3, sharex=False, sharey=False)
particles = jax.device_put(pd.read_csv(data_fn, delim_whitespace=True, engine='c', usecols=(0,1,2)).values)
box_mask = (particles[:,:3] < box_size).all(axis=1)
particles = particles[box_mask]
cosmo = jc.Planck15()
key = jax.random.PRNGKey(42)
n_part = particles.shape[0]
w0 = jnp.ones(particles.shape[0])

shot_noise = box_size**3 / n_part
z = 0.466
z_init = 100.
n_bins = 300
bias = 2.
growth_rate = jc.background.growth_rate(cosmo, jnp.atleast_1d(1. / (1 + z)))[0]
growth_factor = jc.background.growth_factor(cosmo, jnp.atleast_1d(1. / (1 + z)))[0] / jc.background.growth_factor(cosmo, jnp.atleast_1d(1. / (1 + z_init)))[0]
beta = growth_rate / bias
k1, k2 = 0.05, 0.1
k_ny = jnp.pi * n_bins / box_size
k_edges = jnp.arange(2. * jnp.pi / box_size, k_ny, 2. * jnp.pi / box_size)
s_edges = jnp.linspace(5, 200, 41)
smooth=0.
interp_smooth=4.
klin = np.logspace(-3, 0, 2048)
plin_i = jc.power.linear_matter_power(cosmo, klin, a = 1. / (1 + z_init), transfer_fn=jc.transfer.Eisenstein_Hu)

# Build observed delta field
delta_now = jnp.zeros((n_bins, n_bins, n_bins))
delta_now = cic_mas_vec(delta_now,
                particles[:,0], particles[:,1], particles[:,2], w0, 
                n_part, 
                0., 0., 0.,
                box_size,
                n_bins,
                True)
delta_now /= delta_now.mean()
delta_now -= 1.
delta_now_var = delta_now.std()**2
k, pk_now, _ = powspec_vec(delta_now, box_size, k_edges)
s, xi_now, _ = xi_vec(delta_now, box_size, s_edges)
pk_now = pk_now.at[:,0].set(pk_now[:,0] - shot_noise)



# Build delta field of initial condition

delta_init = delta_now / bias / growth_factor

k, pk_init, _ = powspec_vec(delta_init, box_size, k_edges)
s, xi_init, _ = xi_vec(delta_init, box_size, s_edges)

# Init lagrangian positions

pos_lagrangian = jnp.arange(0, box_size, box_size / n_bins)
pos_lagrangian = jnp.array(jnp.meshgrid(pos_lagrangian, pos_lagrangian, pos_lagrangian)).reshape(3, n_bins**3).T


shot_noise = pos_lagrangian.shape[0] / box_size**3




ax[0].imshow(delta_init.mean(axis=0), colorbar='right')
ax[0].format(title='Init')
ax[2].imshow(delta_now.mean(axis=0), colorbar='right', vmin=-0.5, vmax=0.9)
ax[2].format(title='Now')
ax[4].plot(k, pk_now[:,0], label='Now')
ax[4].plot(k, pk_init[:,0], label='Init')
ax[6].plot(s, s**2*xi_now[:,0], label='Now')
ax[7].plot(s, s**2*(bias**2 * growth_rate**2 * xi_init[:,0]), label='b^2D^2 Init')


def evolve_lagrangian_disp(pos_lagrangian, disp, growth_factor):
    pos = pos_lagrangian.copy()
    psi_x, psi_y, psi_z = disp
    #psi_x, psi_y, psi_z = displacements.aug_lpt(delta, box_size, smooth, interp_smooth)
    pos = jax.ops.index_add(pos, jax.ops.index[:,0], growth_factor * displacements.interpolate_field(psi_x, pos_lagrangian, 0., 0., 0., n_bins, box_size))
    pos = jax.ops.index_add(pos, jax.ops.index[:,1], growth_factor * displacements.interpolate_field(psi_y, pos_lagrangian, 0., 0., 0., n_bins, box_size))
    pos = jax.ops.index_add(pos, jax.ops.index[:,2], growth_factor * displacements.interpolate_field(psi_z, pos_lagrangian, 0., 0., 0., n_bins, box_size))

    pos = (pos + box_size) % box_size
    
    del psi_x
    del psi_y
    del psi_z

    # Build field evolved from lagrangian positions
    delta_ev = jnp.zeros((n_bins, n_bins, n_bins))
    delta_ev = cic_mas_vec(delta_ev,
                    pos[:,0], pos[:,1], pos[:,2], jnp.broadcast_to([1.], pos.shape[0]), 
                    pos.shape[0], 
                    0., 0., 0.,
                    box_size,
                    n_bins,
                    True)
    delta_ev /= delta_ev.mean()
    delta_ev -= 1.
    return delta_ev


def evolve_lagrangian(pos_lagrangian, delta, growth_factor):
    pos = pos_lagrangian.copy()
    psi_x, psi_y, psi_z = displacements.zeldovich(delta, box_size, smooth)
    #psi_x, psi_y, psi_z = displacements.two_lpt(delta, box_size, smooth)
    #spsi_x, psi_y, psi_z = displacements.aug_lpt(delta, box_size, smooth, interp_smooth)
    pos = jax.ops.index_add(pos, jax.ops.index[:,0], growth_factor * displacements.interpolate_field(psi_x, pos_lagrangian, 0., 0., 0., n_bins, box_size))
    pos = jax.ops.index_add(pos, jax.ops.index[:,1], growth_factor * displacements.interpolate_field(psi_y, pos_lagrangian, 0., 0., 0., n_bins, box_size))
    pos = jax.ops.index_add(pos, jax.ops.index[:,2], growth_factor * displacements.interpolate_field(psi_z, pos_lagrangian, 0., 0., 0., n_bins, box_size))

    pos = (pos + box_size) % box_size
    
    del psi_x
    del psi_y
    del psi_z

    # Build field evolved from lagrangian positions
    delta_ev = jnp.zeros((n_bins, n_bins, n_bins))
    delta_ev = cic_mas_vec(delta_ev,
                    pos[:,0], pos[:,1], pos[:,2], jnp.broadcast_to([1.], pos.shape[0]), 
                    pos.shape[0], 
                    0., 0., 0.,
                    box_size,
                    n_bins,
                    True)
    delta_ev /= delta_ev.mean()
    delta_ev -= 1.
    return delta_ev

#delta_ev = bias * evolve_lagrangian_disp(pos_lagrangian, displacements.zeldovich(delta_init, box_size, smooth), growth_factor)
delta_ev = bias * evolve_lagrangian(pos_lagrangian, delta_init, growth_factor)
k, pk_ev, _ = powspec_vec(delta_ev, box_size, k_edges)
pk_ev = pk_ev.at[:,0].set(pk_ev[:,0] - shot_noise)
s, xi_ev, _ = xi_vec(delta_ev, box_size, s_edges)
ax[4].plot(k, pk_ev[:,0], label='Ev. Zeld.', ls='--')
ax[6].plot(s, s**2*xi_ev[:,0], label='Ev. Zeld.', ls = '--')

ax[4].format(xscale='log', yscale='log')
#ax[4].legend(loc='top')
#ax[6].legend(loc='top')

#fig.savefig("plots/fwd_recon.png", dpi=300); exit()
#@jax.jit


def loss(conv_kernel, delta_init):
    
    init = jnp.fft.irfftn(conv_kernel * jnp.fft.rfftn(delta_init), delta_init.shape)
    delta_ev = bias * evolve_lagrangian(pos_lagrangian, init, growth_factor)

    k, pk_ev, _ = powspec_vec(delta_ev, box_size, k_edges)
    pk_ev = pk_ev.at[:,0].set(pk_ev[:,0] - shot_noise)
    
    
    pixelwise_loss = 1e5 * jnp.nanmean(jnp.square(delta_ev - delta_now)) / delta_now_var
    mono_k_loss = 1e1 * jnp.nanmean((pk_ev[:,0] - pk_now[:,0])**2) #+ jnp.nanmean((pk_in[:,0] - pk_now[:,0] / bias**2)**2)

    return  pixelwise_loss + mono_k_loss 

  
    




learning_rate = 1e-1
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
conv_kernel = jnp.ones((n_bins, n_bins, n_bins//2 + 1))

opt_state = opt_init(conv_kernel)


@jax.jit
def step(step, opt_state):
    grads = jax.grad(loss)(get_params(opt_state), delta_init)
    opt_state = opt_update(step, grads, opt_state)
    return opt_state
num_steps = 200



s = time.time()
print("Training...", flush=True)
opt_state = jax.lax.fori_loop(0, num_steps, step, opt_state)   
print(f"Training took {time.time() - s} s", flush=True)


new_conv_kernel = get_params(opt_state)

delta_init = jnp.fft.irfftn(new_conv_kernel * jnp.fft.rfftn(delta_init), delta_init.shape)
k, pk_init, _ = powspec_vec(delta_init, box_size, k_edges)
s, xi_init, _ = xi_vec(delta_init, box_size, s_edges)
delta_ev = bias * evolve_lagrangian(pos_lagrangian, delta_init, growth_factor)
k, pk_ev, _ = powspec_vec(delta_ev, box_size, k_edges)
pk_ev = pk_ev.at[:,0].set(pk_ev[:,0] - shot_noise)
s, xi_ev, _ = xi_vec(delta_ev, box_size, s_edges)


os.makedirs("data/tests/", exist_ok = True)
np.save("data/tests/pk_init_corrected.npy", np.c_[k, pk_init])
np.save("data/tests/pk_evolv_corrected.npy", np.c_[k, pk_ev])
np.save("data/tests/pk_now.npy", np.c_[k, pk_now])

ax[1].imshow(delta_init.mean(axis=0), colorbar='right')
ax[1].format(title='Init. Corrected')
ax[3].imshow(delta_ev.mean(axis=0), colorbar='right', vmin=-0.5, vmax=0.9)
ax[3].format(title='Evolved Lag.')
ax[5].imshow(jnp.nanmean(displacements.field_smooth(jnp.abs(delta_now  - delta_ev), 10, box_size), axis=0), colorbar='right', norm='linear')
ax[5].format(title='Difference')
ax[4].plot(k, pk_init[:,0], label='Init Corrected')
ax[4].plot(k, pk_ev[:,0], label='Evolved Lag.', ls = '--')

ax[7].plot(s, s**2*(bias**2 * growth_rate**2 * xi_init[:,0]), label='b^2D^2 Init Corrected')
ax[6].plot(s, s**2*xi_ev[:,0], label='Evolved Lag.', ls = '--')

ax[4].format(xscale='log', yscale='log')
ax[4].legend(loc='top')
ax[6].legend(loc='top')
ax[7].legend(loc='top')
ax[8].imshow(jnp.fft.fftshift(new_conv_kernel).mean(axis=2), colorbar='right', norm='symlog')
ax[8].format(title='Kernel')
fig.savefig("plots/fwd_recon.pdf", dpi=300)