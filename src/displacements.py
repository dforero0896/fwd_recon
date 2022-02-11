import jax
import jax.numpy as jnp

@jax.jit
def divergence_to_displacement(divergence, box_size, smooth):

    n_bins = divergence.shape[0]
    phi_field = jnp.fft.rfftn(divergence)
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * smooth

    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2

    phi_field = jnp.where(ksq > 0., phi_field * norm / (ksq + 1e-6), 0.)
    phi_field = jax.ops.index_update(phi_field, jax.ops.index[0,0,0], 0.)

    psi_x = jnp.fft.irfftn(-1j * k[:,None, None] * phi_field, divergence.shape)
    psi_y = jnp.fft.irfftn(-1j * k[None,:,None] * phi_field, divergence.shape)
    psi_z = jnp.fft.irfftn(-1j * k[None,None,:n_bins//2 + 1] * phi_field, divergence.shape)

    return psi_x, psi_y, psi_z


@jax.jit
def divergence_to_displacement_k(phi_field, box_size, smooth):

    n_bins = phi_field.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * smooth

    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2

    phi_field = jnp.where(ksq > 0., phi_field * norm / (ksq + 1e-6), 0.)
    phi_field = jax.ops.index_update(phi_field, jax.ops.index[0,0,0], 0.)

    psi_x = jnp.fft.irfftn(-1j* k[:,None, None] * phi_field, (n_bins, n_bins, n_bins))
    psi_y = jnp.fft.irfftn(-1j* k[None,:,None] * phi_field, (n_bins, n_bins, n_bins))
    psi_z = jnp.fft.irfftn(-1j* k[None,None,:n_bins//2 + 1] * phi_field, (n_bins, n_bins, n_bins))

    return psi_x, psi_y, psi_z

@jax.jit
def zeldovich(delta, box_size, smooth):

    divergence = - delta

    psi_x, psi_y, psi_z = divergence_to_displacement(divergence, box_size, smooth)

    return psi_x, psi_y, psi_z

@jax.jit
def two_lpt(delta, box_size, smooth):

    """ See eq 2.9 https://florent-leclercq.eu/documents/thesis/Chapter2.pdf"""

    
    divergence = -delta + (1. / 7) * delta**2
    
    psi_x, psi_y, psi_z = divergence_to_displacement(divergence, box_size, smooth)

    return psi_x, psi_y, psi_z

@jax.jit
def spherical_collapse(delta, box_size, smooth):

    """ See eq 2.13 https://florent-leclercq.eu/documents/thesis/Chapter2.pdf"""

    divergence = jnp.where(delta < 3. / 2, 3 * (jnp.sqrt(1. - (2./3) * delta) - 1.), -3.)
    
    psi_x, psi_y, psi_z = divergence_to_displacement(divergence, box_size, smooth)

    return psi_x, psi_y, psi_z

@jax.jit
def aug_lpt(delta, box_size, smooth, interp_smooth):

    """ See eq 2.17 https://florent-leclercq.eu/documents/thesis/Chapter2.pdf"""

    n_bins = delta.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * interp_smooth

    interp_kernel = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))


    divergence_sc_k = jnp.fft.rfftn(jnp.where(delta < 3. / 2, 3 * (jnp.sqrt(1. - (2./3) * delta) - 1.), -3.))
    divergence_2lpt_k = jnp.fft.rfftn(-delta + (1. / 7) * delta**2)
    
    divergence_k = interp_kernel * divergence_2lpt_k + (1. - interp_kernel) * divergence_sc_k

    
    psi_x, psi_y, psi_z = divergence_to_displacement_k(divergence_k, box_size, smooth)

    return psi_x, psi_y, psi_z


@jax.jit
def interpolate_field(disp, positions, xmin, ymin, zmin, n_bins, box_size):

    bin_size = box_size / n_bins
    xpos = (positions[:,0] - xmin) / bin_size
    ypos = (positions[:,1] - ymin) / bin_size
    zpos = (positions[:,2] - zmin) / bin_size

    i = jnp.int32(xpos)
    j = jnp.int32(ypos)
    k = jnp.int32(zpos)

    ddx = xpos - i
    ddy = ypos - j
    ddz = zpos - k

    def weights(ddx, ddy, ddz, ii, jj, kk):
        return (((1 - ddx) + ii * (-1 + 2 * ddx)) * 
                ((1 - ddy) + jj * (-1 + 2 * ddy)) *
                ((1 - ddz) + kk * (-1 + 2 * ddz)))

    shifts = jnp.zeros((positions.shape[0]))

    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                shifts = jax.ops.index_add(shifts, jax.ops.index[:], weights(ddx, ddy, ddz, ii, jj, kk) * disp[(i + ii) % n_bins, (j + jj) % n_bins, (k + kk) % n_bins])            

    
    return shifts

def gaussian_field(grid, kf, Pkf, Rayleigh_sampling, 
                      key, box_size):
    
    """ Taken from https://github.com/bsciolla/gaussian-random-fields"""
    
    phase_prefac = 2.0*jnp.pi
    k_prefac     = 2.0*jnp.pi/box_size
    inv_max_rand = 1.0
    zero = 0. + 1j * 0.
    dims = grid
    k_bins, middle = kf.shape[0], grid//2

    # define the density field in Fourier space
    delta_k = jnp.zeros((grid, grid, grid), dtype=jnp.complex64)
    
    key_real, key_imag = jax.random.split(key)
    
    
    dims_range = jnp.arange(dims)
    
    ki = jnp.where(dims_range > middle, dims_range - dims, dims_range)
    
    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, dims))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, dims))
    kz = jnp.broadcast_to(ki[None, None, :], (dims, dims, dims))
    
    
    kmod = jnp.sqrt(kx**2 + ky**2 + kz**2) * k_prefac
    
    
    Pk = jnp.interp(kmod.flatten(), kf, Pkf).reshape(delta_k.shape)#
    
    Pk *= (grid**2/box_size)**3 
    
    amplitude = jnp.sqrt(Pk)   
    amplitude = jax.ops.index_update(amplitude, jax.ops.index[0,0,0], 0.)
    real_part = jax.random.normal(key_real, shape = (grid, grid, grid))
    imag_part = jax.random.normal(key_imag, shape = (grid, grid, grid))
    phase = jnp.arctan(imag_part / real_part)
    noise = real_part + 1j * imag_part
    #noise = (real_part**2 + imag_part**2)**0.5*jnp.exp(1j * phase)
    
    delta = jnp.fft.ifftn(noise * amplitude).real
    
    
    return delta


def get_positions(key, number_particles, bin_size):
    key, subkey = jax.random.split(key)
    Rs = 2 * jax.random.uniform(subkey, shape = (number_particles,3)) - 1
    Rs = jnp.sign(Rs) * (1 - jnp.sqrt(jnp.abs(Rs)))
    
    return Rs * bin_size

def populate_field(rho, n_bins, box_size, density, key):
    bin_size = box_size / n_bins
    cell_volume = bin_size**3
    mean_obj_per_cell = cell_volume * density
    
    rho *= mean_obj_per_cell / rho.mean()
    nonzero = (rho != 0).sum()
    sorted_rho = jnp.argsort(rho.ravel())[::-1]#[:nonzero+1]
    
    flat_rho = rho.flatten()[sorted_rho]
    #grid_centers = jnp.arange(0, box_size, box_size / n_bins) + 0.5 * box_size / n_bins
    #grid_centers = jnp.array(jnp.meshgrid(grid_centers, grid_centers, grid_centers)).reshape(3, n_bins**3).T
    grid_centers = jnp.array(jnp.unravel_index(sorted_rho, (n_bins, n_bins, n_bins))).T.astype(jnp.float32) * bin_size + 0.5 * bin_size
    
    
    key, subkey = jax.random.split(key)
    number_objects = jax.random.poisson(subkey, flat_rho, shape=flat_rho.shape)
    print(number_objects.sum())
    
    displacements = get_positions(key, number_objects.sum(), bin_size)

    coords = jnp.repeat(grid_centers, number_objects, axis=0) + displacements
    return (coords + box_size) % box_size


def field_smooth(field, scale, box_size):
    n_bins = field.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * scale

    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    
    return jnp.fft.irfftn(norm * jnp.fft.rfftn(field), field.shape)