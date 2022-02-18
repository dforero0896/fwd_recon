import jax
import jax.numpy as jnp


def interp_loglog(x, xp, yp):
    return jnp.interp(jnp.log(x), jnp.log(xp), yp)

def powspec_smooth(k, B_nw, nuisance_a, klin, plin, knw, plin_nw):

    exponents = jnp.arange(-2, 3, 1)
    
    powspec_lin_nw = interp_loglog(k, knw, plin_nw)
    powspec_lin = interp_loglog(k, klin, plin)
    k_exp = jnp.expand_dims(k, -1)**exponents
    
    powspec_sm = B_nw**2 * powspec_lin_nw + jnp.dot(k_exp, nuisance_a)

    return powspec_sm 


def get_nuisance(kobs, pobs, alpha, Sigma_nl, B_nw, klin, plin, knw, plin_nw):


    exponents = jnp.arange(-2, 3, 1)

    powspec_lin_nw = interp_loglog(kobs, knw, plin_nw)
    powspec_lin = interp_loglog(kobs, klin, plin)

    k_exp = jnp.expand_dims(kobs, -1)**exponents
    
    O_factor = interp_loglog(kobs / alpha, klin, plin) / interp_loglog(kobs / alpha, knw, plin_nw)
    O_damp = 1. + (O_factor - 1.) * jnp.exp(-0.5 * kobs**2 * Sigma_nl**2)

    nuisance_term = pobs / O_damp - B_nw**2 * powspec_lin_nw
    
    design_matrix = k_exp.T.dot(k_exp)
    vector = k_exp.T.dot(nuisance_term)
    nuisance_a = jnp.linalg.solve(design_matrix, vector)

    return nuisance_a, nuisance_term

def chisq(kobs, pobs, alpha, Sigma_nl, B_nw, klin, plin, knw, plin_nw, box_size):


    exponents = jnp.arange(-2, 3, 1)

    powspec_lin_nw = interp_loglog(kobs, knw, plin_nw)
    powspec_lin = interp_loglog(kobs, klin, plin)

    k_exp = jnp.expand_dims(kobs, -1)**exponents
    
    O_factor = interp_loglog(kobs / alpha, klin, plin) / interp_loglog(kobs / alpha, knw, plin_nw)
    O_damp = 1. + (O_factor - 1.) * jnp.exp(-0.5 * kobs**2 * Sigma_nl**2)
    
    nuisance_term = pobs / O_damp - B_nw**2 * powspec_lin_nw
    design_matrix = k_exp.T.dot(k_exp)
    vector = k_exp.T.dot(nuisance_term)
    nuisance_a = jnp.linalg.solve(design_matrix, vector)

    precision = jnp.diag(estimate_pk_variance(kobs, pobs, box_size, 0., kobs[1] - kobs[0])**-1)
    powspec_sm = B_nw**2 * powspec_lin_nw + jnp.dot(k_exp, nuisance_a)

    error = pobs - powspec_sm*O_damp 

    return error.T.dot(precision.dot(error))


def powspec_model(k, alpha, Sigma_nl, B_nw, nuisance_a, klin, plin, knw, plin_nw):

    exponents = jnp.arange(-2, 3, 1)

    powspec_lin_nw = jnp.interp(k, knw, plin_nw)
    powspec_lin = jnp.interp(k, klin, plin)
    k_exp = jnp.expand_dims(k, -1)**exponents
    O_factor = powspec_lin / (powspec_lin_nw)# + np.dot(k_exp, nuisance_a))
    O_damp = 1. + (O_factor - 1.) * jnp.exp(-0.5 * k**2 * Sigma_nl**2)

    powspec_sm = B_nw**2 * powspec_lin_nw + jnp.dot(k_exp, nuisance_a)

    return powspec_sm * O_damp

def estimate_pk_variance(k, pk, box_size, shot_noise, dk):
    """
    Must use linear k bins.
    Eq. 17 in https://arxiv.org/pdf/2109.15236.pdf
    """
    return (2*jnp.pi)**3 / box_size**3 * (2 * (pk + shot_noise)**2 / (4*jnp.pi*k**2 * dk))


if __name__ == '__main__':
    import numpy as np
    import proplot as pplt
    knw, plin_nw = np.loadtxt("data/Albert_Pnw.dat", unpack=True)
    klin, plin = np.loadtxt("data/Albert_Plin.dat", unpack=True)
    knw = jax.device_put(knw)
    klin = jax.device_put(klin)
    plin_nw = jax.device_put(plin_nw)
    plin = jax.device_put(plin)

    pk_init = jax.device_put(np.load("data/tests/pk_init_corrected.npy"))
    pk_ev = jax.device_put(np.load("data/tests/pk_evolv_corrected.npy"))
    pk_now = jax.device_put(np.load("data/tests/pk_now.npy"))
    nuisance_init, nuisance_term = get_nuisance(pk_now[:,0], pk_now[:,1], 1., 0., 1., klin, plin, knw, plin_nw)
    psmooth_init = powspec_smooth(pk_now[:,0], 0., nuisance_init, klin, plin, knw, plin_nw)


    fig, ax = pplt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
    #ax[0].plot(pk_now[:,0], pk_now[:,1] / psmooth_init - 1)
    #ax[0].plot(pk_now[:,0], pk_now[:,1] / interp_loglog(pk_now[:,0], knw, plin_nw))
    
    #ax[0].plot(pk_now[:,0], interp_loglog(pk_now[:,0], knw, plin_nw))
    #ax[0].plot(pk_now[:,0], psmooth_init)
    ax[0].plot(pk_now[:,0], nuisance_term)
    #ax[0].plot(klin, plin)
    #ax[0].plot(knw, plin_nw)
    #ax[0].plot(pk_init[:,0], psmooth_init)
    #ax[0].format(xscale='log', yscale='log')
    fig.savefig("plots/smooth.pdf")