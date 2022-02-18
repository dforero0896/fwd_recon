import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from helpers import powspec_smooth, get_nuisance, chisq, powspec_model, estimate_pk_variance

if __name__ == '__main__':
    
    import numpy as np
    import time
    import proplot as pplt
    knw, plin_nw = np.loadtxt("data/Albert_Pnw.dat", unpack=True)
    klin, plin = np.loadtxt("data/Albert_Plin.dat", unpack=True)
    knw = jax.device_put(knw)
    klin = jax.device_put(klin)
    plin_nw = jax.device_put(plin_nw)
    plin = jax.device_put(plin)
    box_size = 2000

    
    
    fig, ax = pplt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    pax = ax.panel('bottom')
    def fit_pk(pk_obs, ax, pax, **ax_kwargs):
        kmin = 0.01; kmax = 0.4
        bkp_pk_obs = pk_obs.copy()
        mask = (pk_obs[:,0] > kmin) & (pk_obs[:,0] < kmax)
        print(pk_obs.max(axis=0))
        pk_obs = pk_obs[mask]
        
        
        @jax.jit
        def loss(params):
            Sigma_nl, B_nw = params
            return  1e-5 * chisq(pk_obs[:,0], pk_obs[:,1], 1., Sigma_nl, B_nw, klin, plin, knw, plin_nw, box_size)

        learning_rate = 1e-1
        init_params = (1e0, 2.)
        opt_init, opt_update, get_params = optimizers.adam(learning_rate)

        opt_state = opt_init(init_params)


        @jax.jit
        def step(step, opt_state):
            grads = jax.grad(loss)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return opt_state
        num_steps = int(1e4)



        s = time.time()
        print("Training...", flush=True)
        opt_state = jax.lax.fori_loop(0, num_steps, step, opt_state)   
        print(f"Training took {time.time() - s} s", flush=True)
        final_params = get_params(opt_state)
        print(final_params)
        print(f"Final loss: {loss(final_params):.3e}", flush=True)

        #pk_obs = bkp_pk_obs
        nuisance, nuisance_term = get_nuisance(pk_obs[:,0], pk_obs[:,1], 1., *final_params, klin, plin, knw, plin_nw)
        psmooth = powspec_smooth(pk_obs[:,0], final_params[1], nuisance, klin, plin, knw, plin_nw)
        pmodel = powspec_model(pk_obs[:,0], 1., *final_params, nuisance, klin, plin, knw, plin_nw)

        std = jnp.sqrt(estimate_pk_variance(pk_obs[:,0], pk_obs[:,1], box_size, 0., pk_obs[:,0][1] - pk_obs[:,0][0]))
        
        p = ax[0].plot(pk_obs[:,0], pmodel / psmooth - 1, **ax_kwargs, ls = '--', zorder=0)
        ax[0].errorbar(pk_obs[:,0], pk_obs[:,1] / psmooth - 1, yerr = std / psmooth, lw = 0, elinewidth=1, errorevery=3, c = p[0].get_color(),  **ax_kwargs, zorder=10, alpha=0.3)
        p = ax[1].plot(pk_obs[:,0], pmodel, **ax_kwargs, ls='--', zorder=0)
        ax[1].errorbar(pk_obs[:,0], pk_obs[:,1], yerr = std, lw = 0, elinewidth=1, errorevery=3, c = p[0].get_color(), **ax_kwargs, zorder=10, alpha=0.3)
        
        pax[1].plot(pk_obs[:,0], (pmodel-pk_obs[:,1]) / std, **ax_kwargs, zorder=0)
        #ax[0].plot(pk_obs[:,0], pk_obs[:,1] )
        #ax[0].plot(pk_obs[:,0], pk_obs[:,1] / interp_loglog(pk_obs[:,0], knw, plin_nw))

        #ax[0].plot(pk_obs[:,0], interp_loglog(pk_obs[:,0], knw, plin_nw))
        #ax[0].plot(pk_obs[:,0], psmooth)
        #ax[0].plot(pk_obs[:,0], nuisance_term)
        #ax[0].plot(klin, plin)
        #ax[0].plot(knw, plin_nw)
        #ax[0].plot(pk_init[:,0], psmooth_init)
        #ax[1].format(xscale='log', yscale='linear')
        ax[1].format(xscale='log', yscale='log', xlabel=r'$k$', ylabel=r'$P(k)$')
        ax[0].format(xscale='linear', yscale='linear', xlabel=r'$k$', ylabel=r'$P / P_{\rm smooth}(k) - 1$')
        pax[1].format(xlabel=r'$k$', ylabel=r'($P_{\rm model} - P_{\rm obs}) / \sigma (k)$')
        return pmodel, psmooth, pk_obs[:,0], pk_obs[:,1], std
        
    
    #pk_obs = jax.device_put(np.load("data/tests/pk_now.npy"))
    #pk_obs = jax.device_put(np.load("data/tests/pk_evolv_corrected.npy"))
    pk_obs = jax.device_put(np.load("data/tests/pk_init_corrected.npy"))
    pk_obs = pk_obs.at[:,1].set(1e4 * pk_obs[:,1])
    
    pmodel_init, psmooth_init , k, pobs_init, std_init= fit_pk(pk_obs, ax, pax, label='1e4 init recon')    
    
    pk_obs = jax.device_put(np.load("data/tests/pk_now.npy"))
    
    pmodel_now, psmooth_now, k , pobs_now,  std_now= fit_pk(pk_obs, ax, pax, label='now')    
    
    pk_obs = jax.device_put(np.load("data/tests/pk_evolv_corrected.npy"))
    
    pmodel_evolved, psmooth_evolved, k , pobs_evolved, std_evolved= fit_pk(pk_obs, ax, pax, label='evolved recon')  
    
    
    pax[0].plot(k, (pmodel_init/psmooth_init - pmodel_now/psmooth_now) , label='init')
    pax[0].plot(k, (pmodel_evolved/psmooth_evolved - pmodel_now/psmooth_now), label='evolved')
    
    
    #pax[0].errorbar(k, (pobs_init/psmooth_init - pobs_now/psmooth_now) , yerr = std_init/psmooth_init, label='init', lw=0)
    #pax[0].errorbar(k, (pobs_evolved/psmooth_evolved - pobs_now/psmooth_now) , yerr = std_evolved/psmooth_evolved, label='evolved', lw=0)
    
    pax[0].format(xlabel=r'$k$', ylabel=r'($S_{\rm model} - S_{\rm model,now})$')
    
    ax[0].legend(loc='top')
    ax[1].legend(loc='top')
    
    fig.savefig("plots/smooth.pdf")

    
    

    
    
    
    