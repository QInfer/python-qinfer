..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _timedep_guide:
    
.. currentmodule:: qinfer

Learning Time-Dependent Models
==============================

Time-Dependent Parameters
-------------------------

In addition to learning static parameters, **QInfer** can be used
to learn the values of parameters that change stochastically as a
function of time. In this case, the model parameter vector
:math:`\vec{x}` is interpreted as a time-dependent vector
:math:`\vec{x}(t)` representing the state of an underlying process.
The resulting statistical problem is often referred to as
*state-space estimation*. By using an appropriate resampling
algorithm, such as the Liu-West algorithm [LW01]_, state-space
and static parameter estimation can be combined such that 
a subset of the components of :math:`\vec{x}` are allowed to
vary with time.

**QInfer** represents state-space filtering by the use of
the :meth:`Simulatable.update_timestep` method, which samples
how a model parameter vector is updated as a function of time.
In particular, this method is used by :class:`~qinfer.SMCUpdater` to draw samples from the
distribution :math:`f`

.. math::

    \vec{x}(t_{\ell+1}) \sim f(\vec{x}(t_{\ell}, e(t_{\ell})),

where :math:`t_{\ell}` is the time at which the experiment :math:`e_{\ell}`
is measured, and where :math:`t_{\ell+1}` is the time step immediately
following :math:`t_{\ell}`. As this distribution is in general dependent
on the experiment being performed, :meth:`~Simulatable.update_timestep`
is vectorized in a manner similar to :meth:`~Model.likelihood` (see
:ref:`models_guide` for details). That is,
given a tensor :math:`X_{i,j}` of model parameter vectors and a vector
:math:`e_k` of experiments,
:meth:`~Simulatable.update_timestep` returns a tensor :math:`X_{i,j,k}'`
of sampled model parameters at the next time step.

Random Walk Models
------------------

As an example, :class:`RandomWalkModel` implements :meth:`~Simulatable.update_timestep`
by taking as an input a :class:`Distribution` specifying steps
:math:`\Delta \vec{x} = \vec{x}(t + \delta t) - \vec{x}(t)`. An
instance of :class:`RandomWalkModel` decorates another model in
a similar fashion to :ref:`other derived models <models_guide_derived>`.
For instance, the following code declares a precession model in
which the unknown frequency :math:`\omega` changes by a normal
random variable with mean 0 and standard deviation 0.005 after
each measurement. 

>>> from qinfer import (
...     SimplePrecessionModel, RandomWalkModel, NormalDistribution
... )
>>> model = RandomWalkModel(
...     underlying_model=SimplePrecessionModel(),
...     step_distribution=NormalDistribution(0, 0.005 ** 2)
... )

We can then draw simulated trajectories for the true and estimated
value of :math:`\omega` using a minor modification to the updater loop
discussed in :ref:`smc_guide`.

.. plot::

    model = RandomWalkModel(
        # Note that we set a minimum frequency of negative
        # infinity to prevent problems if the random walk
        # causes omega to cross zero.
        underlying_model=SimplePrecessionModel(min_freq=-np.inf),
        step_distribution=NormalDistribution(0, 0.005 ** 2)
    )
    prior = UniformDistribution([0, 1])
    updater = SMCUpdater(model, 2000, prior)

    expparams = np.empty((1, ), dtype=model.expparams_dtype)

    true_trajectory = []
    est_trajectory = []

    true_params = prior.sample()

    for idx_exp in range(400):
        # We don't want to grow the evolution time to be arbitrarily
        # large, so we'll instead choose a random time up to some
        # maximum. 
        expparams[0] = np.random.random() * 10 * np.pi
        datum = model.simulate_experiment(true_params, expparams)
        updater.update(datum, expparams)

        # We index by [:, :, 0] to pull off the index corresponding
        # to experiment parameters.
        true_params = model.update_timestep(true_params, expparams)[:, :, 0]

        true_trajectory.append(true_params[0])
        est_trajectory.append(updater.est_mean())

    plt.plot(true_trajectory, label='True')
    plt.plot(est_trajectory, label='Est.')
    plt.legend()
    plt.xlabel('# of Measurements')
    plt.ylabel(r'$\omega$')

    plt.show()


Specifying Custom Time-Step Updates
-----------------------------------

The :class:`RandomWalkModel` example above is somewhat unrealistic,
however, in that the step distribution is independent of the evolution
time. For a more reasonable noise process, we would expect that
:math:`\mathbb{V}[\omega(t + \Delta t) - \omega(t)] \propto \Delta t`.
We can subclass :class:`SimplePrecessionModel` to add this behavior
with a custom :meth:`~Simulatable.update_timestep` implementation.

.. plot::

    class DiffusivePrecessionModel(SimplePrecessionModel):
        diffusion_rate = 0.0005 # We'll multiply this by
                                 # sqrt(time) below.

        def update_timestep(self, modelparams, expparams):
            step_std_dev = self.diffusion_rate * np.sqrt(expparams)
            steps = step_std_dev * np.random.randn(
                # We want the shape of the steps in omega
                # to match the input model parameter and experiment
                # parameter shapes.
                # The axis of length 1 represents that this model
                # has only one model parameter (omega).
                modelparams.shape[0], 1, expparams.shape[0]
            ) 
            # Finally, we add a new axis to the input model parameters
            # to match the experiment parameters.
            return modelparams[:, :, np.newaxis] + steps 

    model = DiffusivePrecessionModel()
    prior = UniformDistribution([0, 1])
    updater = SMCUpdater(model, 2000, prior)

    expparams = np.empty((1, ), dtype=model.expparams_dtype)

    true_trajectory = []
    est_trajectory = []

    true_params = prior.sample()

    for idx_exp in range(400):
        expparams[0] = np.random.random() * 10 * np.pi
        datum = model.simulate_experiment(true_params, expparams)
        updater.update(datum, expparams)

        true_params = model.update_timestep(true_params, expparams)[:, :, 0]

        true_trajectory.append(true_params[0])
        est_trajectory.append(updater.est_mean())

    plt.plot(true_trajectory, label='True')
    plt.plot(est_trajectory, label='Est.')
    plt.legend()
    plt.xlabel('# of Measurements')
    plt.ylabel(r'$\omega$')

    plt.show()
    
Learning Walk Parameters
------------------------

In the above examples, the diffusion distribution was treated as exactly 
known by the model. We can also parameterize this distribution, adding its 
parameters to model to be learned as well. :class:`GaussianRandomWalkModel` 
is a built in model similar to :class:`RandomWalkModel`. It is more 
restrictive in the sense that it is limited to gaussian time-step updates,
but more general in that it has the ability to automatically append a 
parameterization of the gaussian time-step distribution, either diagonal or 
dense, to the underlying model.

For example suppose that we have a coin whose bias is taking a random walk 
in time with an unknown diffusion constant. 
To avoid exiting the allowable space of biases, :math:`[0,1]`, 
we transform to inverse-logit space before taking a gaussian step, and 
transform back to the probability interval after each step.

.. plot::

    import numpy as np
    from scipy.special import expit, logit
    from qinfer import (
       CoinModel, BinomialModel, GaussianRandomWalkModel, 
       UniformDistribution, SMCUpdater
    )
    
    # Put a random walk on top of a binomial coin model
    model = GaussianRandomWalkModel(
       BinomialModel(CoinModel()),
       model_transformation=(logit, expit)
    )

    # Generate some data with a true diffusion 0.05
    true_sigma_p = 0.05
    Nbin = 10
    p = expit(logit(0.5) + np.cumsum(true_sigma_p * np.random.normal(size=300)))
    data = np.random.binomial(Nbin, 1-p)

    # Analyse the data
    prior = UniformDistribution([[0.2,0.8],[0,0.1]])
    u = SMCUpdater(model, 10000, prior)
    ests, stds = np.empty((data.size+1, 2)), np.empty((data.size+1, 2))
    ts = np.arange(ests.shape[0])
    ests[0,:] = u.est_mean()
    for idx in range(data.size):
        expparam = np.array([Nbin]).astype(model.expparams_dtype)
        u.update(np.array([data[idx]]), expparam)
        ests[idx+1,:] = u.est_mean()
        stds[idx+1,:] = np.sqrt(np.diag(u.est_covariance_mtx()))

    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    u.plot_posterior_marginal(1)
    plt.title('Diffusion parameter posterior')

    plt.subplot(2,1,2)
    plt.plot(ts, ests[:,0], label='estimated')
    plt.fill_between(ts, ests[:,0]-stds[:,0], ests[:,0]+stds[:,0],
    alpha=0.2, antialiased=True)
    plt.plot(ts[1:], p, '--', label='actual')
    plt.legend()
    plt.title('Coin bias vs. time')
    plt.show()
    
As a second example, consider a 5-sided die for which the 3rd, 4th and 5th 
sides are taking a correlated gaussian random walk, and the other two sides
are constant. We can attempt to learn the six parameters of the cholesky 
factorization of the random walk covariance matrix as we track the drift 
of the die probabilities.
    
.. plot::

    import numpy as np
    from qinfer.utils import to_simplex, from_simplex, sample_multinomial
    from qinfer import (
       NDieModel, MultinomialModel, GaussianRandomWalkModel, 
       UniformDistribution, ConstrainedSumDistribution, SMCUpdater, ProductDistribution
    )

    # Put a random walk on top of a multinomial die model
    randomwalk_idxs = [2,3,4] # only these sides of the die are taking a walk
    model = GaussianRandomWalkModel(
        MultinomialModel(NDieModel(5)),
        model_transformation=(from_simplex, to_simplex),
        diagonal=False,
        random_walk_idxs = randomwalk_idxs
    )

    # Generate some data with some true covariance matrix
    true_cov = 0.1 * np.random.random(size=(3,3))
    true_cov = np.dot(true_cov, true_cov.T)
    Nmult = 40
    ps = from_simplex(np.array([[0.1,0.2,0.2,0.4,.1]] * 200))
    ps[:, randomwalk_idxs] += np.random.multivariate_normal(np.zeros(3), true_cov, size=200).cumsum(axis=0)
    ps = to_simplex(ps)
    expparam = np.array([(0,Nmult)],dtype=model.expparams_dtype)
    data = sample_multinomial(Nmult, ps.T).T

    # Analyse the data
    prior = ProductDistribution(
        ConstrainedSumDistribution(UniformDistribution([[0,1]] * 5)),
        UniformDistribution([[0,0.2]] * 6)
    )
    u = SMCUpdater(model, 10000, prior)
    ests, stds = np.empty((data.shape[0]+1, model.n_modelparams)), np.empty((data.shape[0]+1, model.n_modelparams))
    ts = np.arange(ests.shape[0])
    ests[0,:] = u.est_mean()
    for idx in range(data.shape[0]):
        expparam = np.array([(0,Nmult)],dtype=model.expparams_dtype)
        outcome = np.array([(data[idx],)], dtype=model.domain(expparam)[0].dtype)
        u.update(outcome, expparam)
        ests[idx+1,:] = u.est_mean()
        stds[idx+1,:] = np.sqrt(np.diag(u.est_covariance_mtx()))
        
    true_chol = np.linalg.cholesky(true_cov)
    k = 1
    plt.figure(figsize=(10,10))
    for idx, coord in enumerate(zip(*model._srw_tri_idxs)):
        i, j = coord
        plt.subplot(3,3,i*3 + j + 1)
        u.plot_posterior_marginal(5 + idx)
        plt.axvline(true_chol[i,j],color='b')
    plt.show()

    plt.figure(figsize=(12,10))
    color=iter(plt.cm.Vega10(range(5)))
    for idx in range(5):
        c=next(color)
        plt.plot(ps[:, idx], '--', label='$p_{}$ actual'.format(idx), color=c)
        plt.plot(ests[:,idx], label='$p_{}$ estimated'.format(idx), color=c)
        plt.fill_between(range(len(ests)), ests[:,idx]-stds[:,idx], ests[:,idx]+stds[:,idx],alpha=0.2, color=c, antialiased=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
