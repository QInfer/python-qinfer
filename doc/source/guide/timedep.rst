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
