from qinfer.abstract_model import Model
import numpy as np

class MultiCosModel(Model):
    
    @property
    def n_modelparams(self):
        return 2
    
    @property
    def is_n_outcomes_constant(self):
        return True
    def n_outcomes(self, expparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(np.logical_and(modelparams > 0, modelparams <= 1), axis=1)
    
    @property
    def expparams_dtype(self):
        return [('ts', 'float', 2)]
       
    def likelihood(self, outcomes, modelparams, expparams):
        # We first call the superclass method, which basically
        # just makes sure that call count diagnostics are properly
        # logged.
        super(MultiCosModel, self).likelihood(outcomes, modelparams, expparams)
        
        # Next, since we have a two-outcome model, everything is defined by
        # Pr(0 | modelparams; expparams), so we find the probability of 0
        # for each model and each experiment.
        #
        # We do so by taking a product along the modelparam index (len 2,
        # indicating omega_1 or omega_2), then squaring the result.
        pr0 = np.prod(
            np.cos(
                # shape (n_models, 1, 2)
                modelparams[:, np.newaxis, :] *
                # shape (n_experiments, 2)
                expparams['ts']
            ), # <- broadcasts to shape (n_models, n_experiments, 2).
            axis=2 # <- product over the final index (len 2)
        ) ** 2 # square each element
        
        # Now we use pr0_to_likelihood_array to turn this two index array
        # above into the form expected by SMCUpdater and other consumers
        # of likelihood().
        return Model.pr0_to_likelihood_array(outcomes, pr0)

