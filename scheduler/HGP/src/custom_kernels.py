# Authors: Jan Hendrik Metzen <janmetzen@mailbox.org>
#
# License: BSD 3 clause

""" Non-stationary kernels that can be used with sklearn's GP module. """

import numpy as np

from scipy.special import gamma, kv

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import Kernel, _approx_fprime, Hyperparameter, RBF

EPSILON = 1e-10

class HeteroscedasticKernel(Kernel):
    """Kernel which learns a heteroscedastic noise model.

    This kernel learns for a set of prototypes values from the data space
    explicit noise levels. These exemplary noise levels are then generalized to
    the entire data space by means for kernel regression.

    Parameters
    ----------
    prototypes : array-like, shape = (n_prototypes, n_X_dims)
        Prototypic samples from the data space for which noise levels are
        estimated.

    sigma_2 : float, default: 1.0
        Parameter controlling the initial noise level

    sigma_2_bounds : pair of floats >= 0, default: (0.1, 10.0)
        The lower and upper bound on sigma_2

    gamma : float, default: 1.0
        Length scale of the kernel regression on the noise level

    gamma_bounds : pair of floats >= 0, default: (1e-2, 1e2)
        The lower and upper bound on gamma
    """
    def __init__(self, prototypes, sigma_2=1.0, sigma_2_bounds=(0.1, 10.0),
                 gamma=1.0, gamma_bounds=(1e-2, 1e2)):
        assert prototypes.shape[0] == sigma_2.shape[0]
        self.prototypes = prototypes

        self.sigma_2 = np.asarray(sigma_2)
        self.sigma_2_bounds = sigma_2_bounds

        self.gamma = gamma
        self.gamma_bounds = gamma_bounds

        self.hyperparameter_sigma_2 = \
                Hyperparameter("sigma_2", "numeric", self.sigma_2_bounds,
                               self.sigma_2.shape[0])

        self.hyperparameter_gamma = \
                Hyperparameter("gamma", "numeric", self.gamma_bounds)

    @classmethod
    def construct(cls, prototypes, sigma_2=1.0, sigma_2_bounds=(0.1, 10.0),
                  gamma=1.0, gamma_bounds=(1e-2, 1e2)):
        prototypes = np.asarray(prototypes)
        if prototypes.shape[0] > 1 and len(np.atleast_1d(sigma_2)) == 1:
            sigma_2 = np.repeat(sigma_2, prototypes.shape[0])
            sigma_2_bounds = np.vstack([sigma_2_bounds] *prototypes.shape[0])
        return cls(prototypes, sigma_2, sigma_2_bounds, gamma, gamma_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        prototypes_std = self.prototypes.std(0)
        n_prototypes = self.prototypes.shape[0]
        n_gradient_dim = \
            n_prototypes + (0 if self.hyperparameter_gamma.fixed else 1)

        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K= np.eye(X.shape[0]) * self.diag(X)
            if eval_gradient:
                K_gradient = \
                    np.zeros((K.shape[0], K.shape[0], n_gradient_dim))
                K_pairwise = \
                    pairwise_kernels(self.prototypes / (prototypes_std + EPSILON),
                                     X / (prototypes_std + EPSILON),
                                     metric="rbf", gamma=self.gamma)
                for i in range(n_prototypes):
                    for j in range(K.shape[0]):
                        K_gradient[j, j, i] = \
                            self.sigma_2[i] * K_pairwise[i, j] \
                            / (K_pairwise[:, j].sum() + EPSILON)
                if not self.hyperparameter_gamma.fixed:
                    # XXX: Analytic expression for gradient?
                    def f(gamma):  # helper function
                        theta = self.theta.copy()
                        theta[-1] = gamma[0]
                        return self.clone_with_theta(theta)(X, Y)
                    K_gradient[:, :, -1] = \
                        _approx_fprime([self.theta[-1]], f, 1e-5)[:, :, 0]
                return K, K_gradient
            else:
                return K
        else:
            K = np.zeros((X.shape[0], Y.shape[0]))
            return K   # XXX: similar entries?

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        prototypes_std = self.prototypes.std(0)
        n_prototypes = self.prototypes.shape[0]

        # kernel regression of noise levels
        K_pairwise = \
            pairwise_kernels(self.prototypes / (prototypes_std + EPSILON),
                             X / (prototypes_std + EPSILON),
                             metric="rbf", gamma=self.gamma)

        return (K_pairwise * self.sigma_2[:, None]).sum(axis=0) \
                / (K_pairwise.sum(axis=0) + EPSILON)

    def __repr__(self):
        return "{0}(sigma_2=[{1}], gamma={2})".format(self.__class__.__name__,
            ", ".join(map("{0:.3g}".format, self.sigma_2)), self.gamma)