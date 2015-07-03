'''
Manifold landmarking with Gaussian Process
as described in the paper "Landmarking Manifolds with Gaussian Processes"
appearing in ICML 2015

CREATED: 2015-03-23 09:28:58 by Dawen Liang <dliang@ee.columbia.edu>

'''

import sys
import numpy as np
from numpy import linalg as LA


class ManifoldGP(object):
    ''' Landmarking manifolds with Gaussian processes
    and stochastic optimization '''
    def __init__(self, n_landmarks=100, batch_size=1000, n_steps=1000,
                 landmarks=None, init_lmk=None, proj=None, rescale=True,
                 random_state=None, verbose=True, **kwargs):
        ''' Gaussian processes manifold landmarking
        Arguments
        ---------
        n_landmarks : int
            Number of landmarks to learn (default 100)
        batch_size : int
            Batch size for stochastic gradient (default 1000)
        n_steps : int
            The number of gradient steps to take for each single landmark (
            default 1000)
        landmarks: ndarray or None
            Existing landmarks to begin with, should be in the shape of
            (n_existing_landmarks, n_feats), If None, landmarks will be learned
            from scratch
        init_lmk : callable or None
            A function to initialize the landmarks. If None, the default
            initialization on the unit sphere will be used
        proj : callable or None
            A projection function for ambient space that is not R^d. If None,
            the default projection to R^d will be used
        rescale : bool
            If true, the gradient is rescaled to have unit norm (default True)
            to prevent overshooting
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Stochastic gradient scheduling hyperparameters
        '''
        self.n_landmarks = n_landmarks
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.landmarks = landmarks
        self.rescale = rescale
        self.random_state = random_state
        self.verbose = verbose

        if callable(init_lmk):
            self.init_lmk = init_lmk
        else:
            self.init_lmk = _default_init

        if callable(proj):
            self.proj = proj
        else:
            self.proj = _do_nothing

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        self.t0 = float(kwargs.get('t0', 0))
        self.gamma = float(kwargs.get('gamma', 0.5))

    def learn_landmarks(self, X, kern_width=None):
        ''' Fit the model to the data in X
        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
            Training data.

        kern_width : float
            The kernel width. If None, set the kernel width to the sum of the
            per-dimensional empirical variance

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        if kern_width is None:
            self.kern_width = np.var(X, axis=0).sum()
        else:
            self.kern_width = kern_width

        for l in xrange(self.n_landmarks):
            if self.verbose:
                print("Learning landmark %i:" % l)
                sys.stdout.flush()
            lmk = self._learn_single_landmark(X)
            if self.landmarks is None:
                self.landmarks = lmk[np.newaxis, :]
            else:
                self.landmarks = np.vstack((self.landmarks, lmk))
        return self

    def _learn_single_landmark(self, X):
        ''' learn a single landmark '''
        n_samples, n_feats = X.shape
        lmk = self.init_lmk(n_feats)
        for i in xrange(1, self.n_steps + 1):
            idx = np.random.choice(n_samples, size=self.batch_size)
            lmk = self._grad_step(X[idx], lmk, i)
            if self.verbose and i % 50 == 0:
                sys.stdout.write('\rProgress: %d/%d' % (i, self.n_steps))
                sys.stdout.flush()
        if self.verbose:
            sys.stdout.write('\n')
        return lmk

    def _grad_step(self, X_batch, lmk, step):
        ''' take one stochastic gradient step '''
        phi = np.exp(-(2 - 2 * X_batch.dot(lmk)) / self.kern_width)
        if self.landmarks is None:
            M2phi = phi
        else:
            K = np.exp(-(2 - 2 * X_batch.dot(self.landmarks.T))
                       / self.kern_width)
            M2phi = phi - K.dot(LA.lstsq(K.T.dot(K), K.T.dot(phi))[0])
        rho = (self.t0 + step)**(-self.gamma)
        grad_lmk = -4. / self.kern_width * (lmk * phi.dot(M2phi) -
                                            X_batch.T.dot(M2phi * phi))
        if self.rescale:
            grad_lmk /= LA.norm(grad_lmk)
        lmk += rho * grad_lmk
        return self.proj(lmk)


# helper functions to initialize landmarks and projection #
def _do_nothing(x):
    ''' no projection, for R^d ambient space '''
    return x


def proj_pos(x):
    ''' project to the positive orthant '''
    x[x < 0] = 0
    return x


def proj_sph_pos(x):
    ''' project to the intersection of the unit sphere with positive orthant '''
    x = proj_pos(x)
    return x/LA.norm(x)


def _default_init(n_feats):
    ''' default initialization on the intersection of the unit sphere with
    positive orthant (if you know what you are doing, you should probably not
    use it) '''
    lmk = proj_sph_pos(np.ones(n_feats))
    return lmk
