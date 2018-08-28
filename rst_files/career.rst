.. _career:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3

***************************************
Job Search IV: Modeling Career Choice
***************************************

.. index::
    single: Modeling; Career Choice

.. contents:: :depth: 2



Overview
============

Next we study a computational problem concerning career and job choices

The model is originally due to Derek Neal :cite:`Neal1999` 

This exposition draws on the presentation in :cite:`Ljungqvist2012`, section 6.5




Model features
----------------

* Career and job within career both chosen to maximize expected discounted wage flow

* Infinite horizon dynamic programming with two state variables


Model
========

In what follows we distinguish between a career and a job, where

* a *career* is understood to be a general field encompassing many possible jobs, and

* a *job*  is understood to be a position with a particular firm

For workers, wages can be decomposed into the contribution of job and career

* :math:`w_t = \theta_t + \epsilon_t`, where

    * :math:`\theta_t` is contribution of career at time :math:`t`
    * :math:`\epsilon_t` is contribution of job at time :math:`t`

At the start of time :math:`t`, a worker has the following options

* retain a current (career, job) pair :math:`(\theta_t, \epsilon_t)`
  --- referred to hereafter as "stay put"
* retain a current career :math:`\theta_t` but redraw a job :math:`\epsilon_t`
  --- referred to hereafter as "new job"
* redraw both a career :math:`\theta_t` and a job :math:`\epsilon_t`
  --- referred to hereafter as "new life"

Draws of :math:`\theta` and :math:`\epsilon` are independent of each other and
past values, with

* :math:`\theta_t \sim F`
* :math:`\epsilon_t \sim G`

Notice that the worker does not have the option to retain a job but redraw
a career --- starting a new career always requires starting a new job

A young worker aims to maximize the expected sum of discounted wages

.. math::
    :label: exw

    \mathbb{E} \sum_{t=0}^{\infty} \beta^t w_t


subject to the choice restrictions specified above

Let :math:`V(\theta, \epsilon)` denote the value function, which is the
maximum of :eq:`exw` over all feasible (career, job) policies, given the
initial state :math:`(\theta, \epsilon)`

The value function obeys

.. math::

    V(\theta, \epsilon) = \max\{I, II, III\},


where

.. math::
    :label: eyes

    \begin{aligned}
    & I = \theta + \epsilon + \beta V(\theta, \epsilon) \\
    & II = \theta + \int \epsilon' G(d \epsilon') + \beta \int V(\theta, \epsilon') G(d \epsilon') \nonumber \\
    & III = \int \theta' F(d \theta') + \int \epsilon' G(d \epsilon') + \beta \int \int V(\theta', \epsilon') G(d \epsilon') F(d \theta') \nonumber
    \end{aligned}


Evidently :math:`I`, :math:`II` and :math:`III` correspond to "stay put", "new job" and "new life", respectively

Parameterization
------------------

As in :cite:`Ljungqvist2012`, section 6.5, we will focus on a discrete version of the model, parameterized as follows:

* both :math:`\theta` and :math:`\epsilon` take values in the set ``np.linspace(0, B, N)`` --- an even grid of :math:`N` points between :math:`0` and :math:`B` inclusive
* :math:`N = 50`
* :math:`B = 5`
* :math:`\beta = 0.95`

The distributions :math:`F` and :math:`G` are discrete distributions
generating draws from the grid points ``np.linspace(0, B, N)``

A very useful family of discrete distributions is the Beta-binomial family,
with probability mass function

.. math::

    p(k \,|\, n, a, b)
    = {n \choose k} \frac{B(k + a, n - k + b)}{B(a, b)},
    \qquad k = 0, \ldots, n


Interpretation:

* draw :math:`q` from a β distribution with shape parameters :math:`(a, b)`
* run :math:`n` independent binary trials, each with success probability :math:`q`
* :math:`p(k \,|\, n, a, b)` is the probability of :math:`k` successes in these :math:`n` trials

Nice properties:

* very flexible class of distributions, including uniform, symmetric unimodal, etc.
* only three parameters

Here's a figure showing the effect of different shape parameters when :math:`n=50`



.. code-block:: python3

    from scipy.special import binom, beta
    import matplotlib.pyplot as plt
    import numpy as np


    def gen_probs(n, a, b):
        probs = np.zeros(n+1)
        for k in range(n+1):
            probs[k] = binom(n, k) * beta(k + a, n - k + b) / beta(a, b)
        return probs

    n = 50
    a_vals = [0.5, 1, 100]
    b_vals = [0.5, 1, 100]
    fig, ax = plt.subplots()
    for a, b in zip(a_vals, b_vals):
        ab_label = f'$a = {a:.1f}$, $b = {b:.1f}$'
        ax.plot(list(range(0, n+1)), gen_probs(n, a, b), '-o', label=ab_label)
    ax.legend()
    plt.show()
    



Implementation: ``career.py``
==============================================

The code for solving the DP problem described above is found in `this file <https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/career/career.py>`__, which is repeated here for convenience


.. code-block:: python3

    from quantecon.distributions import BetaBinomial


    class CareerWorkerProblem:
        """
        An instance of the class is an object with data on a particular
        problem of this type, including probabilites, discount factor and
        sample space for the variables.

        Parameters
        ----------
        β : scalar(float), optional(default=5.0)
            Discount factor
        B : scalar(float), optional(default=0.95)
            Upper bound of for both ϵ and θ
        N : scalar(int), optional(default=50)
            Number of possible realizations for both ϵ and θ
        F_a : scalar(int or float), optional(default=1)
            Parameter `a` from the career distribution
        F_b : scalar(int or float), optional(default=1)
            Parameter `b` from the career distribution
        G_a : scalar(int or float), optional(default=1)
            Parameter `a` from the job distribution
        G_b : scalar(int or float), optional(default=1)
            Parameter `b` from the job distribution

        Attributes
        ----------
        β, B, N : see Parameters
        θ : array_like(float, ndim=1)
            A grid of values from 0 to B
        ϵ : array_like(float, ndim=1)
            A grid of values from 0 to B
        F_probs : array_like(float, ndim=1)
            The probabilities of different values for F
        G_probs : array_like(float, ndim=1)
            The probabilities of different values for G
        F_mean : scalar(float)
            The mean of the distribution for F
        G_mean : scalar(float)
            The mean of the distribution for G

        """

        def __init__(self, B=5.0, β=0.95, N=50, F_a=1, F_b=1, G_a=1,
                     G_b=1):
            self.β, self.N, self.B = β, N, B
            self.θ = np.linspace(0, B, N)     # set of θ values
            self.ϵ = np.linspace(0, B, N)     # set of ϵ values
            self.F_probs = BetaBinomial(N-1, F_a, F_b).pdf()
            self.G_probs = BetaBinomial(N-1, G_a, G_b).pdf()
            self.F_mean = np.sum(self.θ * self.F_probs)
            self.G_mean = np.sum(self.ϵ * self.G_probs)

            # Store these parameters for str and repr methods
            self._F_a, self._F_b = F_a, F_b
            self._G_a, self._G_b = G_a, G_b


        def bellman_operator(self, v):
            """
            The Bellman operator for the career / job choice model of Neal.

            Parameters
            ----------
            v : array_like(float)
                A 2D NumPy array representing the value function
                Interpretation: :math:`v[i, j] = v(\theta_i, \epsilon_j)`

            Returns
            -------
            new_v : array_like(float)
                The updated value function Tv as an array of shape v.shape

            """
            new_v = np.empty(v.shape)
            for i in range(self.N):
                for j in range(self.N):
                    # stay put
                    v1 = self.θ[i] + self.ϵ[j] + self.β * v[i, j]
    
                    # new job
                    v2 = self.θ[i] + self.G_mean + self.β * v[i, :] @ self.G_probs
    
                    # new life
                    v3 = self.G_mean + self.F_mean + self.β * self.F_probs @ v @ self.G_probs
                    new_v[i, j] = max(v1, v2, v3)
            return new_v

        def get_greedy(self, v):
            """
            Compute optimal actions taking v as the value function.
    
            Parameters
            ----------
            v : array_like(float)
                A 2D NumPy array representing the value function
                Interpretation: :math:`v[i, j] = v(\theta_i, \epsilon_j)`

            Returns
            -------
            policy : array_like(float)
                A 2D NumPy array, where policy[i, j] is the optimal action
                at :math:`(\theta_i, \epsilon_j)`.

                The optimal action is represented as an integer in the set
                1, 2, 3, where 1 = 'stay put', 2 = 'new job' and 3 = 'new
                life'

            """
            policy = np.empty(v.shape, dtype=int)
            for i in range(self.N):
                for j in range(self.N):
                    v1 = self.θ[i] + self.ϵ[j] + self.β * v[i, j]
                    v2 = self.θ[i] + self.G_mean + self.β * v[i, :] @ self.G_probs
                    v3 = self.G_mean + self.F_mean + self.β * self.F_probs @ v @ self.G_probs
                    if v1 > max(v2, v3):
                        action = 1
                    elif v2 > max(v1, v3):
                        action = 2
                    else:
                        action = 3
                    policy[i, j] = action
    
            return policy
    

The code defines

* a class ``CareerWorkerProblem`` that

    * encapsulates all the details of a particular parameterization

    * implements the Bellman operator :math:`T`

In this model, :math:`T` is defined by :math:`Tv(\theta, \epsilon) = \max\{I, II, III\}`, where
:math:`I`, :math:`II` and :math:`III` are as given in :eq:`eyes`, replacing :math:`V` with :math:`v`

The default probability distributions in ``CareerWorkerProblem`` correspond to discrete uniform distributions (see :ref:`the Beta-binomial figure <beta-binom>`)

In fact all our default settings correspond to the version studied in :cite:`Ljungqvist2012`, section 6.5.

Hence we can reproduce figures 6.5.1 and 6.5.2 shown there, which exhibit the
value function and optimal policy respectively

Here's the value function



.. code-block:: python3

    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from matplotlib import cm
    import quantecon as qe

    # === set matplotlib parameters === #
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.rcParams['patch.force_edgecolor'] = True

    # === solve for the value function === #
    wp = CareerWorkerProblem()
    v_init = np.ones((wp.N, wp.N)) * 100
    v = qe.compute_fixed_point(wp.bellman_operator, v_init,
                               max_iter=200, print_skip=25)

    # === plot value function === #
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    tg, eg = np.meshgrid(wp.θ, wp.ϵ)
    ax.plot_surface(tg,
                    eg,
                    v.T,
                    rstride=2, cstride=2,
                    cmap=cm.jet,
                    alpha=0.5,
                    linewidth=0.25)
    ax.set_zlim(150, 200)
    ax.set_xlabel('θ', fontsize=14)
    ax.set_ylabel('ϵ', fontsize=14)
    ax.view_init(ax.elev, 225)
    plt.show()
    


The optimal policy can be represented as follows (see :ref:`Exercise 3 <career_ex3>` for code)

.. _career_opt_pol:

.. figure:: /_static/figures/career_solutions_ex3_py.png
   :scale: 100%


Interpretation:

* If both job and career are poor or mediocre, the worker will experiment with new job and new career

* If career is sufficiently good, the worker will hold it and experiment with new jobs until a sufficiently good one is found

* If both job and career are good, the worker will stay put


Notice that the worker will always hold on to a sufficiently good career, but not necessarily hold on to even the best paying job

The reason is that high lifetime wages require both variables to be large, and
the worker cannot change careers without changing jobs

* Sometimes a good job must be sacrificed in order to change to a better career

Exercises
=============

.. _career_ex1:

Exercise 1
------------

Using the default parameterization in the class ``CareerWorkerProblem``,
generate and plot typical sample paths for :math:`\theta` and :math:`\epsilon`
when the worker follows the optimal policy

In particular, modulo randomness, reproduce the following figure (where the horizontal axis represents time)

.. figure:: /_static/figures/career_solutions_ex1_py.png
   :scale: 100%

Hint: To generate the draws from the distributions :math:`F` and :math:`G`, use the class `DiscreteRV <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/discrete_rv.py>`_



.. _career_ex2:

Exercise 2
----------------

Let's now consider how long it takes for the worker to settle down to a
permanent job, given a starting point of :math:`(\theta, \epsilon) = (0, 0)`

In other words, we want to study the distribution of the random variable

.. math::

    T^* := \text{the first point in time from which the worker's job no longer changes}


Evidently, the worker's job becomes permanent if and only if :math:`(\theta_t, \epsilon_t)` enters the
"stay put" region of :math:`(\theta, \epsilon)` space

Letting :math:`S` denote this region, :math:`T^*` can be expressed as the
first passage time to :math:`S` under the optimal policy:

.. math::

    T^* := \inf\{t \geq 0 \,|\, (\theta_t, \epsilon_t) \in S\}


Collect 25,000 draws of this random variable and compute the median (which should be about 7)

Repeat the exercise with :math:`\beta=0.99` and interpret the change


.. _career_ex3:

Exercise 3
----------------

As best you can, reproduce :ref:`the figure showing the optimal policy <career_opt_pol>`

Hint: The ``get_greedy()`` method returns a representation of the optimal
policy where values 1, 2 and 3 correspond to "stay put", "new job" and "new life" respectively.  Use this and ``contourf`` from ``matplotlib.pyplot`` to produce the different shadings.

Now set ``G_a = G_b = 100`` and generate a new figure with these parameters.  Interpret.


Solutions
====================



.. code-block:: python3

    from quantecon import compute_fixed_point

Exercise 1
----------

Simulate job / career paths

In reading the code, recall that ``optimal_policy[i, j]`` = policy at
:math:`(\theta_i, \epsilon_j)` = either 1, 2 or 3; meaning 'stay put',
'new job' and 'new life'

.. code-block:: python3

    wp = CareerWorkerProblem()
    v_init = np.ones((wp.N, wp.N)) * 100
    v = compute_fixed_point(wp.bellman_operator, v_init, verbose=False, max_iter=200)
    optimal_policy = wp.get_greedy(v)
    F = np.cumsum(wp.F_probs)
    G = np.cumsum(wp.G_probs)
    
    def gen_path(T=20):
        i = j = 0  
        θ_index = []
        ϵ_index = []
        for t in range(T):
            if optimal_policy[i, j] == 1:    # Stay put
                pass
            elif optimal_policy[i, j] == 2:  # New job
                j = int(qe.random.draw(G))
            else:                            # New life
                i, j  = int(qe.random.draw(F)), int(qe.random.draw(G))
            θ_index.append(i)
            ϵ_index.append(j)
        return wp.θ[θ_index], wp.ϵ[ϵ_index]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax in axes:
        θ_path, ϵ_path = gen_path()
        ax.plot(ϵ_path, label='ϵ')
        ax.plot(θ_path, label='θ')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 6)
    
    plt.show()


Exercise 2
----------

The median for the original parameterization can be computed as follows

.. code-block:: python3

    wp = CareerWorkerProblem()
    v_init = np.ones((wp.N, wp.N)) * 100
    v = compute_fixed_point(wp.bellman_operator, v_init, max_iter=200, print_skip=25)
    optimal_policy = wp.get_greedy(v)
    F = np.cumsum(wp.F_probs)
    G = np.cumsum(wp.G_probs)
    
    def gen_first_passage_time():
        t = 0
        i = j = 0
        while True:
            if optimal_policy[i, j] == 1:    # Stay put
                return t
            elif optimal_policy[i, j] == 2:  # New job
                j = int(qe.random.draw(G))
            else:                            # New life
                i, j  = int(qe.random.draw(F)), int(qe.random.draw(G))
            t += 1
    
    M = 25000 # Number of samples
    samples = np.empty(M)
    for i in range(M): 
        samples[i] = gen_first_passage_time()
    print(np.median(samples))


To compute the median with :math:`\beta=0.99` instead of the default
value :math:`\beta=0.95`, replace ``wp = CareerWorkerProblem()`` with
``wp = CareerWorkerProblem(β=0.99)`` and increase the ``max_iter=200`` in ``v = compute_fixed_point(...)`` to ``max_iter=1000``

The medians are subject to randomness, but should be about 7 and 14 respectively

Not surprisingly, more patient workers will wait longer to settle down to their final job

Exercise 3
----------

Here’s the code to reproduce the original figure

.. code-block:: python3
    
    wp = CareerWorkerProblem()
    v_init = np.ones((wp.N, wp.N)) * 100
    v = compute_fixed_point(wp.bellman_operator, v_init, max_iter=200, print_skip=25)
    optimal_policy = wp.get_greedy(v)
    
    fig, ax = plt.subplots(figsize=(6,6))
    tg, eg = np.meshgrid(wp.θ, wp.ϵ)
    lvls=(0.5, 1.5, 2.5, 3.5)
    ax.contourf(tg, eg, optimal_policy.T, levels=lvls, cmap=cm.winter, alpha=0.5)
    ax.contour(tg, eg, optimal_policy.T, colors='k', levels=lvls, linewidths=2)
    ax.set_xlabel('θ', fontsize=14)
    ax.set_ylabel('ϵ', fontsize=14)
    ax.text(1.8, 2.5, 'new life', fontsize=14)
    ax.text(4.5, 2.5, 'new job', fontsize=14, rotation='vertical')
    ax.text(4.0, 4.5, 'stay put', fontsize=14)
    plt.show()


Now we want to set ``G_a = G_b = 100`` and generate a new figure with
these parameters

To do this replace: ``wp = CareerWorkerProblem()`` with ``wp = CareerWorkerProblem(G_a=100, G_b=100)``

In the new figure, you will see that the region for which the worker
will stay put has grown because the distribution for :math:`\epsilon`
has become more concentrated around the mean, making high-paying jobs
less realistic

