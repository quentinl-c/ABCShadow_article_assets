# coding: utf-8
#
# Copyright (c) 2019 quentinl-c.
#
# This file is part of ABCShadow 
# (see https://github.com/quentinl-c/ABCShadow_article_assets).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from libc.math cimport exp, sqrt

import numpy as np
from .model.graph_model import GraphModel
from .graph.graph_wrapper import GraphWrapper
cimport numpy as np
from numpy cimport ndarray,float64_t

from .sampler import mcmc_sampler, gibbs_sampler
import copy



INF = -100
SUP = 100

COEF_MULT = 100

"""
Implementation of ABC Shadow Algorithm
"""


cpdef abc_shadow(model, theta_0, y, delta, n, size, iters,
               sampler=None, sampler_kwargs={'it': 1}, mask=None, const_norm=1.0):
    """Executes ABC posterior sampling

    Arguments:
        model {GraphModel | Model} -- Model encompassing phenomenon of nterest
        theta_0 {[type]} -- Initial parameter (must match to model's
                            parameters)
        y {List[Sample]} -- Obseved sample(s)
        delta {List[float]} -- Bounds of proposal volumes for each parameter
        n {in} -- number of iteration for the shadow chain
        size {int} -- Size of sample(s) sampled at initialisation
                       of the shadow chain
        iters {int} -- Number of posterior samples

    Keyword Arguments:
        k {int} -- multiplying factor (default: {1})
        sampler {fct} -- Sampler function respeciting the following constraint
                         on arguments
                         args: model, size, it, seed=None
                         (default: {metropolis_hasting graph sampler})
        sampler_kwargs {dict} -- sampler's parameters (default : {'it': 1})
        mask {[type]} -- Mask array to fix some parameter (by setting 1 at 
                         the parameter position)
                         (default: {None})

    Returns:
        List[ndarray] -- List of sampled parameters -> posterior distribution
    """
    cdef list posteriors = list()

    cdef ndarray[double] theta_res = theta_0

    for i in range(iters):

        # if i % 1000 == 0:
        #    msg = f"🔎  Iteration {i} / {iters} n : {n}, theta {theta_res}"
        #    print(msg)
 
        theta_res = abc_shadow_chain(model,
                                     theta_res,
                                     y,
                                     delta,
                                     n,
                                     size,
                                     sampler=sampler,
                                     sampler_kwargs=sampler_kwargs,
                                     mask=mask,
                                     const_norm=const_norm)

        posteriors.append(theta_res)

    return posteriors


cdef abc_shadow_chain(model, theta_0, y, delta, n, size, sampler_kwargs,
                     sampler=None, mask=None, const_norm=1.0):
    """Executes ABC Shdow chain algorithm

    Arguments:
        model {GraphModel | Model} -- Model encompassing phenomenon of nterest
        theta_0 {[type]} -- Initial parameter
                            (must match to model's parameters)
        y {List[Sample]} -- Obseved sample(s)
        delta {List[float]} -- Bounds of proposal volumes for each parameter
        n {in} -- number of iteration for the shadow chain
        size {int} -- Size of sample(s) sampled at initialisation
                       of the shadow chain
        k {int} -- multiplying factor        
        sampler_kwargs {dict} -- sampler's parameters

    Keyword Arguments:
       sampler {fct} -- Sampler function respeciting the following constraint
                         on arguments
                         args: model, size, it, seed=None
                         (default: {metropolis_hasting graph sampler})
       mask {[type]} -- Mask array to fix some parameter
                         (by setting 1 at the parameter position)
                         (default: {None})

    Returns:
        np.array -- Last accepted candidate parameter
    """

    model.set_params(*theta_0)

    cdef ndarray[double] theta_res = np.array(theta_0)
    cdef ndarray[double] candidate

    if sampler is not None:
        y_sim = sampler(model, *size, **sampler_kwargs) / const_norm
    else:
        y_sim = metropolis_sampler(model, *size, **sampler_kwargs) / const_norm
  
    for _ in range(n):
        candidate = get_candidate(theta_res, delta, mask)
        alpha = get_shadow_density_ratio(model, y, y_sim, theta_res, candidate)
        prob = np.random.uniform(0, 1)

        if alpha > prob:
            theta_res = candidate

    return theta_res

cpdef get_candidate(theta, delta, mask=None):
    """Returns a candidate vector theta prime
       picked from a Uniform distribution centred on theta (old)
       according to a volume bound: delta

    Arguments:
        theta {ndarray[float]} -- intial theta parameter
        delta {ndarray[float]} -- volume : delta

    Keyword Arguments:
        mask {array[bool /int{0,1}]} -- maskek array to fix theta element
                                        (default: {None})
                        example:
                         get_candidate([1,2,3], [0.001, 0.002, 0.003], [1,1,0])
                         array([1., 2. , 3.00018494])
                         The first two elements keep theta values

    Returns:
        ndarray[float] -- picked theta prime
    """

    if len(delta) != len(theta):
        err_msg = "⛔️ delta array should have the same length as theta"
        raise ValueError(err_msg)

    candidate_vector = np.array(theta, dtype=float, copy=True)

    if mask is not None:
        if len(mask) != len(theta):
            err_msg = "🤯 mask array should have the same length as theta"
            raise ValueError(err_msg)

        indices = list(set(range(len(theta))) - set(np.nonzero(mask)[0]))
    else:
        indices = range(len(theta))

    if not indices:
        return candidate_vector

    candidate_indice = np.random.choice(indices)

    d = delta[candidate_indice]
    old = candidate_vector[candidate_indice]
    candidate_value = np.random.uniform(old - d / 2, old + d/2)

    candidate_vector[candidate_indice] = candidate_value if INF < candidate_value < SUP else old

    return candidate_vector


cdef float get_shadow_density_ratio(model, y_obs, y_sim, theta, candidate):
    model.set_params(*candidate)
    # print("candidate", candidate)
    # print("theta", theta)
    # print("y_obs", y_obs)
    # print("y_sim", y_sim)
    #cdef float64_t p1 = model.evaluate_from_stats(*y_obs)
    cdef float64_t p1 = np.dot(candidate,y_obs)

    # cdef float64_t q2 = model.evaluate_from_stats(*y_sim)
    cdef float64_t q2 = np.dot(candidate,y_sim)
    model.set_params(*theta)
    #cdef float64_t p2 = model.evaluate_from_stats(*y_obs)
    cdef float64_t p2 = np.dot(theta,y_obs)
    #cdef float64_t q1 = model.evaluate_from_stats(*y_sim)
    cdef float64_t q1 = np.dot(theta,y_sim)
    cdef float ratio = (exp(p1 - p2)) * (exp(q1 - q2))
    cdef float alpha = min(1, ratio)

    return alpha


"""
===============
Samplers
> Functions used to generate samples from a given probability density function
===============
"""


cpdef normal_sampler(model, size, it):
    samples = list()

    for _ in range(it):

        sample = np.random.normal(model.mean,
                                  sqrt(model.var), size)
        samples.append(sample)

    y_sim = [np.average(stats) for stats in model.summary_dict(samples).values()]
    return np.array(y_sim)


cpdef binom_sampler(model, size, it):
    samples = list()
    theta = model.theta
    p = exp(theta) / (1 + exp(theta))
    n = model.n

    for _ in range(it):

        sample = np.random.binomial(n, p, size)
        samples.append(sample)

    y_sim = [np.average(stats) for stats in model.summary_dict(samples).values()]
    return np.array(y_sim)


"""
_____________
Graph Sampler
-------------
"""


cpdef metropolis_sampler(model, in_size, ext_size, it=100, burnin=0, by=1):
    if not isinstance(model, GraphModel):
        err_msg = "⛔️ metropolis_sampler wrapper may only be used" \
                  "to sample graph"
        raise ValueError(err_msg)

    cdef object init_sample = GraphWrapper(in_size, ext_size)
    cdef list stat_samples = mcmc_sampler(init_sample, model, it)

    cdef ndarray vec = np.mean(stat_samples, axis=0)
    return vec

cpdef gibbs_sampler_wrapper(model, in_size, ext_size, it=100, burnin=0, by=1):

    if not isinstance(model, GraphModel):
        err_msg = "⛔️ metropolis_sampler wrapper may only be used" \
                    "to sample graph"
        raise ValueError(err_msg)

    cdef object init_sample = GraphWrapper(in_size, ext_size)
    cdef list stat_samples = gibbs_sampler(init_sample, model, it)

    cdef ndarray vec = np.mean(stat_samples, axis=0)
    return vec

cpdef binom_graph_sampler(model, size, it=1):
    sample = GraphWrapper(size)

    theta0 = model.theta0
    theta1 = model.theta1

    none_edge_prob = theta0 / (theta1 + theta0)
    edge_prob = theta1 / (theta1 + theta0)
    probs = [none_edge_prob, edge_prob]

    dist = list()

    for i in range(it):
        for edge in sample.get_nodes():
            edge_attr = model.get_random_candidate_val(p=probs)

            sample.set_node_label(edge, edge_attr)

        dist.append(copy.deepcopy(sample))

    mean_stats = (np.mean([s.get_disabled_nodes_count() for s in dist]),
                  np.mean([s.get_enabled_nodes_count() for s in dist]))
    return mean_stats
