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
#
from libc.math cimport exp
import numpy as np
cimport numpy as np
from numpy cimport ndarray
import cython
from .model.graph_model import GraphModel
import random

DEFAULT_ITER = 100


cpdef mcmc_sampler(sample, model, iters=DEFAULT_ITER, burnin=1, by=2):
    """wrapper of cythonized function _mcmc_sampler
    
    Arguments:
        sample {GraphWrapper} -- Initial graph on which MH is applied
        model {GraphModel} -- Model difining the energy function
    
    Keyword Arguments:
        iter {int} -- Number of iterations (default: {DEFAULT_ITER})
        burnin {int} -- burn in iterations (default: {1})
        by {int} -- sampling ratio (default: {1})
    
    Returns:
        list[ndarray[int]] -- Sufficient statistics list of resulting sampled graphs
    """
    return _mcmc_sampler(sample, model, iters, burnin, by)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _mcmc_sampler(object sample, object model, int iters, int burnin, int by):
    """Executes Metropolis Hastings sampler algorith

    Arguments:
        sample {GraphWrapper} -- Initial graph on which MH is applied
        model {GraphModel} -- Model difining the energy function
        iter {int} -- Number of iterations (default: {DEFAULT_ITER})
        burnin {int} -- burn in iterations (default: {1})
        by {int} -- sampling ratio (default: {1})

    Raises:
        ValueError -- Only graphs can be sampled
                      (check if the give model is a GraphModel)

    Returns:
        list[ndarray[int]] -- Sufficient statistics list of resulting sampled graphs
    """

    if not isinstance(model, GraphModel):
        err_msg = "⛔️ mh_sampler function can only sample graph"
        raise ValueError(err_msg)

    #  ++++ resulting list of sufficient statistics ++++
    cdef list results = []

    # ++++ Definition of variables ++++ 
    cdef int old_val, new_val, i
    cdef double epsilon
    cdef np.float_t delta

    # ++++ Model parameters ++++
    cdef ndarray[double] params = np.array(model.params)

    # ++++ Sufficient statistics of the initial graph ++++
    cdef ndarray[double] stats = np.array(model.get_stats(sample))

    # ++++ Initialisation of sufficient statistics delta vector ++++
    cdef ndarray[double] delta_stats = np.zeros(len(params))
    
    # ++++ potential values an edge may take ++++
    cdef list potential_values = [0, 1]

    # ++++ All edges of the graph ++++
    cdef list edges = list(sample.get_nodes())

    # ++++ new labels randomly generated ++++
    cdef ndarray[long] new_labels


    for i in range(iters):
        new_labels = np.random.choice(potential_values, size=len(edges))
        
        if i >= burnin and i % by == 0:
            results.append(stats.copy())

        for j in range(len(edges)):
            e = edges[j]
            old_val = sample.get_node_type(e)
            new_val = new_labels[j]
            # Compute the delta
            delta_stats = model.get_delta_stats(sample, e, new_val)
            delta = np.dot(params, delta_stats)

            epsilon = np.random.uniform(0, 1)
            if epsilon >= exp(delta):
                # Rejected
                sample.set_node_label(e, old_val)
            else:
                # Accepted
                stats += delta_stats

    return results


cpdef gibbs_sampler(sample, model, iters=DEFAULT_ITER, burnin=1, by=2):
    cdef dict graph = sample.graph.copy()
    cdef dict vertex = sample.vertex.copy()
    return _gibbs_sampler(sample, model, iters, burnin, by)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _gibbs_sampler(object sample, object model, int iters, int burnin, int by):
#cdef list _gibbs_sampler(object sample, object model, int iters, int burnin, int by):
    #  ++++ resulting list of sufficient statistics ++++
    cdef list results = []
    
    # ++++ Definition of variables ++++ 
    cdef int old_val, new_val, i, j

    # ++++ Sufficient statistics of the initial graph ++++python -m cProfile -o profile sim_test.py
    cdef ndarray[double] stats = np.array(model.get_stats(sample))
    
    # ++++ potential values an edge may take ++++
    cdef list potential_values = [0, 1]

    # ++++ All edges of the graph ++++
    cdef list edges = list(sample.get_nodes())
    cdef long edges_size = len(edges)

    # ++++ Gibbs proposals array (array of probs) ++++
    cdef ndarray[double] steps = np.zeros(3)

    # ++++ Model Attributes ++++
    cdef ndarray[double] params = np.array(model.params)
    cdef ndarray[long] type_values = np.array(model.type_values)

    for i in range(iters):
        
        if i >= burnin and i % by == 0:
            results.append(stats.copy())

        for j in range(edges_size):
            e = edges[j]
            #steps = model.get_gibbs_proposal(sample, e)
            steps = get_gibbs_proposal(params, sample, e)
            new_val = np.random.choice(potential_values, p=steps)
            stats += model.get_delta_stats(sample, e, new_val)
    return results

cdef ndarray[double] get_gibbs_proposal(ndarray[double] params, object sample, tuple node):
    cdef ndarray[double] steps = np.zeros(2)
    cdef ndarray[double, ndim=2] stats =  np.zeros((2, 3))  #= sample.get_stats_from_list(node, 3, type_values)
    cdef long label
    cdef list neighbourhood = sample.graph[node]

    cdef long activation_val = sample.activation[node]
    cdef int new_ego_type = 0
    cdef int i = 0;

    cdef int idx = 0

    for n in neighbourhood:
        label = sample.vertex[n]
        if label != 0:
            idx = label + activation_val - 2
            stats[1, idx] += 1

    for i in range(2):
        s = stats[i]
        steps[i] = np.exp(s[0] * params[0] + s[1] * params[1] + s[2] * params[2])

    steps = steps / (steps[0] + steps[1])
    return steps