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
from .graph_model import GraphModel
import numpy as np


class IsingFlatGraphModel(GraphModel):
    """
    Stores the parameters value of a three colours Ising graph model.

        Colours meaning:
            - 0: disabled node.
            - 1: intra-organisationnal node.
            - 2: inter-organisationnal node.

        Parameters:
            - theta00: 0 <-> 0
            - theta11: 1 <-> 1
            - theta22: 2 <-> 2

    Encompasses the behaviour of the model:
        - Statistic computation on the graph
        - Energy function computation

    Satisfies thes requirements for the Metropolis-Hastings
    as well as the Gibbs samplings:
        - Change statistics computation
        - Gibbs proposal computation
    """

    type_values = [0, 1, 2]

    def __init__(self, *args):
        """ Inititialises IsingModel

        Raises:
            ValueError: if the number of given arguments is different from 3.
        """

        if len(args) != 3:
            raise ValueError

        super().__init__(*args)

    @property
    def theta00(self):
        return self._params[0]

    @property
    def theta11(self):
        return self._params[1]

    @property
    def beata22(self):
        return self._params[2]

    @property
    def params(self):
        return self._params

    def set_params(self, *args):
        """ Set parameters value.
            - theta00
            - theta11
            - theta22
        Raises:
            ValueError: If the number of given arguments is different from 3.
        """

        if len(args) != 3:
            raise ValueError
        super().set_params(*args)

    def evaluate_from_stats(self, *args):
        """ Evaluates the energy function with statistics given in argument.

        Arguments:
            args {doubles} -- Stistics.

        Raises:
            ValueError: If the number of given arguments is different from 3.

        Returns:
            float -- Energy.
        """

        if len(args) != 3:
            raise ValueError

        return super().evaluate_from_stats(*args)

    def evaluate(self, sample):
        """ Evaluates the energy function with graph 'sample' given in argument.

         Arguments:
            sample {GraphWrapper} -- Input sample.

        Returns:
            float -- Energy.
        """

        interactions_count = sample.get_ising_flat_local_stat()
        return np.dot(self._params, interactions_count)

    def get_local_energy(self, sample, node):
        """ Evaluate the local energy of a node.

        Arguments:
            sample {GraphWrapper} -- Input graph.
            node {nodeId} -- Node identifier.

        Returns:
            float -- Local energy.
        """

        stats = sample.get_ising_flat_local_stat(node)

        res = np.dot(self._params, stats)

        return res

    def get_local_stats(self, sample, node):
        """ Returns the statistics locally to the node 'node'.

        Arguments:
            sample {GraphWrapper} -- Input graph.
            node {nodeId} -- Node identifier.

        Returns:
            List[int] -- Local statistics.
        """

        return sample.get_ising_flat_local_stat(node)

    """
    GIBBS
    """

    def get_gibbs_proposal(self, sample, node):
        """ Returns proposal probabilities for a new Gibbs step.

        Arguments:
            sample {GraphWrapper} -- Input graph.
            node {nodeId} -- Node identifier.

        Returns:
            ndarray[float] -- Proposal probabilities.
        """

        res = np.zeros(3)
        stats = sample.get_gibbs_flat_stats_proposals(node)

        for i, s in enumerate(stats):
            res[i] = np.exp(s.dot(self._params))

        res = res / np.sum(res)
        return res

    """
    METROPOLIS-HASTINGS
    """

    @staticmethod
    def get_delta_stats(mut_sample, node, new_label):
        """ Returns the change statics between the current state and the new one
        induced by new_label (locally to the 'node').

        Arguments:
            mut_sample {GraphWrapper} -- Graph (it will be altered).
            node {nodeId} -- Node identifier.
            new_label {int} -- New proposal.

        Returns:
             ndarray[double] -- Change statics.
        """
        return mut_sample.ising_flat_change_stats(node, new_label)

    """
    ACCESS STATS
    """

    @staticmethod
    def get_stats(sample):
        """ Returns the overall statistics of the given graph.

        Arguments:
            sample {GraphWrapper} -- Input graph.

        Returns:
            ndarray[double] -- Statics.
        """
        return sample.get_ising_flat_stats()

    @staticmethod
    def summary_dict(samples):
        """ Returns the statistics of samples given in argument shaped as a dict.
        Each key corresponds to an interaction satistic covered by the model:
            theta01: 0 <-> 0
            thata02: 1 <-> 1
            theta12: 2 <-> 2

        Arguments:
            samples {List[GraphWrapper]} -- List of samples.

        Returns:
            ditct -- Statistics.
        """

        data = dict()

        res = np.array([g.get_ising_flat_stats() for g in samples])
        data['theta01'] = res[:, 0]
        data['theta02'] = res[:, 1]
        data['theta12'] = res[:, 2]

        return data
