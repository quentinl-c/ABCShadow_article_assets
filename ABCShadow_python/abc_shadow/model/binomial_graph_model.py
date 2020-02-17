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
import numpy as np

from .graph_model import GraphModel


class BinomialGraphModel(GraphModel):
    """
     Stores the parameter value of a binomial model for graph.

    Paramerter:
        - theta: Odds of enabling a node.

    Node label:
        - 0 :enabled node
        - 1 : disabled node

    """

    type_values = [0, 1]

    def __init__(self, *args):
        """Initializes Binomial model

        Keyword Arguments:
            edge_param {float} -- value of edge parameter
        """
        if len(args) != 1:
            raise ValueError
        super().__init__(*args)

    @property
    def theta(self):
        """Get edge parameter

        Returns:
            float -- Node parameter
        """

        return self._params[0]

    def set_params(self, *args):
        """Set all parameter values belonging to the model

        * args corresponds to parameter vector of the model.
        It must be ordered as follows :
        [0] non_edge_param
        [1] edge_param

        Raises:
            ValueError -- if passed argument is not well sized
        """
        if len(args) != 1:
            raise ValueError

        super().set_params(*args)

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of Edge Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """

        return self.theta * sample.get_enabled_nodes_count()

    def get_local_energy(self, sample, edge, neigh=None):
        """Compute the energy delta regarding
        edge and none edge part.

        Arguments:
            sample {GraphWrapper} -- Input sample
            edge {NodeId} -- Node identifier

        Returns:
            float -- Delta energy regarding edge and dyad parts
        """
        edge_type = sample.get_node_type(edge)

        res = 0

        if edge_type == 1:
            res = self.theta

        return res

    def evaluate_from_stats(self, *args):
        """Evaluate the energy (U) from sufficient statistics passed in argument

        Raises:
            ValueError -- If given lentght of stat vector is less than 2

        Returns:
            float -- Energy U
        """

        if len(args) != 1:
            raise ValueError

        return super().evaluate_from_stats(*args)

    @staticmethod
    def get_delta_stats(mut_sample, edge, new_label):
        res = np.zeros(1)
        old_label = mut_sample.get_node_type(edge)
        mut_sample.set_node_label(edge, new_label)
        res[0] = old_label - new_label
        return res

    @staticmethod
    def summary_dict(results):
        """Creates a summary of configuration values according to the current model

        Arguments:
            results {List[GraphWrapper]} -- List of simulation results

        Returns:
            dict -- summary
        """

        dataset = dict()
        dataset['Nodes counts'] = [g.get_enabled_nodes_count() for g in results]
        return dataset
