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


class Binomial2ParamsGraphModel(GraphModel):
    """
    Node labels:
        - 0: disabled node.
        - 1: enabled node.

    """

    def __init__(self, *args):
        """Initialize Bernouilli (Node

        Keyword Arguments:
            theta1 {float} -- value of node parameter
        """
        if len(args) != 2:
            raise ValueError

        super().__init__(*args)

    @property
    def theta0(self):
        """Get none node parameter

        Returns:
            float -- Node parameter
        """

        return self._params[0]

    @property
    def theta1(self):
        """Get node parameter

        Returns:
            float -- Node parameter
        """

        return self._params[1]

    def set_params(self, *args):
        """Set all parameter values belonging to the model

        * args corresponds to parameter vector of the model.
        It must be ordered as follows :
        [0] non_theta1
        [1] theta1

        Raises:
            ValueError -- if passed argument is not well sized
        """
        if len(args) != 2:
            raise ValueError

        return super().set_params(*args)

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """
        res = self.theta0 * sample.get_disabled_nodes_count()
        res = self.theta1 * sample.get_enabled_nodes_count()
        return res

    def get_local_energy(self, sample, node, neigh=None):
        """Compute the energy delta regarding
        node and none node part.

        Arguments:
            sample {GraphWrapper} -- Input sample
            node {NodeId} -- Node identifier

        Returns:
            float -- Delta energy regarding node and dyad parts
        """

        node_type = sample.get_node_type(node)

        if node_type == 0:
            return self.theta0
        else:
            return self.theta1

    def evaluate_from_stats(self, *args):
        """Evaluate the energy (U) from sufficient statistics passed in argument

        Raises:
            ValueError -- If given lentght of stat vector is less than 2

        Returns:
            float -- Energy U
        """

        if len(args) != 2:
            raise ValueError

        return super().evaluate_from_stats(*args)

    @staticmethod
    def get_delta_stats(mut_sample, node, new_label):
        res = np.zeros(2)
        old_label = mut_sample.get_node_type(node)
        res[old_label] -= 1
        mut_sample.set_node_type(node, new_label)
        res[new_label] += 1
        return res

    @staticmethod
    def summary_dict(results):
        """Creates a summary of configuration values according to the current model

        Arguments:
            results {List[GraphWrapper]} -- List of simulation results

        Returns:
            dict -- summary
        """

        data = dict()
        data["None-nodes-counts"] = [g.get_disabled_nodes_count()
                                        for g in results]
        data["Nodes-counts"] = [g.get_enabled_nodes_count() for g in results]
        return data
