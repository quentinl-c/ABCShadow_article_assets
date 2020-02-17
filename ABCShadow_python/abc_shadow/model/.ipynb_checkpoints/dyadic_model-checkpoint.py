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
"""Dyadic model module
"""

from metropolis_hasting.model.edge_model import BernouilliModel
D_DYAD_PARAM = 1.0


class DyadicModel(BernouilliModel):
    """
    type_values is related to the different edge types
    - 0 : edge doesn't exist (none edge)
    - 1 : in edge
    - 2 : out edge
    - 3 : mutual edge
    """

    type_values = {0, 1, 3}

    def __init__(self, none_edge_param, edge_param, dyadic_param):
        """Initialize dyadic model

        Arguments:
            edge_param {float} -- value of edge parameter
            dyadic_param {float} -- value of dyadic parameter
        """
        BernouilliModel.__init__(self, none_edge_param, edge_param)
        self._dyadic_param = dyadic_param

    def set_params(*args):
        """Set all parameter values belonging to the model

        * args corresponds to parameter vector of the model.
        It must be ordered as follows :
        [0] non_edge_param
        [1] edge_param
        [2] dyadic_param

        Raises:
            ValueError -- if passed argument is not well sized
        """
        if len(*args) < len(self.__dict__):
            raise ValueError

        super(DyadicModel, self).set_params(args[:1])
        self.set_dyadic_param(args[-1])

    def set_dyadic_param(self, new_val):
        """Set _dyadic_param instance variable

        Arguments:
            new_val {int} -- new value
        """
        self._dyadic_param = new_val

    def get_dyad_param(self):
        """Get dyadic parameter

        Returns:
            float -- Dyadic parameter
        """

        return self._dyadic_param

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of Dyadic Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """

        edge_side = super(DyadicModel, self).evaluate(sample)
        dyad_side = self._dyadic_param * sample.get_dyadic_count()
        return edge_side + dyad_side

    def edge_dyad_delta(self, sample, edge):
        """Compute the energy delta regarding
        edge and dyad part.

        Arguments:
            sample {GraphWrapper} -- Input sample
            edge {EdgeId} -- Edge identifier

        Returns:
            float -- Delta energy regarding edge and dyad parts
        """
        ego_type = sample.get_edge_type(edge)

        res = 0
        if ego_type == 0:
            res = self._none_edge_param
        if ego_type == 1:
            res = self._edge_param
        elif ego_type == 3:
            res = self._dyadic_param

        return res

    def compute_delta(self, mut_sample, edge, new_val):
        """Given a graph sample (mut_sample), an edge on which we will
        affect the new attribute value (new_val),
        computes difference between the new energy (on the modified sample)
        and the previous one (on the initial sample).
        Instead of counting all directed and dyad edges,
        computes only the difference between x_new - x_old.

        Arguments:
            mut_sample {GraphWrapper} -- initial sample
                                         (mutable - reference passing)
                                         by side effect, one will be modified
            edge {EdgeId} -- designated edge
                                     (for which the attribute will be modified)
            new_val {int} -- new attribute value
        Returns:
            float -- Energy delta between modified sample and initial one
        """

        old_energy = self.edge_dyad_delta(mut_sample, edge)

        mut_sample.set_edge_type(edge, new_val)

        # Computes the delta between old and new energy
        new_energy = self.edge_dyad_delta(mut_sample, edge)

        delta = new_energy - old_energy
        return delta

    def evaluate_from_stats(self, none_edge_count, edge_count, dyad_count):
        eval = super(DyadicModel, self).evaluate_from_stats(none_edge_count,
                                                            edge_count)

        eval += self._dyadic_param * dyad_count
        return eval

    @classmethod
    def summary(cls, results):
        """Creates a summary of configuration values according to the current model

        Arguments:
            results {List[GraphWrapper]} -- List of simulation results

        Returns:
            dict -- summary
        """
        dataset = dict()
        dataset['None edges counts'] = [g.get_none_edge_count()
                                        for g in results]
        dataset['Edges counts'] = [g.get_edge_count() for g in results]
        dataset['Dydadics counts'] = [g.get_dyadic_count() for g in results]
        return dataset
