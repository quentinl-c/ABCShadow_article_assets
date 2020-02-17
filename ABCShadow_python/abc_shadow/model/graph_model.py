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
from abc import abstractmethod

import numpy.random as random

from .model import Model


class GraphModel(Model):
    """
    Abstraction of a graph model.
    A graph is an instance of a GraphWrapper
    (an implementation of a line graph).

    Nodes are basically labelled (coloured):
        - 0: disabled node.
        - 1: enabled node.

    """

    type_values = [0, 1]

    @classmethod
    def get_random_candidate_val(cls, p=None):
        return random.choice(list(cls.type_values), p=p)

    @abstractmethod
    def get_local_energy(self, sample, edge, neigh):
        pass

    @staticmethod
    @abstractmethod
    def get_delta_stats(mut_sample, edge, new_label):
        pass
