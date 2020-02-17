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
from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):

    """
    Basic abstraction of a model.
    Stores parameters value.

    """

    def __init__(self, *params):
        self._params = list(params)

    def set_params(self, *args):
        self._params = list(args)

    def evaluate_from_stats(self, *args):
        return np.dot(self._params, args)

    @property
    def params(self):
        return self._params

    @staticmethod
    @abstractmethod
    def summary_dict(results):
        pass

    @abstractmethod
    def evaluate(self, sample):
        pass
