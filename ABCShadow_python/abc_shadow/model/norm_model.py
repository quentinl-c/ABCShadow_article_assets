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
from .model import Model


class NormModel(Model):

    min_extrem = -10
    max_extrem = 10

    def __init__(self, *args):
        if len(args) != 2:
            raise ValueError
        super().__init__(*args)

    @property
    def mean(self):
        return self._params[0]

    @property
    def var(self):
        return self._params[1]

    def set_params(self, *args):

        if len(args) != 2:
            raise ValueError
        super().__init__(*args)

    def evaluate(self, sample):
        exp = (self.mean / self.var) * sample.get_sample_sum()
        exp -= (2 * self.var)**(-1) * sample.get_sample_square_sum()
        return exp

    def evaluate_from_stats(self, *args):

        if len(args) != 2:
            raise ValueError("⛔️ Given stats lenght:{}, expected: 2".format(
                len(args)))

        res = (self.mean / self.var) * args[0]
        res -= (1/(2 * self.var)) * args[1]
        return res

    @staticmethod
    def summary_dict(results):

        dataset = dict()
        dataset["sum"] = [sum(r) for r in results]
        dataset["square_sum"] = [sum(np.array(r)**2) for r in results]

        return dataset
