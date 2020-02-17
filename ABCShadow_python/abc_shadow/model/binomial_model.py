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
from math import log, exp
from .model import Model


class BinomialModel(Model):
    def __init__(self, *args):
        if len(args) != 2:
            raise ValueError

        super().__init__(*args)

    @property
    def n(self):
        return self._params[0]

    @property
    def theta(self):
        return self._params[1]

    @theta.setter
    def theta(self, new_theta):
        self._params[1] = new_theta

    def set_params(self, *args):
        """ Set the parameter value.
            - theta

        Raises:
            ValueError: If the number of given arguments is different from 1.
        """

        args_len = len(args)
        if args_len != 1:
            raise ValueError(f"⛔️ Given args lenght:{args_len}, expected: 1")

        self.theta = args[0]

    def evaluate(self, sample):

        res = sample * self.theta
        res -= self.n * log(1 + exp(self.theta))
        return res

    def evaluate_from_stats(self, *args):
        """ Evaluates the energy function with statistic given in argument.

        Arguments:
            args {doubles} -- Stistics.

        Raises:
            ValueError: If the number of given arguments is different from 1.

        Returns:
            float -- Energy.
        """

        args_len = len(args)
        if args_len != 1:
            raise ValueError(f"⛔️ Given stats lenght:{args_len}, expected: 1")

        return self.evaluate(args[0])

    @staticmethod
    def summary_dict(results):

        dataset = dict()
        dataset["successes"] = results

        return dataset
