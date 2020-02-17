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
import json
import numpy as np


class ABCExecutionReport(object):

    def __init__(self, delta, n, iters, model, theta_0, y, dim,
                 mh_sampler_iter):
        self.delta = delta
        self.n = n
        self.iters = iters
        self.model = model
        self.theta_0 = theta_0
        self.y = y
        self.dim = dim
        self.mh_sampler_iter = mh_sampler_iter
        self.posteriors = None


class ABCExecutionReportJSON(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, ABCExecutionReport):
            json_dict = dict()
            json_dict['delta'] = o.delta
            json_dict['n'] = o.n
            json_dict['mh_sampler_iter'] = o.mh_sampler_iter
            json_dict['prior'] = (o.theta_0.tolist()
                                  if isinstance(o.theta_0, np.ndarray)
                                  else o.theta_0)
            json_dict['y'] = (o.y.tolist()
                              if isinstance(o.y, np.ndarray)
                              else o.y)
            if o.posteriors is not None:
                json_dict['posteriors'] = [el.tolist()
                                           if isinstance(el, np.ndarray)
                                           else el
                                           for el in o.posteriors]
            return json_dict
        else:
            return super().default(o)
