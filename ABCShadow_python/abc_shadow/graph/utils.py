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
from functools import reduce
import networkx as nx


def get_first_set_elmt(s):
    if len(s) > 0:
        return list(s)[0]
    else:
        raise ValueError("set {} is empty".format(s))


def relabel_inv_line_graph(gr):
    renaming_map = dict()

    for n in gr.nodes():
        renaming_map[n] = get_first_set_elmt(
            reduce(lambda x, y: set(x) & set(y), n))

    return nx.relabel_nodes(gr, renaming_map, copy=True)
