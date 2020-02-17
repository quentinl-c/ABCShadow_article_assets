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
import networkx as nx
#from collections.abc import Iterable
from .utils import relabel_inv_line_graph
from numpy cimport ndarray
import numpy as np
import cython
from collections import Counter

import itertools


DEFAULT_DIM = 10
DEFAULT_LABEL = 0


cdef class GraphWrapper(object):

    cdef dict _graph
    cdef dict _vertices_state
    cdef dict _vertices_activation

    def __init__(self, team_size=0, ext_size=0, gr=None, default_label=DEFAULT_LABEL):
        """Initialize Graph Wrapper object
        This is a wrapper over a networkx graph

        The graph model is based on line graph:
        the observed graph is transformed to its
        corresponding line graph.

        Consequently, observed nodes are nodes (in line graph)
        and observed nodes bridging two nodes are nodes (in line graph).

        Keyword Arguments:
            team_size {int} -- the size of the team

            gr {networkx.Graph} -- input line graph (default: {None})
        """
        
        if gr is None and team_size is None:
            self._graph = None
            self._vertices_state = None
            self._vertices_activation = None

        else:
            if gr is None:
                # Generate a complete graph instead
                intermed_graph = nx.complete_graph(team_size)
                nx.set_edge_attributes(intermed_graph, 1, 'activation')
                edge_attr = nx.get_edge_attributes(intermed_graph, 'activation')
                team_members = list(intermed_graph.nodes())
                ext_nodes = list()
                for ext in range(ext_size):
                    node = team_size + ext
                    ext_nodes.append(node)
                    intermed_graph.add_node(node)
                    for edge in itertools.product([node], team_members):
                        intermed_graph.add_edge(*edge)
                        edge_attr[tuple(sorted(edge))] = 2
                ext_nodes_set = set(ext_nodes)
                graph = nx.line_graph(intermed_graph)
                # Remove the neighbourhood relating to the external nodes
                edges_to_rm = filter(lambda x: set(x[0]) & set(x[1]) & ext_nodes_set, graph.edges())
                graph.remove_edges_from(list(edges_to_rm))

                self._graph = nx.to_dict_of_lists(graph)
                self._vertices_activation = dict(edge_attr)
                self._vertices_state = {n: 0 for n in graph.nodes()}

                assert self._graph.keys() == self._vertices_activation.keys()
                assert self._vertices_state.keys() == self._vertices_activation.keys()

            else:
                if isinstance(gr, nx.DiGraph) or isinstance(gr, nx.MultiGraph):
                    msg = "⛔️ The graph passed in argument must be a Graph"
                    raise TypeError(msg)

                graph = gr.copy()

                self._graph = nx.to_dict_of_lists(graph)
                self._vertices_state = nx.get_node_attributes(graph, 'state')
                self._vertices_activation = nx.get_node_attributes(graph, 'activation')

    def reset(self):
        self._vertices_state = self._vertices_state.fromkeys(self._vertices_state, 0)
        print(self._vertices_state)

    def copy(self):
        copy = GraphWrapper(dim=None)
        copy.graph = self.graph
        copy.vertex = self.vertex.copy()
        copy._vertices_activation = self._vertices_activation.copy()
        return copy

    @property
    def activation(self):
        return self._vertices_activation

    @property
    def vertex(self):
        return self._vertices_state

    @vertex.setter
    def vertex(self, new_ver):
        self._vertices_state = new_ver

    @property
    def graph(self):
        """Returns the Networkx graph corresponding to the graph model

        Returns:
            nx.Graph -- Corresponding graph
        """

        return self._graph

    @graph.setter
    def graph(self, new_gr):
        self._graph = new_gr


    cpdef get_disabled_nodes_count(self):
        """ Returns the number of nodes labelled as none node

        Returns:
            int -- number of none 'nodes'
        """
        return len(self.get_disabled_nodes())

    cpdef get_enabled_nodes_count(self):
        """ Returns the number of nodes labelled differently than 0

        Returns:
            int -- number of directed 'nodes'
        """

        return len(self.get_enabled_nodes())

    cpdef get_nodes(self):
        """ Get de list of nodes

        Returns:
            List[NodeId] -- list of node identifiers (tuple)
        """

        return self._vertices_state.keys()

    cpdef get_node_type(self, node_id):
        """ Given an node id
        return its corresponding type

        Arguments:
            node_id {NodeId} -- node identifier

        Returns:
            int -- node type
        """
        return self._vertices_state[node_id]

    cpdef is_active_node(self, node_id):
        """ Returns True if the node referred by node_id
        is active (i.e. node_type != 0)
        False otherwise

        Arguments:
            node_id {NodeId} -- node identifier

        Returns:
            bool -- True if the node is active
                    False otherwise
        """
        return self.get_node_type(node_id) != 0

    cpdef get_enabled_nodes(self):
        """ Returns nodes labelled differently than 0
        
        Returns:
            List[NodeId] -- list of enabled nodes
        """
        return [k for k, e in self._vertices_state.items() if e != 0]

    cpdef get_disabled_nodes(self):
        """ Returns all nodes labelled 0
        
        Returns:
            List[NodeId] -- list of disabled nodes
        """
        return [k for k, e in self._vertices_state.items() if not e]

    cpdef set_node_label(self, node_id, new_val):
        """Givent an node id
        set a new value of its corresponding type

        Arguments:
            node_id {NodeId} -- node identifier
            new_val {int} -- New value
        """

        try:
            val = int(new_val)
        except ValueError:
            msg = "🤯 node Type must be an integer"
            raise TypeError(msg)
        
        # Avoid inconsistencies, type is either equal to 0 or to the activation value 
        if val != 0:
            val = self._vertices_activation[node_id]

        self._vertices_state[node_id] = val
    
    cpdef toggle(self, node_id):
        """ Toggle the node state

        Arguments:
            node_id {NodeId} -- node identfier


        """
        if  self._vertices_state[node_id] == 0:
            self._vertices_state[node_id] = self._vertices_activation[node_id]
        else:
            self._vertices_state[node_id] = 0

    cpdef get_node_neighbourhood(self, node):
        """ Returns the neighbourhood of the node.

        Arguments:
            node {NodeId} -- node identfier

        Returns:
            List[NodeId] -- Neighbours
        """

        return self._graph[node]
    
    cpdef list get_neighbourhood_type(self, node):
        return [self._vertices_state[n] for n in self._graph[node]]

    cpdef get_density(self):
        """ Returns the density of the graph.
        i.e. the ratio between enabled nodes and all existing nodes

        Returns:
            float -- density
        """
        enabled_nodes = len(self.get_enabled_nodes())
        disabled_nodes = len(self.get_disabled_nodes())

        if enabled_nodes <= 0 and disabled_nodes <= 0:
            return 0

        d = enabled_nodes / (enabled_nodes + disabled_nodes)
        return d

    """
    _________________
    Global statistics 
    -----------------
    """

    cpdef get_node_label_count(self, label):
        """ Returns the number of nodes which are 
        labelled with the some one given in argument
        
        Arguments:
            label {int} -- Node label
        
        Returns:
            int -- count of corresponding nodes
        """
        l_nodes = [k for k, e in self._vertices_state.items() if e == label]
        return len(l_nodes)

#    cpdef get_repulsion_count(self, excluded_labels=None):
#        """ Returns the 
#        
#        Keyword Arguments:
#            excluded_labels {[type]} -- [description] (default: {None})
#        
#        Returns:
#            [type] -- [description]
#        """
#        count = 0
#
#        nodes = self._vertices_state.keys()
#
#        for e in nodes:
#            count += self.get_local_repulsion_count(e,
#                                                    excluded_labels=excluded_labels)
#
#        return count / 2

    
    """
    _____________________
    Ising repulsion stats
    ---------------------
    Three different interactions : 
    * 0 <-> 1
    * 0 <-> 2
    * 1 <-> 2
    """

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef ndarray[double] get_ising_repulsion_stats(self):
        cdef ndarray[double] interactions_count = np.zeros(3)
        cdef list nodes = list(self._vertices_state.keys())
        cdef ndarray[double] local_count
        for e in nodes:
            local_count = self.get_ising_repulsion_local_stats(e)
            interactions_count = np.add(interactions_count, local_count)


        return interactions_count / 2

#    cpdef get_local_repulsion_count(self, node, list excluded_labels=None):
#
#        excluded_labels = [] if excluded_labels is None else list(
#            excluded_labels)
#
#        cdef int ego_type = self.get_node_type(node)
#
#        if ego_type in excluded_labels:
#            return 0
#
#        cdef int count = 0
#        cdef int label
#
#        for n in self.get_node_neighbourhood(node):
#            label = self._vertices_state[n]
#            if label != ego_type and label not in excluded_labels:
#                count += 1
#
#        return count

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double] get_ising_repulsion_local_stats(self, tuple node):
        cdef ndarray[double] stats = np.zeros(3)
        cdef int ego_type = self._vertices_state[node]
        cdef int label, idx

        #cdef ndarray[long] neigh_labels = np.array([self._vertices_state[n] for n in self._graph[node]])
        #neigh_labels = neigh_labels[neigh_labels != ego_type] + ego_type
        #cdef tuple uniq = np.unique(neigh_labels, return_counts=True)
        #interactions_count[uniq[0] - 1] = uniq[1]
        for n in self._graph[node]:
            label = self._vertices_state[n]
            if label != ego_type:
                idx = ego_type + label - 1
                stats[idx] += 1

        return stats
    

    # TODO : Fix this method   
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double, ndim=2] get_stats_from_list(self, tuple node):
 
        cdef  ndarray[double, ndim=2] stats = np.zeros((3, 3)) # 3 = len(labels)
#       cdef int current_ego_type = self._vertices_state[node]
        cdef long label#, neigh_label
       # cdef int idx
       # cdef ndarray[double] counts = np.zeros(3)

        for n in self._graph[node]:
            label = self._vertices_state[n]
            if label == 0:
                stats[1, 0] += 1
                stats[2, 1] += 1
            elif label == 1:
                stats[0, 0] += 1
                stats[2, 2] += 1
            else:
                stats[0, 1] += 1
                stats[1, 2] += 1
        #stats[0, 0] += counts[1]
        #stats[0, 1] += counts[2]
#
#
        #stats[1, 0] += counts[0]
        #stats[1, 2] += counts[2]
#
        #stats[2, 1] += counts[0]
        #stats[2, 2] +=  counts[1]



        #for n in self._graph[node]:
        #    neigh_label = self._vertices_state[n]
        #    for l in range(3):# 3 = len(labels)
        #        label = labels[l]
        #        if neigh_label != label:
        #            stats[l,label + neigh_label - 1] += 1
      
        return stats

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double] ising_repulsion_change_stats(self, tuple node, int new_ego_type):
        cdef ndarray[double] stats = np.zeros(3)
        cdef int current_ego_type = self._vertices_state[node]
        cdef int label, idx

        # Consitency check
        if not new_ego_type:
            new_ego_type = 0  
        else: 
            new_ego_type = self._vertices_activation[node]

        for n in self._graph[node]:
            label = self._vertices_state[n]
            if label != current_ego_type:
                stats[current_ego_type + label - 1] -= 1
            if label != new_ego_type:
                stats[new_ego_type + label - 1] += 1

        self._vertices_state[node] = new_ego_type
        return stats

    """
    =======================================
    _____________________
    Ising flat stats
    ---------------------
    Three different interactions : 
    * 0 <-> 0
    * 1 <-> 1
    * 2 <-> 2
    =======================================
    """

    """
    Metropolis-Hastings utils
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double] ising_flat_change_stats(self, tuple node, int new_ego_type):
        cdef ndarray[double] change_stats = np.zeros(3)
        cdef int current_ego_type = self._vertices_state[node]
        cdef int label, idx


        # Consitency check
        if not new_ego_type:
            new_ego_type = 0  
        else: 
            new_ego_type = self._vertices_activation[node]

        
        for n in self._graph[node]:
            label = self._vertices_state[n]
            if current_ego_type == label:
                idx = int((label + current_ego_type) / 2)
                change_stats[idx] -= 1
            if new_ego_type == label:
                idx = int((label + new_ego_type) / 2)
                change_stats[idx] += 1

        self._vertices_state[node] = new_ego_type
        print(change_stats)
        return change_stats

    """
    Gibbs utils
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double, ndim=2] get_gibbs_flat_stats_proposals(self, tuple node):
        cdef  ndarray[double, ndim=2] stats = np.zeros((3, 3))
        #         |  stats if new label = 0
        # stats = |  stats if new label = 1
        #         |  stats if new label = 2
        cdef long label

        for n in self._graph[node]:
            label = self._vertices_state[n]
            if label == 0:
                stats[1, 0] += 1
                stats[2, 1] += 1
            elif label == 1:
                stats[0, 0] += 1
                stats[2, 2] += 1
            else:
                stats[0, 1] += 1
                stats[1, 2] += 1
        return stats


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double] get_ising_flat_stats(self):
        cdef ndarray[double] stats = np.zeros(3)
        cdef ndarray[double] local_stats = np.zeros(3)
        cdef list nodes = list(self._vertices_state.keys())
        cdef int current_ego_type


        for node in nodes:
            local_stats = self.get_ising_flat_local_stats(node)
            stats = np.add(stats, local_stats)
            #assert stats % 2 == 0

        return stats / 2


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double] get_ising_flat_local_stats(self, tuple node):
        cdef ndarray[double] stats = np.zeros(3)
        cdef list nodes = list(self._vertices_state.keys())
        cdef int current_ego_type = self._vertices_state[node]
        
        for n in self._graph[node]:
            label = self._vertices_state[n]
            if label == current_ego_type:
                idx = int((label + current_ego_type) / 2)
                #print(idx)
                stats[idx] += 1

        return stats
   
    """
    =======================================
    _____________________
    Ising Next Gen stats
    ---------------------
    Three different interactions : 
    * 1 <-> 1
    * 1 <-> 2
    * 2 <-> 2
    =======================================
    """

    """
    Metropolis-Hastings utils
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double] ising_ng_change_stats(self, tuple node, int new_ego_type):
        cdef ndarray[double] change_stats = np.zeros(3)
        cdef int current_ego_type = self._vertices_state[node]
        cdef int label, idx


        # Consitency check
        if not new_ego_type:
            new_ego_type = 0  
        else: 
            new_ego_type = self._vertices_activation[node]

        if new_ego_type != current_ego_type:
            for n in self._graph[node]:
                label = self._vertices_state[n]
                if label != 0:
                    if current_ego_type !=0:
                        idx = label + current_ego_type - 2
                        change_stats[idx] -= 1
                    if new_ego_type != 0:
                        idx = label + new_ego_type - 2
                        change_stats[idx] += 1

        self._vertices_state[node] = new_ego_type

        return change_stats

    """
    Gibbs utils
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double, ndim=2] get_gibbs_ng_stats_proposals(self, tuple node):
        cdef  ndarray[double, ndim=2] stats = np.zeros((2, 3))
        cdef int activation_val = self._vertices_activation[node]
        #         |  stats if new label = 0
        # stats = |  stats if new label = 1
        #         |  stats if new label = 2
        cdef long label

        for n in self._graph[node]:
            label = self._vertices_state[n]
            if label != 0:
                idx = label + activation_val - 2
                stats[1, idx] += 1

        return stats


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double] get_ising_ng_stats(self):
        cdef ndarray[double] stats = np.zeros(3)
        cdef ndarray[double] local_stats = np.zeros(3)
        cdef list nodes = list(self._vertices_state.keys())
        cdef int current_ego_type


        for node in nodes:
            local_stats = self.get_ising_ng_local_stats(node)
            stats = np.add(stats, local_stats)
            #assert stats % 2 == 0

        return stats / 2


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double] get_ising_ng_local_stats(self, tuple node):
        cdef ndarray[double] stats = np.zeros(3)
        cdef list nodes = list(self._vertices_state.keys())
        cdef int current_ego_type = self._vertices_state[node]
        
        if current_ego_type !=0:
            for n in self._graph[node]:
                label = self._vertices_state[n]
                if label != 0:
                    idx = label + current_ego_type - 2
                    #print(idx)
                    stats[idx] += 1

        return stats
   