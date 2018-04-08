# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause

import numba
import numpy as np

import tdc.distances as dist

from tdc.utils import (rejection_sample,
                               make_heap,
                               heap_push,
                               unchecked_heap_push,
                               deheap_sort,
                               smallest_flagged,
                               build_candidates)

from tdc.rp_trees import (make_euclidean_tree,
                                  make_angular_tree,
                                  flatten_tree,
                                  search_flat_tree)

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def make_initialisations(dist, dist_args):
    @numba.njit(parallel=True)
    def init_from_random(n_neighbors, data, query_points, heap, rng_state):
        for i in range(query_points.shape[0]):
            indices = rejection_sample(n_neighbors, data.shape[0],
                                       rng_state)
            for j in range(indices.shape[0]):
                if indices[j] < 0:
                    continue
                d = dist(data[indices[j]], query_points[i], *dist_args)
                heap_push(heap, i, d, indices[j], 1)
        return

    @numba.njit(parallel=True)
    def init_from_tree(tree, data, query_points, heap, rng_state):
        for i in range(query_points.shape[0]):
            indices = search_flat_tree(query_points[i], tree.hyperplanes,
                                       tree.offsets, tree.children, tree.indices,
                                       rng_state)

            for j in range(indices.shape[0]):
                if indices[j] < 0:
                    continue
                d = dist(data[indices[j]], query_points[i], *dist_args)
                heap_push(heap, i, d, indices[j], 1)

        return

    return init_from_random, init_from_tree


def initialise_search(forest, data, query_points, n_neighbors,
                      init_from_random, init_from_tree, rng_state):
    results = make_heap(query_points.shape[0], n_neighbors)
    init_from_random(n_neighbors, data, query_points, results, rng_state)
    if forest is not None:
        for tree in forest:
            init_from_tree(tree, data, query_points, results, rng_state)

    return results


def make_initialized_nnd_search(dist, dist_args):
    @numba.njit(parallel=True)
    def initialized_nnd_search(data,
                               indptr,
                               indices,
                               initialization,
                               query_points):

        for i in numba.prange(query_points.shape[0]):

            tried = set(initialization[0, i])

            while True:

                # Find smallest flagged vertex
                vertex = smallest_flagged(initialization, i)

                if vertex == -1:
                    break
                candidates = indices[indptr[vertex]:indptr[vertex + 1]]
                for j in range(candidates.shape[0]):
                    if candidates[j] == vertex or candidates[j] == -1 or \
                            candidates[j] in tried:
                        continue
                    d = dist(data[candidates[j]], query_points[i], *dist_args)
                    unchecked_heap_push(initialization, i, d, candidates[j], 1)
                    tried.add(candidates[j])

        return initialization

    return initialized_nnd_search


def make_nn_descent(dist, dist_args):
    """Create a numba accelerated version of nearest neighbor descent
    specialised for the given distance metric and metric arguments. Numba
    doesn't support higher order functions directly, but we can instead JIT
    compile the version of NN-descent for any given metric.

    Parameters
    ----------
    dist: function
        A numba JITd distance function which, given two arrays computes a
        dissimilarity between them.

    dist_args: tuple
        Any extra arguments that need to be passed to the distance function
        beyond the two arrays to be compared.

    Returns
    -------
    A numba JITd function for nearest neighbor descent computation that is
    specialised to the given metric.
    """

    @numba.njit(parallel=True)
    def nn_descent(data, n_neighbors, rng_state, max_candidates=50,
                   n_iters=10, delta=0.001, rho=0.5,
                   rp_tree_init=True, leaf_array=None, verbose=False):
        n_vertices = data.shape[0]

        current_graph = make_heap(data.shape[0], n_neighbors)
        for i in range(data.shape[0]):
            indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
            for j in range(indices.shape[0]):
                d = dist(data[i], data[indices[j]], *dist_args)
                heap_push(current_graph, i, d, indices[j], 1)
                heap_push(current_graph, indices[j], d, i, 1)

        if rp_tree_init:
            for n in range(leaf_array.shape[0]):
                tried = set([(-1, -1)])
                for i in range(leaf_array.shape[1]):
                    if leaf_array[n, i] < 0:
                        break
                    for j in range(i + 1, leaf_array.shape[1]):
                        if leaf_array[n, j] < 0:
                            break
                        if (leaf_array[n, i], leaf_array[n, j]) in tried:
                            continue
                        d = dist(data[leaf_array[n, i]], data[leaf_array[n, j]],
                                 *dist_args)
                        heap_push(current_graph, leaf_array[n, i], d,
                                  leaf_array[n, j],
                                  1)
                        heap_push(current_graph, leaf_array[n, j], d,
                                  leaf_array[n, i],
                                  1)
                        tried.add((leaf_array[n, i], leaf_array[n, j]))

        for n in range(n_iters):

            (new_candidate_neighbors,
             old_candidate_neighbors) = build_candidates(current_graph,
                                                         n_vertices,
                                                         n_neighbors,
                                                         max_candidates,
                                                         rng_state, rho)

            c = 0
            for i in range(n_vertices):
                for j in range(max_candidates):
                    p = int(new_candidate_neighbors[0, i, j])
                    if p < 0:
                        continue
                    for k in range(j, max_candidates):
                        q = int(new_candidate_neighbors[0, i, k])
                        if q < 0:
                            continue

                        d = dist(data[p], data[q], *dist_args)
                        c += heap_push(current_graph, p, d, q, 1)
                        c += heap_push(current_graph, q, d, p, 1)

                    for k in range(max_candidates):
                        q = int(old_candidate_neighbors[0, i, k])
                        if q < 0:
                            continue

                        d = dist(data[p], data[q], *dist_args)
                        c += heap_push(current_graph, p, d, q, 1)
                        c += heap_push(current_graph, q, d, p, 1)


            if c <= delta * n_neighbors * data.shape[0]:
                break

        return deheap_sort(current_graph)

    return nn_descent

def make_heap_initializer(dist, dist_args):
    """Create a numba accelerated version of heap initialization for the
    alternative k-neighbor graph algorithm. This approach builds two heaps
    of neighbors simultaneously, one is a heap used to construct a very
    approximate k-neighbor graph for searching; the other is the
    initialization for searching.

    Parameters
    ----------
    dist: function
        A numba JITd distance function which, given two arrays computes a
        dissimilarity between them.

    dist_args: tuple
        Any extra arguments that need to be passed to the distance function
        beyond the two arrays to be compared.

    Returns
    -------
    A numba JITd function for for heap initialization that is
    specialised to the given metric.
    """

    @numba.njit(parallel=True)
    def initialize_heaps(data, n_neighbors, leaf_array):
        graph_heap = make_heap(data.shape[0], 10)
        search_heap = make_heap(data.shape[0], n_neighbors * 2)
        tried = set([(-1, -1)])
        for n in range(leaf_array.shape[0]):
            for i in range(leaf_array.shape[1]):
                if leaf_array[n, i] < 0:
                    break
                for j in range(i + 1, leaf_array.shape[1]):
                    if leaf_array[n, j] < 0:
                        break
                    if (leaf_array[n, i], leaf_array[n, j]) in tried:
                        continue

                    d = dist(data[leaf_array[n, i]], data[leaf_array[n, j]],
                             *dist_args)
                    unchecked_heap_push(graph_heap, leaf_array[n, i], d,
                              leaf_array[n, j], 1)
                    unchecked_heap_push(graph_heap, leaf_array[n, j], d,
                              leaf_array[n, i], 1)
                    unchecked_heap_push(search_heap, leaf_array[n, i], d,
                                        leaf_array[n, j], 1)
                    unchecked_heap_push(search_heap, leaf_array[n, j], d,
                                        leaf_array[n, i], 1)
                    tried.add((leaf_array[n, i], leaf_array[n, j]))

        return graph_heap, search_heap

    return initialize_heaps

