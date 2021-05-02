import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob
from collections import Counter
import os
import math
from itertools import islice
import numpy as np
import random
import heapq
from itertools import islice


def greedy_edges(OGgraph, num_k, num_c):
    H = OGgraph.copy()
    nodes_from_edges = [0] * (len(G))
    remove_edge_list = []
    shortest_path = nx.algorithms.shortest_paths.weighted.dijkstra_path(OGgraph, 0, len(G) - 1)
    path_length = nx.dijkstra_path_length(OGgraph, 0, len(G) - 1)
    dists = []
    for i in range(len(shortest_path) - 1):
        dists.append(
            (
                H[shortest_path[i]][shortest_path[i + 1]]["weight"],
                shortest_path[i],
                shortest_path[i + 1],
            )
        )
    dists.sort()
    k_count = 0
    c_count = 0
    heuristic = 1

    T = 1000
    while num_k > 0:
        if i >= len(dists):
            break
        if num_k - 1 < k_count:
            break
        curr_path_len_best = path_length
        B = H.copy()
        best_edge = (0, 0)
        best_path = shortest_path
        for i in range(0, len(dists)):
            J = H.copy()
            J.remove_edge(dists[i][1], dists[i][2])
            if nx.is_connected(J) == False:
                # print("HELLO")
                continue
            if nx.algorithms.shortest_paths.generic.has_path(J, 0, len(G) - 1):
                path_length_new = nx.dijkstra_path_length(J, 0, len(G) - 1)
                # prob = math.exp((curr_path_len_best - path_length_new) / (-T))
                if path_length_new > curr_path_len_best:
                    B = J.copy()
                    best_edge = (dists[i][1], dists[i][2])
                    curr_path_len_best = path_length_new
                    best_path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
                        J, 0, len(G) - 1
                    )
                # elif random.random() < prob:
                #     T *= 0.98
                #     B = J.copy()
                #     best_edge = (dists[i][1], dists[i][2])
                #     curr_path_len_best = path_length_new
                #     best_path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
                #         J, 0, len(G) - 1
                #     )

        H = B.copy()
        if best_edge == (0, 0):
            # print("HELLO IM BACK")
            break
        # print(nx.is_connected(H))
        remove_edge_list.append(best_edge)
        # print(best_edge[1])
        # print(best_edge[0])
        nodes_from_edges[best_edge[0]] = 1 / heuristic
        nodes_from_edges[best_edge[1]] = 1 / heuristic
        path_length = curr_path_len_best
        k_count += 1
        heuristic += 1
        dists = []
        for i in range(len(best_path) - 1):
            dists.append(
                (H[best_path[i]][best_path[i + 1]]["weight"], best_path[i], best_path[i + 1])
            )
        dists.sort()
    return nodes_from_edges, path_length, remove_edge_list


def greedy_best_edge_removal_small_medium(OGgraph, num_k, num_c):
    # GREEDILY SELECT BEST EDGE REMOVAL IN SHORTEST PATH for small and medium
    H = G.copy()
    remove_edge_list = []
    remove_city_list = []
    shortest_path = nx.algorithms.shortest_paths.weighted.dijkstra_path(G, 0, len(G) - 1)
    path_length = nx.dijkstra_path_length(G, 0, len(G) - 1)

    dists = []
    for i in range(len(shortest_path) - 1):
        dists.append(
            (
                H[shortest_path[i]][shortest_path[i + 1]]["weight"],
                shortest_path[i],
                shortest_path[i + 1],
            )
        )
    dists.sort()
    k_count = 0
    c_count = 0
    best_path_city = shortest_path
    best_path_road = shortest_path
    best_path = shortest_path
    while num_k > 0 and num_c > 0:
        if i >= len(dists):
            break
        if num_k - 1 < k_count:
            break
        if num_c - 1 < c_count:
            break
        curr_path_len_best_road = path_length
        curr_path_len_best_city = path_length
        B = H.copy()
        C = H.copy()
        best_edge = (0, 0)
        best_city = -1
        # print(best_path)
        # best path greedy
        for i in range(0, len(dists)):
            J = H.copy()
            J.remove_edge(dists[i][1], dists[i][2])
            if nx.is_connected(J) == False:
                continue
            if nx.algorithms.shortest_paths.generic.has_path(J, 0, len(G) - 1):
                path_length_new = nx.dijkstra_path_length(J, 0, len(G) - 1)
                if path_length_new > curr_path_len_best_road:
                    B = J.copy()
                    best_edge = (dists[i][1], dists[i][2])
                    curr_path_len_best_road = path_length_new
                    best_path_road = nx.algorithms.shortest_paths.weighted.dijkstra_path(
                        J, 0, len(G) - 1
                    )

        # best city greedy
        # print(best_path)
        for j in best_path:
            if j == 0 or j == (len(G) - 1):
                continue
            D = H.copy()
            D.remove_node(j)
            if nx.is_connected(D) == False:
                continue
            if nx.algorithms.shortest_paths.generic.has_path(D, 0, len(G) - 1):
                path_length_new = nx.dijkstra_path_length(D, 0, len(G) - 1)
                if path_length_new > curr_path_len_best_city:
                    C = D.copy()
                    best_city = j
                    curr_path_len_best_city = path_length_new
                    best_path_city = nx.algorithms.shortest_paths.weighted.dijkstra_path(
                        D, 0, len(G) - 1
                    )
                    # print(best_path_city)

        if best_edge == (0, 0):
            break
        if best_city == -1:
            break

        if curr_path_len_best_city < curr_path_len_best_road:
            H = B.copy()
            remove_edge_list.append(best_edge)
            path_length = curr_path_len_best_road

            best_path = best_path_road[:]
            k_count += 1
            # print("Deleted: ", best_edge)

        else:
            H = C.copy()
            # H.remove_node(best_city)
            remove_city_list.append(best_city)
            path_length = curr_path_len_best_city
            # print("here" , best_path_city)
            best_path = best_path_city[:]
            c_count += 1
            # print("Deleted: ", best_city)
        dists = []
        for i in range(len(best_path) - 1):
            dists.append(
                (H[best_path[i]][best_path[i + 1]]["weight"], best_path[i], best_path[i + 1])
            )
        dists.sort()
        # print(best_path)
        if nx.is_connected(H) == False:
            print("oh no")
            break
    return remove_city_list, remove_edge_list


def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """
    # pass
    # A = nx.adjacency_matrix(G).todense()

    # return parameters
    remove_edge_list = []
    remove_city_list = []
    H = G.copy()

    # set k and c accordingly
    if len(G) <= 30:
        num_k = 15
        num_c = 1
    elif len(G) <= 50:
        num_k = 50
        num_c = 3
    else:
        num_k = 100
        num_c = 5

    return create_heuristic(G, num_k, num_c)

    # big bad graph(s) time aaaa
    # notes: find node that repeats the most time, remove that node, run the whole thing again


def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def simulated_annealing(G, num_k, num_c):

    remove_edge_list = []
    remove_city_list = []
    H = G.copy()
    if len(G) <= 30:
        num_k = 15
        num_c = 1
    elif len(G) <= 50:
        num_k = 50
        num_c = 3
    else:
        num_k = 100
        num_c = 5

    # heuristic T
    T = 1000

    nodes_from_edges, short_path1, remove_edge_list = greedy_edges(G, num_k, num_c)
    # to_return = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(G, 0, len(G)-1)
    # a = Counter(nodes_from_edges)
    P = G.copy()
    # print(nodes_from_edges)
    # common = [i[0] for i in a.most_common()]
    # common_node = common[0]

    # print(common_node)
    while nodes_from_edges:
        common_node = np.argmax(nodes_from_edges)
        # print(nodes_from_edges)
        # print("hey im here!")
        if nx.algorithms.shortest_paths.generic.has_path(P, 0, len(G) - 1) and nx.is_connected(P):
            shortest_path = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(
                P, 0, len(G) - 1
            )
        if (
            common_node
            and common_node != 0
            and common_node != (len(G) - 1)
            and P.has_node(common_node)
        ):
            P.remove_node(common_node)
            if nx.algorithms.shortest_paths.generic.has_path(P, 0, len(G) - 1) and nx.is_connected(
                P
            ):
                shortest_path = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(
                    P, 0, len(G) - 1
                )
                # prob = np.exp((short_path1 - shortest_path) / (-T))
                # if (shortest_path >= short_path1) or (random.random() < prob):
                if (shortest_path >= short_path1) and num_c > 0:
                    remove_city_list.append(common_node)
                    nodes_from_edges.pop(common_node)
                    num_c -= 1
                    maxK = num_k - len(remove_edge_list)
                    nodescopy, short_path2, edge2 = greedy_edges(P, num_k, num_c)
                    remove_edge_list = edge2
                    while (
                        common_node and common_node != 0 and common_node != (len(G) - 1)
                    ) and num_c > 0:
                        # T -= 50
                        # prob = np.exp((short_path1 - shortest_path) / (-T))
                        # print(prob)
                        # if (short_path2 >= short_path1) or (random.random() < prob):
                        if short_path2 >= short_path1:
                            H = P.copy()
                            remove_edge_list = edge2
                            short_path1 = short_path2
                            common_node = np.argmax(np.asarray(nodescopy))
                            # print(nodescopy)
                            if not P.has_node(common_node):
                                break
                            # a = Counter(nodescopy)
                            # common_node = [i[0] for i in a.most_common()][0]
                            if (
                                common_node
                                and common_node != 0
                                and common_node != (len(G) - 1)
                                and P.has_node(common_node)
                            ):
                                P.remove_node(common_node)
                                if not nx.is_connected(P):
                                    continue
                                if nx.algorithms.shortest_paths.generic.has_path(P, 0, len(G) - 1):
                                    remove_city_list.append(common_node)
                                    num_c -= 1
                                maxK = num_k - len(edge2)
                                nodescopy, short_path2, edge2 = greedy_edges(P, num_k, num_c)
                                remove_edge_list = edge2
        nodes_from_edges.pop(common_node)

    return remove_city_list, remove_edge_list


def create_heuristic(Ograph, num_edge, num_city):
    edges_removed = 0
    cities_removed = 0
    remove_edge_list = []
    remove_city_list = []
    batch = 10
    for i in range(0, math.ceil(num_edge / batch)):
        paths = k_shortest_paths(Ograph, 0, len(G) - 1, batch, weight="weight")
        city_her = {}
        edge_her = {}
        her_cnt = 1
        for path in paths:
            dists = []
            for i in range(len(path) - 1):
                city_her[path[i]] = city_her.get(path[i], 0) + (1 / her_cnt) / random.uniform(
                    0.75, 1
                )
                edge_her[(path[i], path[i + 1])] = edge_her.get((path[i], path[i + 1]), 0) + (
                    1 / her_cnt
                ) / random.uniform(0.75, 1)
            city_her[path[len(path) - 1]] = city_her.get(path[i], 0) + (
                1 / her_cnt
            ) / random.uniform(0.75, 1)
            her_cnt += 2.5
        city_her = dict(sorted(city_her.items(), key=lambda item: item[1], reverse=True))
        edge_her = dict(sorted(edge_her.items(), key=lambda item: item[1], reverse=True))

        J = Ograph.copy()
        # print(city_her)
        edge_iter = iter(edge_her)
        city_iter = iter(city_her)

        # combine heuristics with greedy picking
        i = 0
        while i < batch:
            if edges_removed >= num_edge:
                break
            C = J.copy()
            remove_edge = next(edge_iter, None)
            if remove_edge == None:
                break
            if not C.has_node(remove_edge[1]) or not C.has_node(remove_edge[0]):
                continue
            if not C.has_edge(remove_edge[0], remove_edge[1]):
                continue
            C.remove_edge(remove_edge[0], remove_edge[1])
            if nx.is_connected(C):
                remove_edge_list.append(remove_edge)
                J = C.copy()
                edges_removed += 1
                i += 1
            else:
                continue
        i = 0
        while i < num_city:
            if cities_removed >= num_city:
                # print(cities_removed)
                # print(num_city)
                break
            C = J.copy()
            remove_city = next(city_iter, None)
            if remove_city == 0 or remove_city == len(G) - 1:
                continue
            if remove_city == None:
                break
            # print(remove_city)
            C.remove_node(remove_city)
            if nx.is_connected(C):
                remove_city_list.append(remove_city)
                J = C.copy()
                cities_removed += 1
                i += 1
            else:
                continue
        Ograph = J

    return remove_city_list, remove_edge_list

def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def create_heuristic(Ograph, num_edge, num_city):
    edges_removed = 0
    cities_removed = 0
    remove_edge_list = []
    remove_city_list = []

    batch = 6
    used_node = False
    while(edges_removed<num_edge or cities_removed<num_city):
        paths = k_shortest_paths(Ograph, 0, len(G)-1, batch, weight='weight')
        edge_her = {}
        city_her = {}
        her_cnt = batch*batch*batch
        #her_cnt = 1
        for path in paths:
            dists = []
            for i in range(len(path)-1):
                city_her[path[i]] = city_her.get(path[i], 1) + her_cnt
                edge_her[(path[i], path[i+1])] = edge_her.get((path[i], path[i+1]), 1) + her_cnt #1/her_cnt
            if her_cnt == batch*batch*batch:
                her_cnt /= batch*.7
            her_cnt -= batch/2
            #her_cnt +=1
        edge_her = dict(sorted(edge_her.items(), key=lambda item: item[1], reverse = True))

        J = Ograph.copy()
        H = Ograph.copy()
        edge_iter = iter(edge_her)
        city_iter = iter(city_her)
        best_edge = None
        best_node = None

        i = 0
        while(i < 1):
            if(edges_removed >= num_edge):
                break
            C = J.copy()
            remove_edge = next(edge_iter, None)
            if(remove_edge == None):
                break
            if(not C.has_node(remove_edge[1]) or not C.has_node(remove_edge[0])):
                continue
            if(not C.has_edge(remove_edge[0], remove_edge[1])):
                continue
            C.remove_edge(remove_edge[0], remove_edge[1])
            if(nx.is_connected(C)):
                best_edge = remove_edge
                J = C.copy()
                i+=1
            else:
                continue
        edge_graph = J.copy()
        
        i = 0
        while(i < 1):
            if(cities_removed >= num_city):
                break
            C = H.copy()
            remove_city = next(city_iter, None)
            if(remove_city == None):
                break
            if(not C.has_node(remove_city)):
                continue
            if(remove_city == 0 or remove_city == len(G)-1):
                continue
            C.remove_node(remove_city)
            if(nx.is_connected(C)):
                best_node = remove_city
                H = C.copy()
                i+=1
            else:
                continue
        if(best_node != None):
            node_graph = H.copy()
            node_cost = nx.dijkstra_path_length(node_graph, 0, len(G)-1)
            edge_cost = nx.dijkstra_path_length(edge_graph, 0, len(G)-1)
            #try comparing heuristics instead of costs
            if(best_edge == None or node_cost>edge_cost):
                print(node_cost, " ", edge_cost)
                remove_city_list.append(best_node)
                cities_removed +=1
                temp = remove_edge_list.copy()
                for edge in temp:
                    if(edge[0] == best_node or edge[1] == best_node):
                        remove_edge_list.remove(edge)
                        edges_removed-=1
                Ograph = node_graph.copy()
            else:
                if(best_edge != None):
                    remove_edge_list.append(best_edge)
                    edges_removed +=1
                    Ograph = edge_graph.copy()
        elif(best_edge != None):
            remove_edge_list.append(best_edge)
            edges_removed +=1
            Ograph = edge_graph.copy()
        else:
            break
    return remove_city_list, remove_edge_list


def greedy_all_shortest_edges(G, num_k, num_c):
    all_paths = list(nx.all_shortest_paths(G, 0, len(G) - 1))
    edge_removed_graph = G.copy()
    edges_removed = []
    for i in range(num_k):  # can iterate over up to k edges
        longest_path = all_paths[0]  # new shortest path
        longest_path_index = 0
        optimal_edge = None
        bridging_edges = list(nx.bridges(edge_removed_graph))
        for j in range(len(all_paths[0]) - 1):  # iterate over all nodes
            u = all_paths[0][j]
            v = all_paths[0][j + 1]  # (u, v)
            path_ind = 0
            if (u, v) not in bridging_edges:  # removal won't disconnect graph
                for long_path in all_paths[1:]:  # check longer paths
                    path_ind += 1
                    if u not in long_path and v not in long_path:
                        if path_ind > longest_path_index:  # new shortest_path
                            longest_path_index = path_ind
                            longest_path = long_path
                            optimal_edge = (u, v)
                        break
        if optimal_edge is None:
            break
        cp = edge_removed_graph.copy()
        cp.remove_edge(optimal_edge[0], optimal_edge[1])
        assert nx.is_connected(cp)
        edge_removed_graph.remove_edge(optimal_edge[0], optimal_edge[1])
        edges_removed.append(optimal_edge)
        new_all_paths = []
        for p in all_paths:
            if optimal_edge[0] not in p or (
                p.index(optimal_edge[0]) + 1 != len(p) - 1
                and p[p.index(optimal_edge[0]) + 1] != optimal_edge[1]
            ):
                new_all_paths.append(p)
        all_paths = new_all_paths
    G = edge_removed_graph
    return [], edges_removed


def min_cut_life(G, num_k, num_c):
    # SARTHAK MIN CUTS LIFE
    # print(G.edges().data())
    # set capacity to be 1/weight so that min cut priortizes includes shortest edges
    # print(G.nodes())
    ST_pairs = [[[(0, len(G) - 1)]]]
    Graph_List = [[H]]

    for edge in G.edges().data():
        H[edge[0]][edge[1]]["capacity"] = 1.0 / edge[2]["weight"]

    partition_cnt = 0
    while num_k > 0 and num_c > 0:
        if partition_cnt >= 100:
            break
        # get the set of edges in the min cut
        # compare all partitions of the graph and only do the max min cut of all partitions(can be optimized by storing paritions in a list/priority queue)
        max_cut_val = -1
        # print(partition_cnt)
        cutset = None
        for i in range(0, len(ST_pairs[partition_cnt])):
            # print(s_list)
            # print(t_list)
            # print(str(s_list[partition_cnt][i]) + " " + str(t_list[partition_cnt][i]))
            # print(ST_pairs[partition_cnt][i][0][0])
            if ST_pairs[partition_cnt][i][0][0] == ST_pairs[partition_cnt][i][0][1]:
                continue
            cut_val, partition = nx.minimum_cut(
                H, ST_pairs[partition_cnt][i][0][0], ST_pairs[partition_cnt][i][0][1]
            )
            reachable, non_reachable = partition
            if cut_val > max_cut_val:
                cutset = set()
                for u, nbrs in ((n, H[n]) for n in reachable):
                    cutset.update((u, v, H[u][v]["capacity"]) for v in nbrs if v in non_reachable)
                cutset = list(cutset)
                cutset.sort(reverse=True, key=lambda x: x[2])
                max_cut_val = cut_val

        # print("cutset is: " , cutset)
        # remove all edges(disconnects graph)
        # we don't check which nodes are disconnected yet. Can be done by checking adjacency list?
        if cutset == None:
            break
        for i in range(0, len(cutset) - 1):
            H.remove_edge(cutset[i][0], cutset[i][1])
            remove_edge_list.append((cutset[i][0], cutset[i][1]))
            num_k -= 1
            if num_k == 0:
                break
            # Check if a node is disconnected and add to list
            # could be more efficent
            for node, val in H.degree():
                if val == 0:
                    num_c -= 1
                    remove_city_list.append(node)
                if num_c == 0:
                    break
        # H.remove_edge(cutset[len(cutset)-1][0], cutset[len(cutset)-1][1])

        # Deal with the 2 partition of graphs and new s, t
        # Say we partition a graph S---Split--Split----T
        # Then We need S_list and T_list to be S---T--S----T
        # S-----T--S---SPlit---SPLIT---T
        # S-----T--S----T------S-----T
        # We need to keep edge to not disconnect graph
        # PARTITION STEP BROKEN FIX THIS
        remaining_edge = cutset[len(cutset) - 1]
        s_new = cutset[len(cutset) - 1][1]
        t_new = cutset[len(cutset) - 1][0]
        tempST = ST_pairs[partition_cnt].copy()
        copy1_lst = []
        copy2_lst = []
        for s_t in tempST:
            # print(s_t)
            if s_t[0][0] in partition[0]:
                copy1_lst.append((s_t[0][0], t_new))
            if s_t[0][1] in partition[1]:
                copy2_lst.append((s_new, s_t[0][1]))
            if s_t[0][0] in partition[0] and s_t[0][1] in partition[0]:
                copy1_lst.append((s_t[0][0], s_t[0][1]))
            if s_t[0][0] in partition[1] and s_t[0][1] in partition[1]:
                copy2_lst.append((s_t[0][0], s_t[0][1]))
        ST_pairs.append([copy1_lst, copy2_lst])
        # print(remaining_edge)
        # s_list.append(s_list[partition_cnt].copy())
        # s_list[partition_cnt+1].append(cutset[len(cutset)-1][1]) # 0 | 0,s | 0,s,sp
        # t_list.append([cutset[len(cutset)-1][0]])
        # t_list[partition_cnt+1].extend(t_list[partition_cnt])
        partition_cnt += 1
        # print("remove edge list: ", remove_edge_list)
        # print("remove city list: ", remove_city_list)
        # print("new shortest path is:", nx.algorithms.shortest_paths.weighted.dijkstra_path(H, 0, len(G)-1), "with weight: ", nx.dijkstra_path_length(H, 0, len(G)-1)
    return remove_edge_list, remove_city_list


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     c, k = solve(G)
#     assert is_valid_solution(G, c, k)
#     print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
#     write_output_file(G, c, k, 'outputs/small-1.out')

# if __name__ == '__main__':
# 	inputs = glob.glob('inputs/medium/*')
# 	for input_path in inputs:
# 		output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
# 		G = read_input_file(input_path)
# 		c, k = solve(G)
# 		assert is_valid_solution(G, c, k)
# 		distance = calculate_score(G, c, k)
# 		print(distance)
# 		write_output_file(G, c, k, output_path)


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
if __name__ == "__main__":
    inputs = glob.glob("inputs/medium/*")
    inputs = sorted(inputs)
    os.rename("current_distances.txt", "old_distances.txt")
    curr = open("current_distances.txt", "w")
    # short = open("shortest_distances.txt", "w")
    delta = open("delta_distances.txt", "w")
    old = open("old_distances.txt", "r")
    delta_dist_array = []
    for input_path in inputs:
        output_path = "outputs/" + basename(normpath(input_path))[:-3] + ".out"
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        distance = calculate_score(G, c, k)
        curr.write(input_path.split("/")[2] + ": " + str(distance) + "\n")
        # short.write(input_path.split("/")[2] + ": " + str(shortest) + "\n")
        delta_dist_array.append(distance)
        write_output_file(G, c, k, output_path)
    curr.close()
    # short.close()
    short = open("shortest_distances.txt", "r")

    i = 0
    for line in short:
        old_score = float(line.split(": ")[1][:-1])
        delta_dist_array[i] = (delta_dist_array[i] - old_score) / old_score
        delta.write(line.split(": ")[0] + ": " + str(delta_dist_array[i]) + "\n")
        i += 1
    print(np.mean(delta_dist_array))
    short.close()

    """
    shortest_path = nx.algorithms.shortest_paths.weighted.dijkstra_path(G, 0, len(G)-1)
    path_length = nx.dijkstra_path_length(G, 0, len(G)-1)
    
    #remove nodes/edges along the path to make it move somewhere else
    dists = []
    for i in range(len(shortest_path)-1):
        dists.append((H[shortest_path[i]][shortest_path[i+1]]['weight'], shortest_path[i], shortest_path[i+1]))	
    dists.sort()

    k_count = 0
    c_count = 0
    for i in range(len(dists)):
        if i >= len(dists): break
        if num_k < k_count: break
        if num_c < c_count: break
        if not nx.algorithms.shortest_paths.generic.has_path(H, 0, len(G)-1):
            print ("somehow something went wrong :C")
            break
        J = H.copy()
        prob = random.random()
        if prob > 0.5:
            J.remove_edge(dists[i][1], dists[i][2])
            if nx.algorithms.shortest_paths.generic.has_path(J, 0, len(G)-1):
                path_length_new = nx.dijkstra_path_length(J, 0, len(G)-1)
                if path_length_new > path_length:
                    H = J.copy()
                    remove_edge_list.append((dists[i][1], dists[i][2]))
                    path_length = path_length_new
                    k_count += 1
        else:
            if dists[i][1] == 0 or dists[i][1] == len(dists) -1: continue
            J.remove_node(dists[i][1])
            if nx.algorithms.shortest_paths.generic.has_path(J, 0, len(G)-1):
                path_length_new = nx.dijkstra_path_length(J, 0, len(G)-1)
                if path_length_new > path_length:
                    H = J.copy()
                    remove_city_list.append(dists[i][1])
                    path_length = path_length_new
                    dists = [(length, val, key) for (length, val, key) in dists if val == dists[i][1]]
                    c_count += 1
    # print(dists)
    #---end---
    """
