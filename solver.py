import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob
from collections import Counter
import os
import math
import multiprocessing as mp

import numpy as np
from itertools import islice

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
        remove_city_list1, remove_edge_list1, her1 = create_heuristic(H,num_k, num_c, 9, 100, 13)
        remove_city_list2, remove_edge_list2, her2 = create_heuristic(H,num_k, num_c, 10, 40, 3)
        remove_city_list3, remove_edge_list3, her3 = create_heuristic(H,num_k, num_c, 3, 80, 27)
        remove_city_list4, remove_edge_list4, her4 = create_heuristic(H,num_k, num_c, 6, 100, 23)
        remove_city_list5, remove_edge_list5, her5 = create_heuristic(H,num_k, num_c, 11, 420, 69)
        remove_city_list6, remove_edge_list6, her6 = create_heuristic(H,num_k, num_c, 13, 77, 9)
        rcl = [remove_city_list1, remove_city_list2, remove_city_list3, remove_city_list4, remove_city_list5, remove_city_list6]
        rel = [remove_edge_list1, remove_edge_list2, remove_edge_list3, remove_edge_list4, remove_edge_list5, remove_edge_list6]
        herl = [her1, her2, her3, her4, her5, her6]
        best = np.argmax(herl)
        return rcl[best], rel[best]
    elif len(G) <= 50:
        num_k = 50
        num_c = 3
        remove_city_list1, remove_edge_list1, her1 = create_heuristic(H,num_k, num_c, 9, 100, 13)
        remove_city_list2, remove_edge_list2, her2 = create_heuristic(H,num_k, num_c, 10, 40, 3)
        remove_city_list3, remove_edge_list3, her3 = create_heuristic(H,num_k, num_c, 3, 80, 27)
        remove_city_list4, remove_edge_list4, her4 = create_heuristic(H,num_k, num_c, 6, 100, 23)
        rcl = [remove_city_list1, remove_city_list2, remove_city_list3, remove_city_list4]
        rel = [remove_edge_list1, remove_edge_list2, remove_edge_list3, remove_edge_list4]
        herl = [her1, her2, her3, her4]
        best = np.argmax(herl)
        return rcl[best], rel[best]
    else:
        num_k = 100
        num_c = 5
        remove_city_list1, remove_edge_list1, her1 = create_heuristic(H,num_k, num_c, 9, 100, 13) 
        remove_city_list2, remove_edge_list2, her2 = create_heuristic(H,num_k, num_c, 10, 40, 3)
        rcl = [remove_city_list1, remove_city_list2]
        rel = [remove_edge_list1, remove_edge_list2]
        herl = [her1, her2]
        best = np.argmax(herl)
        return rcl[best], rel[best]

def k_shortest_paths(G, source, target, k, weight=None):
  return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def create_heuristic(Ograph, num_edge, num_city, bat, her_c, dec_num):
	edges_removed = 0
	cities_removed = 0
	remove_edge_list = []
	remove_city_list = []

	batch = bat
	used_node = False
	while(edges_removed<num_edge or cities_removed<num_city):
		paths = k_shortest_paths(Ograph, 0, len(G)-1, batch, weight='weight')
		edge_her = {}
		city_her = {}
		her_cnt = her_c
		#her_cnt = 1
		path_cnt = 0
		for path in paths:
			path_cnt+=1
			for i in range(len(path)-1):
				if(path_cnt == 1):
					city_her[path[i]] = city_her.get(path[i], 50) + her_cnt
					edge_her[(path[i], path[i+1])] = edge_her.get((path[i], path[i+1]), 50) + her_cnt
					#her_cnt -= 13
					continue
				if(path[i] in city_her):
					city_her[path[i]] = city_her.get(path[i]) + her_cnt
				if((path[i], path[i+1]) in edge_her):
					edge_her[(path[i], path[i+1])] = edge_her.get((path[i], path[i+1])) + her_cnt #1/her_cnt
			her_cnt -= dec_num
			#her_cnt +=1
		edge_her = dict(sorted(edge_her.items(), key=lambda item: item[1], reverse = True))
		city_her = dict(sorted(city_her.items(), key=lambda item: item[1], reverse = True))

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
			#node_cost = nx.dijkstra_path_length(node_graph, 0, len(G)-1)
			#edge_cost = nx.dijkstra_path_length(edge_graph, 0, len(G)-1)
			#try comparing heuristics instead of costs
			if(best_edge == None or city_her[best_node]>edge_her[best_edge]):
				#print(node_cost, " ", edge_cost)
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
	val = nx.dijkstra_path_length(Ograph, 0, len(G)-1)
	return remove_city_list, remove_edge_list, val

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
# 	inputs = glob.glob('inputs/small/*')
# 	for input_path in inputs:
# 		output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
# 		G = read_input_file(input_path)
# 		c, k = solve(G)
# 		assert is_valid_solution(G, c, k)
# 		distance = calculate_score(G, c, k)
# 		print(distance)
# 		write_output_file(G, c, k, output_path)

if __name__ == '__main__':
    size = "large"
    inputs = sorted(glob.glob('inputs/' + size + '/large-9*')) #change this
    for input_path in inputs:
        output_path = 'outputs/' + size + "/" + basename(normpath(input_path))[:-3] + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        b = calculate_score(G, c, k)
        distance = b
        print(basename(normpath(input_path))[:-3] + ": " + str(distance))
        write_output_file(G, c, k, output_path)
