import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob


#TODO: (jas) can we import this ??
import random
import heapq


def solve(G):
	"""
	Args:
		G: networkx.Graph
	Returns:
		c: list of cities to remove
		k: list of edges to remove
	"""
	#pass
	#A = nx.adjacency_matrix(G).todense()

	#return parameters
	remove_edge_list = []
	remove_city_list = []
	H = G.copy()

	# set k and c accordingly
	if(len(G)<=30):
		num_k = 15
		num_c = 1
	elif(len(G)<=50):
		num_k = 50
		num_c = 3
	else:
		num_k = 100
		num_c = 5

	s_list = [[0]]
	t_list = [[len(G)-1]]

#Jasmine's jank Dijsktra's
	#its 2 am and this doesnt work i give up
	# dist = [float('inf') for i in range(G.number_of_nodes())]
	# # prev = [-1 for i in range(G.number_of_nodes())]
	# visited = set()
	# path = set()
	
	# dist[0] = 0
	# queue = [(0, 0)]
	# path.add(0)

	# while queue:
	# 	dist_u, u = heapq.heappop(queue)
	# 	if u in visited: continue
	# 	visited.add(u)
	# 	dist[u] = dist_u
	# 	if u == len(G)-1:
	# 		path.add(u)
	# 		print("shortestpath is : ", path)
	# 		break
	# 	for v in G[u]:
	# 		if v in visited: continue
	# 		if dist[v] > dist[u] + G[u][v]['weight']:
	# 			dist[v] = dist[u] + G[u][v]['weight']
	# 			# prev[v] = u
	# 			heapq.heappush(queue, (dist[v], v))
	# 			path.add(v)
	#I just realized we can just do aaa
	# print("shortest path actually is: " , nx.algorithms.shortest_paths.weighted.dijkstra_path(G, 0, len(G)-1), " with weight :", nx.dijkstra_path_length(G, 0, len(G)-1))
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



# # Sarthak's Code 
# 	# print(G.edges().data())
# 	# set capacity to be 1/weight so that min cut priortizes includes shortest edges
# 	for edge in G.edges().data():
# 		H[edge[0]][edge[1]]['capacity'] = 1.0/edge[2]["weight"]
	
# 	partition_cnt = 0
# 	while(num_k>0 and num_c >0):
# 		if(partition_cnt>=100):
# 			return
# 		# get the set of edges in the min cut
# 		# compare all partitions of the graph and only do the max min cut of all partitions(can be optimized by storing paritions in a list/priority queue)
# 		max_cut_val = -1
# 		#print(partition_cnt)
# 		for i in range(0, len(s_list[partition_cnt])):
# 			#print(s_list)
# 			#print(t_list)
# 			#print(str(s_list[partition_cnt][i]) + " " + str(t_list[partition_cnt][i]))
# 			if(s_list[partition_cnt][i] == t_list[partition_cnt][i]):
# 				continue
# 			cut_val, partition = nx.minimum_cut(H, s_list[partition_cnt][i], t_list[partition_cnt][i])
# 			if(cut_val>max_cut_val):
# 				reachable, non_reachable = partition
# 				cutset = set()
# 				for u, nbrs in ((n, H[n]) for n in reachable):
# 				    cutset.update((u, v, H[u][v]['capacity']) for v in nbrs if v in non_reachable)
# 				cutset = list(cutset)
# 				cutset.sort(reverse = True, key = lambda x : x[2])
# 				max_cut_val = cut_val
# 		print("cutset is: " , cutset)
# 		#remove all edges but 1(keeps graph connected)
# 		#we don't check which nodes are disconnected yet. Can be done by checking adjacency list? 
# 		for i in range(0,len(cutset)-1):
# 			H.remove_edge(cutset[i][0], cutset[i][1])
# 			remove_edge_list.append((cutset[i][0], cutset[i][1]))
# 			num_k -= 1
# 			if(num_k==0):
# 				break
# 			# Check if a node is disconnected and add to list
# 			# could be more efficent
# 			for node, val in H.degree():
# 				if(val == 0):
# 					num_c -=1
# 					remove_city_list.append(node)
# 				if(num_c==0):
# 					break

# 		# Deal with the 2 partition of graphs and new s, t
# 		# Say we partition a graph S---Split--Split----T
# 		# Then We need S_list and T_list to be S---T--S----T
# 		# We need to keep edge to not disconnect graph
# 		# PARTITION STEP BROKEN FIX THIS
# 		remaining_edge = cutset[len(cutset)-1]
# 		#print(remaining_edge)
# 		s_list.append(s_list[partition_cnt].copy())
# 		s_list[partition_cnt+1].append(cutset[len(cutset)-1][1]) # 0 | 0,s | 0,s,sp 
# 		t_list.append([cutset[len(cutset)-1][0]])
# 		t_list[partition_cnt+1].extend(t_list[partition_cnt])
# 		partition_cnt+=1
# # *** end Sarthak's code
		
	# print("remove edge list: ", remove_edge_list)
	# print("remove city list: ", remove_city_list)
	# print("new shortest path is:", nx.algorithms.shortest_paths.weighted.dijkstra_path(H, 0, len(G)-1), "with weight: ", nx.dijkstra_path_length(H, 0, len(G)-1))
	return remove_city_list, remove_edge_list

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


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
if __name__ == '__main__':
	inputs = glob.glob('inputs/large/*')
	for input_path in inputs:
		output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
		G = read_input_file(input_path)
		c, k = solve(G)
		assert is_valid_solution(G, c, k)
		distance = calculate_score(G, c, k)
		print(distance)
		write_output_file(G, c, k, output_path)
