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
	'''
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
	'''


	#SARTHAK GREEDILY SELECT BEST EDGE REMOVAL IN SHORTEST PATH for small and medium
	
	shortest_path = nx.algorithms.shortest_paths.weighted.dijkstra_path(G, 0, len(G)-1)
	path_length = nx.dijkstra_path_length(G, 0, len(G)-1)

	dists = []
	for i in range(len(shortest_path)-1):
		dists.append((H[shortest_path[i]][shortest_path[i+1]]['weight'], shortest_path[i], shortest_path[i+1]))	
	dists.sort()
	k_count = 0
	c_count = 0
	best_path_city = shortest_path
	best_path_road = shortest_path
	best_path = shortest_path
	while(num_k>0 and num_c >0):
		if i >= len(dists): break
		if num_k -1 < k_count: break
		if num_c -1 < c_count: break
		curr_path_len_best_road = path_length
		curr_path_len_best_city = path_length
		B = H.copy()
		C = H.copy()
		best_edge = (0,0)
		best_city = -1
		# print(best_path)
		#best path greedy
		for i in range(0, len(dists)):
			J = H.copy()
			J.remove_edge(dists[i][1], dists[i][2])
			if nx.is_connected(J) == False:
				continue
			if nx.algorithms.shortest_paths.generic.has_path(J, 0, len(G)-1):
				path_length_new = nx.dijkstra_path_length(J, 0, len(G)-1)
				if path_length_new > curr_path_len_best_road:
					B = J.copy()
					best_edge = (dists[i][1], dists[i][2])
					curr_path_len_best_road = path_length_new
					best_path_road = nx.algorithms.shortest_paths.weighted.dijkstra_path(J, 0, len(G)-1)

		#best city greedy
		# print(best_path)
		for j in best_path:
			if j == 0 or j == (len(G) -1):
				continue
			D = H.copy()
			D.remove_node(j)
			if nx.is_connected(D) == False:
				continue
			if nx.algorithms.shortest_paths.generic.has_path(D, 0, len(G)-1):
				path_length_new = nx.dijkstra_path_length(D, 0, len(G)-1)
				if path_length_new > curr_path_len_best_city:
					C = D.copy()
					best_city = j
					curr_path_len_best_city = path_length_new
					best_path_city = nx.algorithms.shortest_paths.weighted.dijkstra_path(D, 0, len(G)-1)	
					# print(best_path_city)
					

		if(best_edge ==(0,0)):
			break
		if best_city == -1:
			break
		
		if (curr_path_len_best_city - 20< curr_path_len_best_road):
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
		for i in range(len(best_path)-1):
			dists.append((H[best_path[i]][best_path[i+1]]['weight'], best_path[i], best_path[i+1]))	
		dists.sort()
		# print(best_path)
		if nx.is_connected(H) == False:
			print("oh no")
			break


		
	'''


    # SARTHAK MIN CUTS LIFE
 	# print(G.edges().data())
 	# set capacity to be 1/weight so that min cut priortizes includes shortest edges
	#print(G.nodes())
	ST_pairs = [[[(0, len(G)-1)]]]
	Graph_List = [[H]]

	for edge in G.edges().data():
		H[edge[0]][edge[1]]['capacity'] = 1.0/edge[2]["weight"]

	partition_cnt = 0
	while(num_k>0 and num_c >0):
		if(partition_cnt>=100):
			break
		# get the set of edges in the min cut
		# compare all partitions of the graph and only do the max min cut of all partitions(can be optimized by storing paritions in a list/priority queue)
		max_cut_val = -1
		#print(partition_cnt)
		cutset = None
		for i in range(0, len(ST_pairs[partition_cnt])):
			#print(s_list)
			#print(t_list)
			#print(str(s_list[partition_cnt][i]) + " " + str(t_list[partition_cnt][i])) 
			#print(ST_pairs[partition_cnt][i][0][0])
			if(ST_pairs[partition_cnt][i][0][0] == ST_pairs[partition_cnt][i][0][1]):
				continue
			cut_val, partition = nx.minimum_cut(H, ST_pairs[partition_cnt][i][0][0], ST_pairs[partition_cnt][i][0][1])
			reachable, non_reachable = partition
			if(cut_val>max_cut_val):
				cutset = set()
				for u, nbrs in ((n, H[n]) for n in reachable):
				    cutset.update((u, v, H[u][v]['capacity']) for v in nbrs if v in non_reachable)
				cutset = list(cutset)
				cutset.sort(reverse = True, key = lambda x : x[2])
				max_cut_val = cut_val

		#print("cutset is: " , cutset)
		#remove all edges(disconnects graph)
		#we don't check which nodes are disconnected yet. Can be done by checking adjacency list?
		if(cutset == None):
			break
		for i in range(0,len(cutset)-1):
			H.remove_edge(cutset[i][0], cutset[i][1])
			remove_edge_list.append((cutset[i][0], cutset[i][1]))
			num_k -= 1
			if(num_k==0):
				break
			# Check if a node is disconnected and add to list
			# could be more efficent
			for node, val in H.degree():
				if(val == 0):
					num_c -=1
					remove_city_list.append(node)
				if(num_c==0):
					break
		#H.remove_edge(cutset[len(cutset)-1][0], cutset[len(cutset)-1][1])

		# Deal with the 2 partition of graphs and new s, t
		# Say we partition a graph S---Split--Split----T
		# Then We need S_list and T_list to be S---T--S----T
		# S-----T--S---SPlit---SPLIT---T
		# S-----T--S----T------S-----T
		# We need to keep edge to not disconnect graph
		# PARTITION STEP BROKEN FIX THIS
		remaining_edge = cutset[len(cutset)-1]
		s_new = cutset[len(cutset)-1][1]
		t_new = cutset[len(cutset)-1][0]
		tempST = ST_pairs[partition_cnt].copy()
		copy1_lst = []
		copy2_lst = []
		for s_t in tempST:
			#print(s_t)
			if(s_t[0][0] in partition[0]):
				copy1_lst.append((s_t[0][0], t_new))
			if(s_t[0][1] in partition[1]):
				copy2_lst.append((s_new,s_t[0][1]))
			if(s_t[0][0] in partition[0] and s_t[0][1] in partition[0]):
				copy1_lst.append((s_t[0][0], s_t[0][1]))
			if(s_t[0][0] in partition[1] and s_t[0][1] in partition[1]):
				copy2_lst.append((s_t[0][0], s_t[0][1]))
		ST_pairs.append([copy1_lst, copy2_lst])
		#print(remaining_edge)
		#s_list.append(s_list[partition_cnt].copy())
		#s_list[partition_cnt+1].append(cutset[len(cutset)-1][1]) # 0 | 0,s | 0,s,sp 
		#t_list.append([cutset[len(cutset)-1][0]])
		#t_list[partition_cnt+1].extend(t_list[partition_cnt])
		partition_cnt+=1
# *** end Sarthak's code
'''
		
	print("remove edge list: ", remove_edge_list)
	print("remove city list: ", remove_city_list)
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
	inputs = glob.glob('inputs/small/*')
	for input_path in inputs:
		output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
		G = read_input_file(input_path)
		c, k = solve(G)
		assert is_valid_solution(G, c, k)
		distance = calculate_score(G, c, k)
		print(distance)
		write_output_file(G, c, k, output_path)
