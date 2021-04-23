import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob


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

	#print(G.edges().data())
	#set capacity to be 1/weight so that min cut priortizes includes shortest edges
	for edge in G.edges().data():
		G[edge[0]][edge[1]]['capacity'] = 1.0/edge[2]["weight"]
	
	partition_cnt = 0
	while(num_k>0 and num_c >0):
		# get the set of edges in the min cut
		# compare all partitions of the graph and only do the max min cut of all partitions(can be optimized by storing paritions in a list/priority queue)
		max_cut_val = -1
		for i in range(0, len(s_list[partition_cnt])):
			cut_val, partition = nx.minimum_cut(G, s_list[partition_cnt][i], t_list[partition_cnt][i])
			if(cut_val>max_cut_val):
				reachable, non_reachable = partition
				cutset = set()
				for u, nbrs in ((n, G[n]) for n in reachable):
				    cutset.update((u, v, G[u][v]['capacity']) for v in nbrs if v in non_reachable)
				cutset = list(cutset)
				cutset.sort(reverse = True, key = lambda x : x[2])
				max_cut_val = cut_val

		#remove all edges but 1(keeps graph connected)
		#we don't check which nodes are disconnected yet. Can be done by checking adjacency list? 
		for i in range(0,len(cutset)-1):
			G.remove_edge(cutset[i][0], cutset[i][1])
			remove_edge_list.append((cutset[i][0], cutset[i][1]))
			num_k -= 1
			if(num_k==0):
				break

		# Deal with the 2 partition of graphs and new s, t
		# Say we partition a graph S---Split--Split----T
		# Then We need S_list and T_list to be S---T--S----T
		# We need to keep edge to not disconnect graph
		s_list.append(s_list[partition_cnt])
		s_list[partition_cnt+1].append(cutset[len(cutset)-1][1]) # 0 | 0,s | 0,s,sp 
		t_list.append([cutset[len(cutset)-1][0]])
		t_list[partition_cnt+1].append(t_list[partition_cnt])
		partition_cnt+=1

	print(remove_edge_list)
	return remove_city_list,remove_edge_list


    


G = read_input_file("30.in")
solve(G)


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
		write_output_file(G, c, k, output_path)
