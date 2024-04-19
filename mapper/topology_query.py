import networkx as nx

def create_graph_from_topology(file_path):
    graph = nx.DiGraph()
    
    with open(file_path, 'r') as file:
        for line in file:
            nodes = line.split()
            parent_node = nodes[0]
            
            if parent_node not in graph:
                graph.add_node(parent_node)
            
            for i, child_node in enumerate(nodes[1:], start=1):
                if child_node == '0':
                    continue
                child_label = chr(65 + i)  # A, B, C, D...
                graph.add_node(child_node)
                
                # Dynamically determine the direction based on child's position in the input
                if i == 1:
                    direction = "up"
                elif i == 2:
                    direction = "down"
                elif i == 3:
                    direction = "left"
                elif i == 4:
                    direction = "right"
                
                graph.add_edge(parent_node, child_node, label=child_label, direction=direction)
    
    return graph

def query_graph(graph, node, direction):
    for successor, attrs in graph[node].items():
        if attrs["direction"] == direction:
            return successor
    return None

file_path = "network_map.txt"  # Updated file path
graph = create_graph_from_topology(file_path)

while True:
    query = input("Enter query (e.g., 'A left'): ").split()
    if len(query) != 2:
        print("Invalid query format. Please enter in the format 'Node Direction'.")
        continue
    
    node, direction = query
    result = query_graph(graph, node, direction)
    
    if result:
        print("Result:", result)
    else:
        print("No node found in the specified direction or node does not exist.")
