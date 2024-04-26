import networkx as nx

def create_graph_from_file(filename):
    G = nx.DiGraph()
    node_dict = {}  
    
    with open(filename, 'r') as file:
        for line in file:
            nodes = line.strip().split()  
            parent = nodes[0]  
            children = nodes[1:-1]
            restricted = nodes[-1]  
            
            if parent not in node_dict:
                G.add_node(parent, restricted=restricted)
                node_dict[parent] = True

            G.nodes[parent]["restricted"] = restricted

            for i, child in enumerate(children, start=1):
               
                if child != '~': 
                    if child not in node_dict:
                        G.add_node(child, restricted = "NR")
                        node_dict[child] = True

                    if i == 1:
                        direction = "up"
                    elif i == 2:
                        direction = "down"
                    elif i == 3:
                        direction = "left"
                    elif i == 4:
                        direction = "right"     
                    G.add_edge(parent, child, direction=direction)  
    
    return G

def query_graph(graph, node, direction):
    for successor, attrs in graph[node].items():
        if attrs["direction"] == direction:
            return successor
    return None

def node_isRestricted(graph,node):
    if graph.nodes[node]["restricted"] == "R":
        return "Restricted Area"
    return "Unrestricted Area"

file_path = "network_map.txt"  # Updated file path
graph = create_graph_from_file(file_path)

while True:
    query = input("Enter query (e.g., 'A left'): ").split()
    if len(query) != 2:
        print("Invalid query format. Please enter in the format 'Node Direction'.")
        continue
    
    node, direction = query
    result = query_graph(graph, node, direction)
    # print(graph.nodes[node].get("restricted", None))        
    print(node_isRestricted(graph,node))
    if result:
        print("Result:", result)
    else:
        print("No node found in the specified direction or node does not exist.")

