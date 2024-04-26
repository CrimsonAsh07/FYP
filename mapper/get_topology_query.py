import networkx as nx

restricted_default_value = "NR"

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

            # To update a node's restricted value if it was created in a previous line with the default value
            G.nodes[parent]["restricted"] = restricted

            for i, child in enumerate(children, start=1):
               
                if child != '~': 
                    if child not in node_dict:
                        G.add_node(child, restricted = restricted_default_value)  
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

def query_graph(node, direction, graph):
    
    for successor, attrs in graph[node].items():
        if attrs["direction"] == direction:
            return successor
    return None

  # Updated file path

def node_isRestricted(graph, node):
    if graph.nodes[node]["restricted"] == "R":
        return "Restricted Area"
    return "Unrestricted Area"