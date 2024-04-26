import networkx as nx

def create_graph_from_topology(filename):
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

            for i, child in enumerate(children, start=1):
               
                if child != '~': 
                    print(parent,child,i)
                    if child not in node_dict:
                        G.add_node(child)
                        node_dict[child] = True

                    if i == 1:
                        direction = "up"
                    elif i == 2:
                        direction = "down"
                    elif i == 3:
                        direction = "left"
                    elif i == 4:
                        direction = "right"     
                    print(direction)
                    G.add_edge(parent, child,direction=direction)  
    
    return G

def query_graph(node, direction, graph):
    
    for successor, attrs in graph[node].items():
        if attrs["direction"] == direction:
            return successor
    return None

  # Updated file path

