import networkx as nx
import datetime
import pywhatkit as kit

def send_whatsapp_message(phone_number, message):
    try:
        now = datetime.datetime.now()
        kit.sendwhatmsg_instantly(phone_number, message,10, True,5)  
        print("Message sent successfully!")
        return True
    except Exception as e:
        print("Error sending Alert:", str(e))
        return False

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

def node_isRestricted(graph, node):
    if graph.nodes[node].get("restricted") == "R":
        path = path_finder(graph, node)
        msg = "You have entered a restricted area.\n"
        messages = {
            "forward": "Please move forward to the next room to exit.",
            "backward": "Please move backward to the previous room to exit.",
            "left": "Please turn left from your current location to exit.",
            "right": "Please turn right from your current location to exit.",
        }

        msg+=messages.get(path, "Please wait for security personnel to guide you.")

        send_whatsapp_message("+918248822161",msg)

    else:
        return "Unrestricted Area"

def path_finder(graph, node):
    neighbors = graph.neighbors(node)
    with open("mapper/geographic_map.txt", "r") as file:
        for line in file:
            if line.startswith(node):
                adjacency_list = line.strip().split()[1:]  
                for i, neighbor in enumerate(neighbors):
                    if graph.nodes[neighbor].get("restricted") != "R" and neighbor in adjacency_list:
                        return ["forward", "backward", "left", "right"][i]
    return "No Unrestricted Areas detected"


file_path = "mapper/network_map.txt"  # Updated file path
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

