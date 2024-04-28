import networkx as nx
import datetime
import pywhatkit as kit

restricted_default_value = "NR"
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def node_isRestricted(graph, node):
    if graph.nodes[node].get("restricted") == "R":
        path, loc = path_finder(graph, node)
        msg = "*A visitor has entered a restricted area.*\n"
        messages = {
            "forward": "Please guide them to the next room.",
            "backward": "Please guide them to the previous room.",
            "left": "Please guide them to the room to their left.",
            "right": "Please guide them to the room to their right.",
        }

        msg += f"_Location_: {node}\n"
        msg += f"_Time_: {current_time}\n"
        msg+=messages.get(path, "Please wait for security personnel to guide you.")
        if loc:
            msg += f"  *[Room {loc}]*"

        send_whatsapp_message("+918248822161", msg)

    else:
        return "Unrestricted Area"

def path_finder(graph, node):
    neighbors = list(graph.neighbors(node))
    for neighbor in neighbors:
        if graph.nodes[neighbor].get('restricted') != 'R':
            with open("mapper/geographic_map.txt", "r") as file:
                for line in file:
                    if line.startswith(node):
                        adjacency_list = line.strip().split()[1:]  
                        for i, v in enumerate(adjacency_list):
                            if v == neighbor:
                                return ["forward","backward","left","right"][i], v
    return "No Unrestricted Areas detected" 