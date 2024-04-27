from customtkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from PIL import Image

set_appearance_mode("dark")

bgcolor = "#22202e"
lframe_color = "#2e2b3e"
text_color = "#403d50"
purple_color = "#836ef1"
purple_hover = "#6757BC"
red_color = "#F26F6F"

def submit_values():
    name = name_entry.get()
    if not name:
        return

    up = up_entry.get() if up_entry.get() else '~'
    down = down_entry.get() if down_entry.get() else '~'
    left = left_entry.get() if left_entry.get() else '~'
    right = right_entry.get() if right_entry.get() else '~'
    restricted = 'R' if restricted_checkbox.get() else 'NR'

    forward = up_geo_entry.get() if up_geo_entry.get() else '~'
    backward = down_geo_entry.get() if down_geo_entry.get() else '~'
    leftg = left_geo_entry.get() if left_geo_entry.get() else '~'
    rightg = right_geo_entry.get() if right_geo_entry.get() else '~'
    
    with open("mapper/network_map.txt", "a") as file:
        file.write(f"{name} {up} {down} {left} {right} {restricted}\n")

    with open("mapper/geographic_map.txt", "a") as file:
        file.write(f"{name} {forward} {backward} {leftg} {rightg} {restricted}\n")
    
    name_entry.delete(0, END)
    up_entry.delete(0, END)
    down_entry.delete(0, END)
    left_entry.delete(0, END)
    right_entry.delete(0, END)
    up_geo_entry.delete(0, END)
    down_geo_entry.delete(0, END)
    left_geo_entry.delete(0, END)
    right_geo_entry.delete(0, END)

    global G  
    G = create_graph_from_file("mapper/network_map.txt")  
    visualize_graph()
    update_list_frame()

def customEntry(parent_frame, label, row):
    entry = CTkEntry(parent_frame, placeholder_text=label, placeholder_text_color=text_color, justify="center",
                     bg_color=lframe_color, fg_color=bgcolor, height=30, width=150, corner_radius=5,
                     font=("Consolas", 15, "bold"), text_color="white", border_width=0)
    entry.grid(row=row, columnspan=2, padx=10, pady=(0, 5))
    return entry

root = CTk()
root.title("Surveillance Mapper")
root.geometry("920x585")
root.config(bg=bgcolor)

frame = CTkFrame(master=root, fg_color=lframe_color, bg_color=bgcolor, corner_radius=15)
frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(30,10), pady=20)

CTkLabel(frame, text="ADD CAMERA", font=("Consolas", 20, "bold"), justify="center", 
         text_color="white").grid(row=0, columnspan=2, pady=(20,20))

name_entry = customEntry(frame, "CAMERA NAME*", row=1)

CTkLabel(frame, text="CAMERA TOPOLOGY", font=("Consolas", 15, "bold"), justify="center", 
         text_color="white").grid(row=2, columnspan=2, pady=(0,5))

up_entry = customEntry(frame, "UP", row=3)
down_entry = customEntry(frame, "DOWN", row=4)
left_entry = customEntry(frame, "LEFT", row=5)
right_entry = customEntry(frame, "RIGHT", row=6)

CTkLabel(frame, text="GEOGRAPHIC TOPOLOGY", font=("Consolas", 15, "bold"), justify="center", 
         text_color="white").grid(row=7, columnspan=2, pady=(0,5))

up_geo_entry = customEntry(frame, "FORWARD", row=8)
down_geo_entry = customEntry(frame, "BACKWARD", row=9)
left_geo_entry = customEntry(frame, "LEFT", row=10)
right_geo_entry = customEntry(frame, "RIGHT", row=11)

restricted_checkbox = CTkCheckBox(frame, text="Restricted", font=("Consolas", 15, "bold"),
                                  fg_color=text_color,hover_color=bgcolor, corner_radius=5)
restricted_checkbox.grid(row=12, columnspan=2, pady=(5,0))

CTkButton(frame, text="SUBMIT", font=("Consolas", 15, "bold"), 
          fg_color=purple_color, bg_color=lframe_color, hover_color=purple_hover, 
          width=150, corner_radius=5, command=submit_values).grid(row=13, columnspan=2, padx=20, pady=(15,20))

#############################

def visualize_graph():
    plt.figure(figsize=(5, 3.5))
    pos = nx.spring_layout(G)

    for node in G.nodes():
        if G.nodes[node].get('restricted') == 'R':
            G.nodes[node]['color'] = red_color
        else:
            G.nodes[node]['color'] = purple_color
    
    # Retrieve node colors from the graph object
    node_colors = [G.nodes[node].get('color', purple_color) for node in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, font_color='white', font_weight="bold", ax=plt.gca())
         
    plt.axis('off')  
    plt.tight_layout()  

    for widget in graph_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

graph_frame = CTkFrame(master=root,corner_radius=15)
graph_frame.grid(row=0, column=4, rowspan=5, sticky="nsew", padx=(10,20), pady=25)

def create_graph_from_file(filename):
    G = nx.Graph()
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

            for child in children:
                if child != '~': 
                    if child not in node_dict:
                        G.add_node(child, restricted = "NR")
                        node_dict[child] = True
                    G.add_edge(parent, child)  
    
    return G

filename = "mapper/network_map.txt"
G = create_graph_from_file(filename)

visualize_graph()

###########################

list_frame = CTkScrollableFrame(master=root, fg_color=lframe_color, bg_color=bgcolor, corner_radius=15)
list_frame.grid(row=0, column=5, rowspan=2, sticky="nsew", padx=0, pady=20)

title_label = CTkLabel(list_frame, text="NODE LIST", font=("Consolas", 20, "bold"), justify="center", text_color="white")
title_label.grid(column=0, columnspan=2, pady=(8,20))

delete_icon = CTkImage(Image.open("mapper/icon.png"), size=(25, 25))

def update_list_frame():
    for widget in list_frame.winfo_children():
        if widget != title_label:
            widget.destroy()

    for i, node in enumerate(G.nodes()):
        label = CTkLabel(list_frame, text=node, 
                         font=("Consolas", 15, "bold"), text_color="white",
                         justify="center", bg_color=lframe_color, fg_color=bgcolor,
                         height=35, width=130, corner_radius=5)
        label.grid(row=i+1, column=0, padx=5, pady=(0,5))

        delete_button = CTkButton(list_frame, text="", image=delete_icon, 
                                  width=20, fg_color=red_color,corner_radius=5,hover_color=red_color,
                                  command=lambda n=node: delete_node(n))
        delete_button.grid(row=i+1, column=1, padx=(5,0), pady=(0,5))

def delete_node(node):
    G.remove_node(node)
    
    with open("mapper/network_map.txt", "r") as file:
        lines = file.readlines()
    with open("mapper/network_map.txt", "w") as file:
        for line in lines:
            if line.split()[0] == node:  # If the node has its own entry
                continue  
            file.write(line)
    
    visualize_graph()
    update_list_frame()

update_list_frame()

root.mainloop()
