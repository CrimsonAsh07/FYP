from customtkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

set_appearance_mode("dark")

bgcolor = "#22202e"
lframe_color = "#2e2b3e"
text_color = "#403d50"
purple_color = "#836ef1"

def submit_values():
    name = name_entry.get()
    up = up_entry.get() if up_entry.get() else '0'
    down = down_entry.get() if down_entry.get() else '0'
    left = left_entry.get() if left_entry.get() else '0'
    right = right_entry.get() if right_entry.get() else '0'
    
    with open("network_map.txt", "a") as file:
        file.write(f"{name} {up} {down} {left} {right}\n")
    
    name_entry.delete(0, END)
    up_entry.delete(0, END)
    down_entry.delete(0, END)
    left_entry.delete(0, END)
    right_entry.delete(0, END)

    global G  
    G = create_graph_from_file("network_map.txt")  
    visualize_graph()

def customEntry(parent_frame, label, row):
    entry = CTkEntry(parent_frame, placeholder_text=label, placeholder_text_color=text_color, justify="center",
                     bg_color=lframe_color, fg_color=bgcolor, height=30, width=150, corner_radius=7.5,
                     font=("Consolas", 15, "bold"), text_color="white", border_width=0)
    entry.grid(row=row, columnspan=2, padx=10, pady=(0, 5))
    return entry

root = CTk()
root.title("Surveillance Mapper")
root.geometry("680x350")
root.config(bg=bgcolor)

frame = CTkFrame(master=root, fg_color=lframe_color, bg_color=bgcolor, corner_radius=15)
frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(30,10), pady=20)

CTkLabel(frame, text="ADD CAMERA", font=("Consolas", 20, "bold"), justify="center", text_color="white").grid(row=0, columnspan=2, pady=(20,20))

name_entry = customEntry(frame, "CAMERA NAME*", row=1)
up_entry = customEntry(frame, "UP", row=2)
down_entry = customEntry(frame, "DOWN", row=3)
left_entry = customEntry(frame, "LEFT", row=4)
right_entry = customEntry(frame, "RIGHT", row=5)

CTkButton(frame, text="SUBMIT", font=("Consolas", 15, "bold"), fg_color=purple_color, bg_color=lframe_color, width=150, 
          corner_radius=7.5, command=submit_values).grid(row=6, columnspan=2, padx=20, pady=(20,20))

#############################

def visualize_graph():
    plt.figure(figsize=(5, 3.5))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=purple_color, ax=plt.gca())
    plt.axis('off')  
    plt.tight_layout()  

    for widget in graph_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=graph_frame)
    canvas.get_tk_widget().configure(background='red')
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

graph_frame = CTkFrame(master=root,corner_radius=15)
graph_frame.grid(row=0, column=4, rowspan=5, sticky="nsew", padx=20, pady=25)

def create_graph_from_file(filename):
    G = nx.Graph()
    node_dict = {}  
    
    with open(filename, 'r') as file:
        for line in file:
            nodes = line.strip().split()  
            parent = nodes[0]  
            children = nodes[1:]  
            
            if parent not in node_dict:
                G.add_node(parent)
                node_dict[parent] = True

            for child in children:
                if child != '0': 
                    if child not in node_dict:
                        G.add_node(child)
                        node_dict[child] = True
                    G.add_edge(parent, child)  
    
    return G

filename = "network_map.txt"
G = create_graph_from_file(filename)

visualize_graph()

root.mainloop()
