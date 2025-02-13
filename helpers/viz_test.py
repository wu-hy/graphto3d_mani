from graphviz import Digraph

def visualize_custom_scene_graph(objects, relationships, title="Custom Scene Graph", outfolder="./vis_graphs/"):
    g = Digraph(comment=title, format='png')
    
    object_nodes = set()
    for obj1, obj2, relation in relationships:
        object_nodes.add(obj1)
        object_nodes.add(obj2)
    
    for obj in object_nodes:
        g.node(obj, obj, fontname='helvetica', style='filled', color='lightblue')
    
    for obj1, obj2, relation in relationships:
        g.edge(obj1, obj2, label=relation, color='grey')
    
    output_path = outfolder + title.replace(" ", "_")
    g.render(output_path)
    print(f"Graph saved to {output_path}.png")
    
def read_scene_graph_from_file(filepath):
    objects = set()
    relationships = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                obj1, obj2, relation = parts
                objects.add(obj1)
                objects.add(obj2)
                relationships.append((obj1, obj2, relation))
    return list(objects), relationships

# objects = [
#     "shower_curtain_1", "sink_1", "curtain_1", "curtain_2", "washing_machine_1", 
#     "trash_bin_1", "toilet_1", "rack_stand_1", "water_heater_1", "soap_dispenser_1", 
#     "shelf_1", "shower_1", "floor_1", "wall"
# ]

# relationships = [
#     ("shower_curtain_1", "wall", "hanging_on"),
#     ("sink_1", "wall", "hung_on"),
#     ("curtain_1", "wall", "hanging_on"),
#     ("curtain_2", "wall", "hanging_on"),
#     ("washing_machine_1", "wall", "hanging_on"),
#     ("trash_bin_1", "wall", "hanging_on"),
#     ("toilet_1", "wall", "hung_on"),
#     ("rack_stand_1", "wall", "hung_on"),
#     ("water_heater_1", "wall", "hung_on"),
#     ("water_heater_1", "washing_machine_1", "2_o’clock_direction_near"),
#     ("washing_machine_1", "water_heater_1", "to_the_left_of"),
#     ("soap_dispenser_1", "wall", "supported_by"),
#     ("shelf_1", "wall", "placed_on"),
#     ("shower_1", "floor_1", "on")
# ]

if __name__ == "__main__":
    objects, relationships = read_scene_graph_from_file("/cluster/home/hanywu/scene_generation/graphto3d/helpers/scene_graph_test.txt")
    visualize_custom_scene_graph(objects, relationships)