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


if __name__ == "__main__":
    objects, relationships = read_scene_graph_from_file("/cluster/project/cvg/students/shangwu/SceneVerse/preprocess/gpt/output1new.txt")
    visualize_custom_scene_graph(objects, relationships)