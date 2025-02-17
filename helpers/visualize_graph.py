from graphviz import Digraph
import os
from helpers import viz_util
import json
import random

def visualize_scene_graph(graph, relationships, rel_filter_in = [], rel_filter_out = [], obj_ids = [], title ="", scan_id="",
													outfolder="./vis_graphs/"):
	g = Digraph(comment='Scene Graph' + title, format='png')

	for (i,obj) in enumerate(graph["objects"]):
		if (len(obj_ids) == 0) or (int(obj['id']) in obj_ids):
			if "node_mask" in graph.keys() and graph["node_mask"][i] == 0:
				g.node(str(obj['id']), obj["label"], fontname='helvetica', color=obj["ply_color"], fontcolor='red')
			else:
				g.node(str(obj['id']), obj["label"], fontname='helvetica', color=obj["ply_color"], style='filled')
	if "edge_mask" in graph.keys():
		edge_mask = graph["edge_mask"]
	else:
		edge_mask = None
	draw_edges(g, graph["relationships"], relationships, rel_filter_in, rel_filter_out, obj_ids, edge_mask)
	g.render(outfolder + scan_id)


def draw_edges(g, graph_relationships, relationships, rel_filter_in, rel_filter_out, obj_ids, edge_mask=None):
	edges = {}
	if edge_mask is not None:
		joined_edge_mask = {}
	for (i, rel) in enumerate(graph_relationships):
		rel_text = relationships[rel[2]]
		if (len(rel_filter_in) == 0 or (rel_text.rstrip() in rel_filter_in)) and not rel_text.rstrip() in rel_filter_out:
			if (len(obj_ids) == 0) or ((rel[1] in obj_ids) and (rel[0] in obj_ids)):
				index = str(rel[0]) + "_" + str(rel[1])
				if index not in edges:
					edges[index] = []
					if edge_mask is not None:
						joined_edge_mask[index] = []
				edges[index].append(rel[3])
				if edge_mask is not None:
					joined_edge_mask[index].append(edge_mask[i])

	for (i,edge) in enumerate(edges):
		edge_obj_sub = edge.split("_")
		rels = ', '.join(edges[edge])
		if edge_mask is not None and 0 in joined_edge_mask[edge]:
			g.edge(str(edge_obj_sub[0]), str(edge_obj_sub[1]), label=rels, color='red', style='dotted')
		else:
			g.edge(str(edge_obj_sub[0]), str(edge_obj_sub[1]), label=rels, color='grey')

#4d3d82b6-8cf4-2e04-830a-4303fa0e79c7
def run(use_sampled_graphs=True, scan_id="6e67e550-1209-2cd0-8294-7cc2564cf82c", split="1", with_manipulation=False,
				data_path='./GT', outfolder="./vis_graphs/", graphfile='graphs_layout.yml', gen_custom_scene=True):

	if use_sampled_graphs:
		# use this option to customize your own graphs in the yaml format
		palette_json = os.path.join(data_path, "color_palette.json")
		color_palette = json.load(open(palette_json, 'r'))['hex']
		graph_yaml = os.path.join(data_path, graphfile)
	else:
		# use this option to read scene graphs from the dataset
		# relationships_json = os.path.join(data_path, 'relationships_validation_clean.json') #"relationships_train.json")
		# objects_json = os.path.join(data_path, "objects.json")
		relationships_json = os.path.join(data_path, 'relationships_validation_one.json')
		objects_json = os.path.join(data_path, "objects.json")
		if gen_custom_scene:
			objects_json = os.path.join(data_path, "objects_one.json")
			convert_relationship_to_object(relationships_json, objects_json)
			# breakpoint()	

	relationships = viz_util.read_relationships(os.path.join(data_path, "relationships.txt"))

	if use_sampled_graphs:
			rel_label_to_id = {}
			for (i,r) in enumerate(relationships):
				rel_label_to_id[r] = i
			graph = viz_util.load_semantic_scene_graphs_custom(graph_yaml, color_palette, rel_label_to_id, with_manipuation=False)
			if with_manipulation:
				graph_mani = viz_util.load_semantic_scene_graphs_custom(graph_yaml, color_palette, rel_label_to_id, with_manipuation=True)
	else:
		graph = viz_util.load_semantic_scene_graphs(relationships_json, objects_json)

	if split != '':
		scan_id = scan_id + '_' + split

	filter_dict_in = [] 
	filter_dict_out = [] # ["left", "right", "behind", "front", "same as", "same symmetry as", "bigger than", "lower than", "higher than", "close by"]
	for scan_id in [scan_id]:
		visualize_scene_graph(graph[scan_id], relationships, filter_dict_in, filter_dict_out, [], "v1", scan_id=scan_id,
													outfolder=outfolder)
		if with_manipulation and use_sampled_graphs:
			# manipulation only supported for custom graphs
			visualize_scene_graph(graph_mani[scan_id], relationships, filter_dict_in, filter_dict_out, [], "v1", scan_id=scan_id + "_mani",
														outfolder=outfolder)

	idx = [o['id'] for o in graph[scan_id]['objects']]
	color = [o['ply_color'] for o in graph[scan_id]['objects']]
	# return used colors so that they can be used for 3D model visualization
	return dict(zip(idx, color))

def convert_relationship_to_object(relationship_path, output_path):
    colors_hex = [
    "#98df8a",   
    "#2ca02c",    
    "#ff9896",    
    "#d62728",    
    "#c5b0d5",    
    "#9467bd",    
    "#c49c94",    
    "#8c564b",    
    "#f7b6d2",
	"#e377c2",
	"#dbdb8d",     
	]
    
    # relationship_path = os.path.abspath(relationship_path)
    # output_path = os.path.abspath(output_path)
    
    with open(relationship_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    objects = []
    seen_ids = set()
    for scan in data["scans"]:
        for obj_id, label in scan["objects"].items():
            if obj_id not in seen_ids:
                objects.append({
                    "ply_color": random.choices(colors_hex)[0],
                    "label": label,
                    "id": obj_id,
                    "global_id": str(random.randint(1, 200)),
                    "affordances": [],
                    "attributes": {}
                })
                seen_ids.add(obj_id)
    
    output_data = {"scans": [{"scan": data["scans"][0]["scan"], "objects": objects}]} # only one scan
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)
if __name__ == "__main__": 
	run(use_sampled_graphs=False)