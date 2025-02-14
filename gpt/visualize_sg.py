import networkx as nx
import matplotlib.pyplot as plt

from PIL import Image
import io

import argparse
import os

def draw_scene_graph_and_save(sequence, output_file="scene_graph.png"):
    """
    Draws a scene graph from a sequence of triplets and saves it as a PNG file.

    Args:
        sequence (str): A string of triplets representing the scene graph, ending with GRAPH_END.
                        Example: "floor_1 shelf_1 support floor_1 bed_1 support GRAPH_END"
        output_file (str): The file path to save the graph image.
    """
    if not sequence.endswith("GRAPH_END"):
        print("Warning: The sequence should end with GRAPH_END. Appending GRAPH_END to the sequence.")
        sequence += " GRAPH_END"

    # Remove GRAPH_END and split the sequence into tokens
    tokens = sequence.replace("GRAPH_END", "").split()

    if len(tokens) % 3 != 0:
        # raise a warning and remove the last incomplete triplet
        print("Warning: The sequence should be divisible into triplets of (subject, object, relation). Proceed with the complete triplets.")
        tokens = tokens[:-(len(tokens) % 3)]

    # Create a directed graph
    G = nx.DiGraph()

    # Parse triplets and add edges to the graph
    for i in range(0, len(tokens), 3):
        subject = tokens[i]
        obj = tokens[i + 1]
        relation = tokens[i + 2]
        G.add_edge(subject, obj, label=relation)

    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for positioning nodes
    plt.figure(figsize=(10, 8))
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Save the plot to a file
    plt.title("Scene Graph")
    plt.savefig(output_file, format='png')
    plt.close()

# Example usage
# sequence = "floor_1 shelf_1 support floor_1 bed_1 support floor_1 suitcase_1 support floor_1 poster_1 support GRAPH_END"
# draw_scene_graph(sequence, "scene_graph.png")


def draw_scene_graph(sequence):
    """
    Draws a scene graph from a sequence of triplets and returns it as a PIL Image.

    Args:
        sequence (str): A string of triplets representing the scene graph, ending with GRAPH_END.
                        Example: "floor_1 shelf_1 support floor_1 bed_1 support GRAPH_END"

    Returns:
        PIL.Image: The image of the scene graph.
    """
    if not sequence.endswith("GRAPH_END"):
        print("Warning: The sequence should end with GRAPH_END. Appending GRAPH_END to the sequence.")
        sequence += " GRAPH_END"

    # Remove GRAPH_END and split the sequence into tokens
    tokens = sequence.replace("GRAPH_END", "").split()

    if len(tokens) % 3 != 0:
        # raise a warning and remove the last incomplete triplet
        print("Warning: The sequence should be divisible into triplets of (subject, object, relation). Proceed with the complete triplets.")
        tokens = tokens[:-(len(tokens) % 3)]
    # Create a directed graph
    G = nx.DiGraph()

    # Parse triplets and add edges to the graph
    for i in range(0, len(tokens), 3):
        subject = tokens[i]
        obj = tokens[i + 1]
        relation = tokens[i + 2]
        G.add_edge(subject, obj, label=relation)

    # Check for an empty graph
    if G.number_of_nodes() == 0:
        raise ValueError("The sequence does not define a valid graph.")

    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for positioning nodes
    plt.figure(figsize=(10, 8))

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Save the plot to a buffer and convert to PIL Image
    with io.BytesIO() as buf:
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img.load()
    plt.close()

    return img

if __name__ == "__main__":
    # this script accepts a dir to triplets.txt file and outputs a scene graph image and saves it in the same dir


    parser = argparse.ArgumentParser(description="Visualize scene graph from triplets.txt file")
    parser.add_argument("triplets_file", type=str, help="Path to the triplets.txt file")
    args = parser.parse_args()

    with open(args.triplets_file, 'r') as f:
        sequence = f.read()

    img = draw_scene_graph(sequence)
    output_file = os.path.splitext(args.triplets_file)[0] + ".png"

    img.save(output_file)
    print(f"Scene graph image saved to {output_file}")


