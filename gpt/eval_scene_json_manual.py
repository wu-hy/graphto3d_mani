
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
# from visualize_sg import draw_scene_graph_and_save
from tqdm import tqdm
import re
import json
import os

from collections import defaultdict
rel_dict = {
    "none": 0,
    "supported by": 1,
    "left": 2,
    "right": 3,
    "front": 4,
    "behind": 5,
    "close by": 6,
    "inside": 7,
    "bigger than": 8,
    "smaller than": 9,
    "higher than": 10,
    "lower than": 11,
    "same symmetry as": 12,
    "same as": 13,
    "attached to": 14,
    "standing on": 15,
    "lying on": 16,
    "hanging on": 17,
    "connected to": 18,
    "leaning against": 19,
    "part of": 20,
    "belonging to": 21,
    "build in": 22,
    "standing in": 23,
    "cover": 24,
    "lying in": 25,
    "hanging in": 26
}

def construct_scan_from_input(scan_id, split, objects, relationships):
    scan = {
        "scan": scan_id,
        "split": split,
        "objects": objects,
        "relationships": relationships
    }
    return scan

def construct_objects_from_txt(txt_path):
    """
    Reads a text file where each line is formatted as:
      subject s_idx object o_idx predicate

    Returns two dictionaries:
      - json_obj: maps index to the entity name (without index)
      - obj: maps index to the full entity string (e.g. "subject_4")
    """
    objects = set()
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Split each line into 5 parts: subject, subject_index, object, object_index, predicate
            s, s_idx, o, o_idx, p = line.split(" ")
            objects.add(f"{s}\t{s_idx}")
            objects.add(f"{o}\t{o_idx}")
    
    # Convert the set to a list (order will be arbitrary)
    objects = list(objects)
    json_obj = {}
    obj = {}

    # Note: The original code reversed the order of variables from enumerate().
    # Here we use "for num, entry in enumerate(objects)" so that:
    #   num -> integer index; entry -> object string (e.g. "subject_4")
    for num, entry in enumerate(objects):
        obj[num] = entry
        # print(entry)
        # Split the entry into the name and its index; only keep the name in json_obj
        name, idx = entry.split("\t")
        json_obj[str(num)] = name.replace("_", " ")

    return json_obj, obj


def construct_relationships_from_txt(txt_path, objects, rel_dict):
    # txt: subject_i object_j predicate
    # output: [[subject id, object id, predicate id], ...]
    relationships = []
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            s, s_idx, o, o_idx, p = line.split()
            s_id = [k for k, v in objects.items() if v == f"{s}\t{s_idx}"]
            o_id = [k for k, v in objects.items() if v == f"{o}\t{o_idx}"]
            p_id = rel_dict[p.replace('_', ' ')]
            relationships.append([s_id[0], o_id[0], p_id, p.replace('_', ' ')])
    return relationships


    
def normalize_triplets(text):
    """
    Normalize the generated text into a list of triplet strings.
    Each triplet is formed by extracting the SUB, OBJ, and REL parts.
    """
    segments = text.split(" [/REL] ")
    normalized_lines = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Extract subject
        sub_match = re.search(r'\[SUB\](.*?)\[/SUB\]', segment, re.DOTALL)
        sub_text = ' '.join(sub_match.group(1).split()) if sub_match else ''

        # Extract object
        obj_match = re.search(r'\[OBJ\](.*?)\[/OBJ\]', segment, re.DOTALL)
        obj_text = ' '.join(obj_match.group(1).split()) if obj_match else ''

        # Extract relation
        rel_match = re.search(r'\[REL\](.*)', segment, re.DOTALL)
        rel_text = rel_match.group(1).strip() if rel_match else ''

        # Combine non-empty parts
        parts = [part for part in (sub_text, obj_text, rel_text) if part]
        normalized_lines.append(" ".join(parts))
    return normalized_lines


def eval_scene(model, tokenizer, scan, save_dir=None, vis_triplets=False, lines_to_keep=None, triplet_to_add=None):
    """
    Process a scan by generating triplets from relationships.
    Optionally, add extra triplets (via triplet_to_add) and visualize results.
    """
    # Ensure lines_to_keep is not mutable by default.
    if lines_to_keep is None:
        lines_to_keep = []

    # Determine the device from the model.
    device = next(model.parameters()).device

    # Retrieve scan metadata.
    scan_id = scan.get("scan", "")
    split = scan.get("split", "")
    objects = scan.get("objects", {})

    # Update object names: replace spaces with underscores and append a unique count.
    new_objects = {}
    name_count = defaultdict(int)
    for obj_id, name in objects.items():
        clean_name = name.replace(' ', '_')
        name_count[clean_name] += 1
        new_objects[obj_id] = f"{clean_name} {name_count[clean_name]}"
    objects = new_objects

    relationships = scan.get("relationships", [])
    triplets = []
    gt_triplets = []

    # Process the additional triplet, if provided.
    if triplet_to_add:
        print("Skipping and regenerating for triplet:", triplet_to_add)
        triplet_to_add_parts = triplet_to_add.split()
    else:
        triplet_to_add_parts = None

    # Process existing relationships into triplet strings.
    for rel in relationships:
        # Each relationship is expected as: [subject_id, object_id, score, relation_label]
        subj_id = str(rel[0])
        obj_id = str(rel[1])
        relation_label = str(rel[3]).replace(' ', '_')

        subject_label = objects.get(subj_id)
        object_label = objects.get(obj_id)
        if subject_label is None:
            print(f"Failed to find subject in {scan_id} for subject id {subj_id} in rel {rel}")
            continue
        if object_label is None:
            print(f"Failed to find object in {scan_id} for object id {obj_id} in rel {rel}")
            continue

        # Skip triplets that involve the nodes from triplet_to_add.
        if triplet_to_add_parts and (subject_label == triplet_to_add_parts[0] or object_label == triplet_to_add_parts[2]):
            continue

        triplet_str = f"[SUB] {subject_label} [/SUB] [OBJ] {object_label} [/OBJ] [REL] {relation_label} [/REL]"
        triplets.append(triplet_str)
        gt_triplets.append(f"{subject_label} {object_label} {relation_label}")

    # Process additional triplets if triplet_to_add is provided.
    add_triplets = []
    if triplet_to_add_parts:
        sub_label_add, sub_idx, obj_label_add, obj_idx, rel_label_add = triplet_to_add_parts
        sub_label_add = sub_label_add + " " + sub_idx
        obj_label_add = obj_label_add + " " + obj_idx
        # Add the main extra triplet.
        main_triplet = f"[SUB] {sub_label_add} [/SUB] [OBJ] {obj_label_add} [/OBJ] [REL] {rel_label_add} [/REL]"
        add_triplets.append(main_triplet)
        gt_triplets.append(f"{sub_label_add} {obj_label_add} {rel_label_add}")

        # Generate triplets from the subject to all other objects.
        for obj_name in objects.values():
            if obj_name in {sub_label_add, obj_label_add}:
                continue
            extra_triplet = f"[SUB] {sub_label_add} [/SUB] [OBJ] {obj_name} [/OBJ] [REL] [PAD] [/REL]"
            add_triplets.append(extra_triplet)
            gt_triplets.append(f"{sub_label_add} {obj_name} {rel_label_add}")

    def generate_next_token(current_text):
        """
        Given current_text, encode it, generate one new token, and return the decoded text.
        """
        input_ids = tokenizer.encode(current_text, return_tensors="pt").to(device)
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=True,
            temperature=0.6,
            eos_token_id=tokenizer.eos_token_ids
        )
        return tokenizer.decode(generated_ids[0])

    # Build the full input text by processing triplets.
    input_text = ""
    for idx, line in enumerate(triplets):
        if idx not in lines_to_keep:
            # Remove the last two tokens and generate a new token.
            prefix = " ".join(line.split()[:-2]) + " "
            input_text += prefix
            input_text = generate_next_token(input_text) + " [/REL] "
        else:
            input_text += line + " "

    # Append additional triplets if provided.
    if triplet_to_add_parts:
        input_text += add_triplets[0] + " "
        for line in add_triplets[1:]:
            prefix = " ".join(line.split()[:-2]) + " "
            input_text += prefix
            input_text = generate_next_token(input_text) + " [/REL] "

    if "[UNK]" in input_text:
        print(f"The file {scan_id} contains unknown tokens.")

    # Normalize the triplet text.
    normalized_lines = normalize_triplets(input_text)

    # Determine the save directory.
    if not save_dir:
        save_dir = os.path.join("./vis_triplets", scan_id, split)
    os.makedirs(save_dir, exist_ok=True)

    # Optionally, save visualizations of the triplets.
    if vis_triplets:
        with open(os.path.join(save_dir, "output.txt"), "w") as f:
            f.write("\n".join(normalized_lines))
        with open(os.path.join(save_dir, "gt.txt"), "w") as f:
            f.write("\n".join(gt_triplets))

    # Convert triplets into relationship JSON.
    relationships_json = triplets_to_relation_json(normalized_lines, objects, rel_dict)
    scan["relationships"] = relationships_json
    output = {"scans": [scan]}
    with open(os.path.join(save_dir, "relationships_validation_one.json"), "w") as f:
        json.dump(output, f)

    return

def triplets_to_relation_json(triplets, objects, rel_dict):
    # objects is not the same as the objects in the scan, but one processed to contain instance ids
    new_relationships = []
    for item in triplets:
        triplet = item.split()
        subj = str(f"{triplet[0]} {triplet[1]}")
        obj = str(f"{triplet[2]} {triplet[3]}")
        rel = triplet[4].replace('_', ' ')
        # look for the dict key for subj and obj
        subj_id = [k for k, v in objects.items() if v == subj]
        obj_id = [k for k, v in objects.items() if v == obj]
        if not subj_id:
            print(f"failed to find {subj}")
            continue
        if not obj_id:
            print(f"failed to find {obj}")
            continue
        subj_id = int(subj_id[0])
        obj_id = int(obj_id[0])
        # look for the dict key for rel
        rel_id = rel_dict.get(rel)
        if not rel_id:
            print(f"failed to find {rel}")
            continue
        new_relationships.append([subj_id, obj_id, rel_id, rel])
    return new_relationships
    
def main(scan_id="ab835faa-54c6-29a1-9b55-1a5217fcba19", split=1):
    chpt_path = "./gpt/output_3RScan/checkpoint-8520"
    tokenizer_path = chpt_path + "/tokenizer.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    tokenizer.eos_token = "[EOS]"
    tokenizer.bos_token = "[CLS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.pad_token = "[PAD]"

    model = GPT2LMHeadModel.from_pretrained(chpt_path)

    txt_path = "./GT/triplets.txt"
    print("generating scene from txt file!!")
    json_obj, obj = construct_objects_from_txt(txt_path)
    rels = construct_relationships_from_txt(txt_path, obj, rel_dict)
    scan = construct_scan_from_input(scan_id, split, json_obj, rels)
    lines_to_keep = [9, 14]
    triplet_to_add = "pillow 1 pillow 3 left"


    eval_scene(model, tokenizer, scan, save_dir="./GT", lines_to_keep=list(lines_to_keep), triplet_to_add=triplet_to_add)
if __name__ == "__main__":
    main("fake-scan", 1)
    