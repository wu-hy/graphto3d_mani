
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
# from visualize_sg import draw_scene_graph_and_save
from tqdm import tqdm
import re
import json
import os


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

def eval_scene(model, tokenizer, scan, save_dir=None, vis_triplets=False):
    device = next(model.parameters()).device
    scan_id = scan.get("scan", [])
    split = scan.get("split", [])
    objects = scan.get("objects", {})
    # append indices to all values of object dict
    new_objects = {}
    name_count = {}  # Keep track of how many times each object name has been seen.
    for obj_id, name in objects.items():
        # Increment the count for this object name.
        name = name.replace(' ', '_')
        count = name_count.get(name, 0) + 1
        name_count[name] = count
        # Append the count to the name
        new_objects[obj_id] = f"{name} {count}"
    # Replace the original objects with the updated version.
    objects = new_objects

    relationships = scan.get("relationships", [])

    triplets = []
    gt_triplets = []
    for rel in relationships:
        # Assume each relationship is a 4-element list:
        # [subject_id, object_id, score, relation_label]
        subj_id = str(rel[0])  # cast to string because keys in 'objects' are strings
        obj_id  = str(rel[1])
        relation_label = str(rel[3]) 
        
        # Look up the corresponding names (or use a placeholder if missing)
        try:
            subject_label = str(objects[subj_id])
        except:
            print(f"fail to find subject in {scan.get('scan', {})} for subject id {subj_id} in rel {rel}")
            continue
        try:
            object_label = str(objects[obj_id])
        except:
            print(f"fail to find object in {scan.get('scan', {})} for object id {obj_id} in rel {rel}")
            continue
        # object_token  = objects.get(obj_id, f"unknown_{obj_id}")
        
        # Optionally split tokens if they match the pattern (e.g., "washing_machine_3" becomes "washing_machine 3")
        # subject_label = split_if_needed(str(subject_token))
        # object_label  = split_if_needed(str(object_token))
        
        # Create the triplet string
        relation_label = relation_label.replace(' ', '_')
        triplet = f"[SUB] {subject_label} [/SUB] [OBJ] {object_label} [/OBJ] [REL] {relation_label} [/REL]"
        triplets.append(triplet)
        gt_triplets.append(f"{subject_label} {object_label} {relation_label}")


        

    input_text = ""
    for line in triplets:
        input_text += " ".join(line.split()[:-2]) + " "
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=True,
            temperature=0.6,
            eos_token_id=tokenizer.eos_token_id
        )
        input_text = tokenizer.decode(generated_ids[0])
        if "[UNK]" in input_text:
            print(f"The file {scan_id} contains unknowns")
        input_text += " [/REL] "
    if "[UNK]" in input_text:
        print(f"The file {scan_id} contains unknowns")
    def normalize_triplets(input_text):
        
        # break the input_text string into triplets and skip [SUB] and [OBJ]
        segments = input_text.split(" [/REL] ")

        lines = []

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Extract and process SUB content: join words with underscores.
            sub_match = re.search(r'\[SUB\](.*?)\[/SUB\]', segment, re.DOTALL)
            sub_text = ' '.join(sub_match.group(1).split()) if sub_match else ''

            # Extract and process OBJ content: join words with underscores.
            obj_match = re.search(r'\[OBJ\](.*?)\[/OBJ\]', segment, re.DOTALL)
            obj_text = ' '.join(obj_match.group(1).split()) if obj_match else ''

            # Extract REL content: simply remove the tag and trim whitespace.
            # Note: Because we split on " [/REL] ", the closing tag is already removed.
            rel_match = re.search(r'\[REL\](.*)', segment, re.DOTALL)
            rel_text = rel_match.group(1).strip() if rel_match else ''

            # Combine the processed parts into one line.
            parts = []
            if sub_text:
                parts.append(sub_text)
            if obj_text:
                parts.append(obj_text)
            if rel_text:
                parts.append(rel_text)

            lines.append(" ".join(parts))
        return lines

    lines = normalize_triplets(input_text)
    # gt_lines = "\n".join(gt_triplets)
    # write to output.txt
    
    if not save_dir:
        save_dir = f"./vis_triplets/{scan_id}/{split}"
    os.makedirs(save_dir, exist_ok=True)
    if vis_triplets:
        with open(f"{save_dir}/output.txt", "w") as f:
            f.write("\n".join(lines))
        with open(f"{save_dir}/gt.txt", "w") as f:
            f.write("\n".join(gt_triplets))

    relationships = triplets_to_relation_json(lines, objects, rel_dict)
    scan["relationships"] = relationships
    output = {"scans": []}
    output["scans"].append(scan)
    with open(f"{save_dir}/relationships_validation_one.json", "w") as f:
        json.dump(output, f)
    # print(lines)
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
    json_path = "./GT/relationships_validation_clean.json"
    
    with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    scans = data.get("scans", [])
    if not scans:
        raise ValueError("No scans found in the provided JSON file.")

    try:
        scan = [d for d in scans if d["scan"] == scan_id and d["split"] == split]
        # scan = scans['scan']
    except:
        raise ValueError(f"error loading scan {scan_id}_{split}")
    scan = scan[0]
    eval_scene(model, tokenizer, scan, save_dir="./GT")
if __name__ == "__main__":
    main()
    