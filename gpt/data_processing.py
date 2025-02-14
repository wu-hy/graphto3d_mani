# data_processing.py

import glob
import random  # Import the random module for shuffling
import pandas as pd
from datasets import Dataset, DatasetDict
import re
import json

def load_and_prepare_dataset(dataset_path, tokenizer, context_length, seed=None, num_augmentations=1): 
    """
    Loads and prepares the dataset by reading files, shuffling triplets reproducibly
    if a seed is provided, and tokenizing the resulting text.
    
    Args:
        dataset_path (str): The glob path to the dataset files.
        tokenizer: The tokenizer to use.
        context_length (int): Maximum length of the tokenized sequence.
        seed (int, optional): Base random seed for reproducible shuffling.
        num_augmentations (int, optional): Number of augmented (shuffled) versions per graph.
                                           If 1, then only a single (shuffled) version is created.
    
    Returns:
        A Hugging Face DatasetDict containing the tokenized data.
    """
    dataset_files = glob.glob(dataset_path)
    if not dataset_files:
        raise FileNotFoundError(f"No files found in {dataset_path}")

    rows = []  # List to store the transformed content of each file


    # Define a regex pattern: one or more characters (non-greedy) followed by an underscore and one or more digits at the end of the string.
    pattern = re.compile(r"^(.+?)_(\d+)$")

    def split_if_needed(token):
        """
        If the token matches the pattern 'word_index' (e.g. washing_machine_3),
        split it into 'word' and 'index', separated by a space.
        Otherwise, return the token unchanged.
        """
        match = pattern.match(token)
        if match:
            # Return the two parts joined by a space (or any delimiter you prefer)
            return f"{match.group(1)} {match.group(2)}"
        return token
    # Process each file
    for file_idx, file_path in enumerate(dataset_files):
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.readlines()  # Read all lines in the file
            # Generate a list of triplets for lines that have at least three words
            triplets_original = [
                f"[SUB] {split_if_needed(line.split()[0])} [/SUB] "
                f"[OBJ] {split_if_needed(line.split()[1])} [/OBJ] "
                f"[REL] {line.split()[2]} [/REL]"
                for line in file_content if len(line.split()) >= 3
            ]
            # For each augmentation, create a shuffled version of the graph
            for aug_idx in range(num_augmentations):
                # If a seed is provided, adjust it uniquely for each file and augmentation
                if seed is not None:
                    random.seed(seed + file_idx * num_augmentations + aug_idx)
                # Make a copy of the original triplets and shuffle it
                triplets = triplets_original.copy()
                random.shuffle(triplets)
                # Combine all shuffled triplets into a single string representing one graph
                merged_text = " ".join(triplets)
                rows.append(merged_text)  # Add the merged string as a separate example
    # Create a DataFrame where each row corresponds to one graph's content (including augmentations)
    df = pd.DataFrame({"text": rows})
    
    print(df.head())  # Print the first few rows for debugging
    
    # Convert the DataFrame back to a Hugging Face Dataset
    ds = Dataset.from_pandas(df)
    # If we have more than one example, perform a train-test split.
    if len(ds) > 1:
        raw_datasets = ds.train_test_split(test_size=0.1, shuffle=True)
    else:
        # Put all data in "train" and use it for test as well
        raw_datasets = DatasetDict({"train": ds, "test": ds})
    
    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            padding=False
        )
        return {"input_ids": outputs["input_ids"]}
    
    return raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)

def load_and_prepare_dataset_from_json(json_path, tokenizer, context_length, seed=None, num_augmentations=1):
    """
    Loads and prepares the dataset from a JSON file containing scans.
    Each scan is expected to be a dictionary with:
      - "objects": a dict mapping object IDs (as strings) to object names.
      - "relationships": a list of relationships, each of which is a 4-element list:
           [subject_id, object_id, score, relation_label]
    For each relationship, a text string is generated in the format:
         "[SUB] subject_name [/SUB] [OBJ] object_name [/OBJ] [REL] relation_label [/REL]"
    Optionally, multiple augmented (shuffled) versions are generated per scan.
    Finally, the texts are tokenized and returned as a Hugging Face DatasetDict.

    Args:
        json_path (str): Path to the JSON file.
        tokenizer: The tokenizer to use.
        context_length (int): Maximum length of the tokenized sequence.
        seed (int, optional): Base random seed for reproducible shuffling.
        num_augmentations (int, optional): Number of shuffled versions to create per scan.
    
    Returns:
        A Hugging Face DatasetDict containing the tokenized data.
    """
    # Load the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scans = data.get("scans", [])
    if not scans:
        raise ValueError("No scans found in the provided JSON file.")

    rows = []  # This will store the merged text for each (augmented) scan.
    
    # Process each scan in the JSON
    for scan_idx, scan in enumerate(scans):
        # Get the objects mapping and relationships list for the current scan.
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
        
        triplets_original = []
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
            triplet = f"[SUB] {subject_label} [/SUB] [OBJ] {object_label} [/OBJ] [REL] {relation_label.replace(' ', '_')} [/REL]"
            triplets_original.append(triplet)
        
        # For each augmentation, shuffle the triplets and combine them into one text.
        for aug_idx in range(num_augmentations):
            if seed is not None:
                random.seed(seed + scan_idx * num_augmentations + aug_idx)
            triplets = triplets_original.copy()
            random.shuffle(triplets)
            merged_text = " ".join(triplets)
            rows.append(merged_text)

    # Create a DataFrame with one row per (augmented) scan
    df = pd.DataFrame({"text": rows})
    df.to_csv('data.csv', index=False)
    print(df.head())  # Debug: print first few rows
    
    # Convert the DataFrame into a Hugging Face Dataset.
    ds = Dataset.from_pandas(df)
    if len(ds) > 1:
        raw_datasets = ds.train_test_split(test_size=0.1, shuffle=True)
    else:
        raw_datasets = DatasetDict({"train": ds, "test": ds})
    
    # Define a tokenize function that applies your tokenizer
    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            padding=False
        )
        return {"input_ids": outputs["input_ids"]}
    
    # Map the tokenizer over the dataset, removing the original "text" column.
    return raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)
        
    