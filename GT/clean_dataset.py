import json
from typing import List

def find_all_obj(json_file) -> List[str]:
    obj_set = set()

    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        for scan in data['scans']:
            objs = list(scan['objects'].values())
            obj_set.update(objs)
            
    return obj_set


import json

def filter_relationships(obj_list, input_file_path,  output_file_path):
    """
    input_file_path: path to the original JSON file to be filtered 
    obj_list: list of objects that need to be kept
    output_file_path: path to the new JSON file to save the filtered data
    """
    with open(input_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    for scan in data["scans"]:
        objects = scan["objects"]
        
        invalid_objects = {key for key, value in objects.items() if value not in obj_list}
        scan["objects"] = {key: value for key, value in objects.items() if key not in invalid_objects}

        scan["relationships"] = [
            relationship for relationship in scan["relationships"]
            if relationship[0] not in invalid_objects and 
               relationship[1] not in invalid_objects
        ]

    with open(output_file_path, 'w') as json_file:
        json.dump(data, json_file)
        print("filtered file saved!!")


def remove_isolated_obj(input_file_path, output_file_path):
    """
    input_file_path: path to the original JSON file to be filtered
    output_file_path: path to the new JSON file to save the filtered data
    """

    with open(input_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    for scan in data["scans"]:
        objects = scan["objects"]
        relationships = scan["relationships"]

        related_objects = set()
        for relationship in relationships:
            related_objects.update(relationship[:2]) 

        scan["objects"] = {key: value for key, value in objects.items() if key in related_objects}

    with open(output_file_path, 'w') as json_file:
        json.dump(data, json_file) 
        print("Filtered file saved!!")



if __name__ == "__main__":
    obj_set_train = find_all_obj("./GT/relationships_train_clean.json")
    obj_set_val = find_all_obj("./GT/relationships_validation_clean.json")
    print(f"num train objs: {len(obj_set_train)}")
    print(f"num train objs: {len(obj_set_val)}")
    obj_set_common = obj_set_train.intersection(obj_set_val)
    print(f"num common objs: {len(obj_set_common)}")

    # filter validation set
    input_file_path = "./GT/relationships_validation_clean.json"
    output_file_path = "./GT/relationships_validation_filtered.json"
    filter_relationships(list(obj_set_common), input_file_path, output_file_path)

    # filter train set
    input_file_path = "./GT/relationships_train_clean.json"
    output_file_path = "./GT/relationships_train_filtered.json"
    filter_relationships(list(obj_set_common), input_file_path, output_file_path)


    filtered_obj_num = len(find_all_obj("./GT/relationships_train_filtered.json"))
    print(f"filtered obj num: {filtered_obj_num}")
