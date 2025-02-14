from eval_scene_json import eval_scene
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import json
import torch
from tqdm import tqdm

chpt_path = "/cluster/project/cvg/students/shangwu/SceneVerse/preprocess/gpt/output_3RScan/checkpoint-8520"
tokenizer_path = chpt_path + "/tokenizer.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

tokenizer.eos_token = "[EOS]"
tokenizer.bos_token = "[CLS]"
tokenizer.sep_token = "[SEP]"
tokenizer.unk_token = "[UNK]"
tokenizer.pad_token = "[PAD]"

model = GPT2LMHeadModel.from_pretrained(chpt_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

json_path = "/cluster/project/cvg/students/shangwu/graphto3d/GT/3DSSG_processed_files/relationships_validation_clean.json"
scan_id = "ab835faa-54c6-29a1-9b55-1a5217fcba19"
split = 1

with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
scans = data.get("scans", [])
if not scans:
    raise ValueError("No scans found in the provided JSON file.")

with torch.no_grad():
    for scan in tqdm(scans):
        eval_scene(model, tokenizer, scan)

