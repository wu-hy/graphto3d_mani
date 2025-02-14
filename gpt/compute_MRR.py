# given a txt file containing GT triplets for a SG, 
# we compute Mean Reciprocal Rank.

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm
from MRR_utils import get_next_token_rank
import numpy as np
import torch
import re

chpt_path = "/cluster/project/cvg/students/shangwu/SceneVerse/preprocess/gpt/output_3RScan/checkpoint-3120"
tokenizer_path = chpt_path + "/tokenizer.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.eos_token = "[EOS]"
tokenizer.bos_token = "[CLS]"
tokenizer.sep_token = "[SEP]"
tokenizer.unk_token = "[UNK]"
tokenizer.pad_token = "[PAD]"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained(chpt_path).to(device)
scene_path = "/cluster/project/cvg/students/shangwu/3RScan/triplets/3RScan/0ad2d3a1-79e2-2212-9b99-a96495d9f7fe/triplets.txt"
# scene_path = "/cluster/project/cvg/students/shangwu/3RScan/triplets/3RScan/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca/triplets.txt"
gt_relationships = []

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
with open(scene_path, "r", encoding="utf-8") as file:
    file_content = file.readlines()  # Read all lines in the file
    gt_relationships = [
                f"{line.split()[2]}"
                for line in file_content if len(line.split()) >= 3
            ]
    lines = [
                f"[SUB] {split_if_needed(line.split()[0])} [/SUB] "
                f"[OBJ] {split_if_needed(line.split()[1])} [/OBJ] "
                f"[REL] {line.split()[2]} [/REL]"
                for line in file_content if len(line.split()) >= 3
            ]
input_text = ""

assert len(lines) == len(gt_relationships)
ranks = []
model.eval()
with torch.no_grad():
    for id, line in enumerate(tqdm(lines)):
        
        input_text += " ".join(line.split()[:-2]) + " "
        rank, generated_id = get_next_token_rank(model, tokenizer, input_text, gt_relationships[id])
        ranks.append(rank)
        # input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # generated_ids = model.generate(
        #     input_ids,
        #     max_new_tokens=1,
        #     do_sample=True,
        #     temperature=0.75,
        #     eos_token_id=tokenizer.eos_token_id
        # )
        generated_rel = tokenizer.decode(generated_id)
        input_text += f"{generated_rel} [/REL] "
    
# Mean Rank
mean_rank = np.mean(ranks)

# Mean Reciprocal Rank (MRR)
mrr = np.mean([1.0 / r for r in ranks])

# Hits@k
ks = [1,3,10]
hits = []
for k in ks:
    hits.append(np.mean([1.0 if r <= k else 0.0 for r in ranks]))
    
assert len(ks) == len(hits)

print(f"Mean Rank: {mean_rank}, MRR: {mrr}, hits@{ks}: {hits}")

