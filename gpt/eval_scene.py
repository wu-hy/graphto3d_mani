
from transformers import AutoConfig, GPT2LMHeadModel, PreTrainedTokenizerFast
from visualize_sg import draw_scene_graph_and_save
from tqdm import tqdm
import re

chpt_path = "/cluster/project/cvg/students/shangwu/SceneVerse/preprocess/gpt/output_3RScan/checkpoint-3120"
tokenizer_path = chpt_path + "/tokenizer.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

tokenizer.eos_token = "[EOS]"
tokenizer.bos_token = "[CLS]"
tokenizer.sep_token = "[SEP]"
tokenizer.unk_token = "[UNK]"
tokenizer.pad_token = "[PAD]"

model = GPT2LMHeadModel.from_pretrained(chpt_path)
scene_path = "/cluster/project/cvg/students/shangwu/3RScan/triplets/3RScan/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca/triplets.txt"
# scene_path = "/cluster/project/cvg/students/shangwu/3RScan/triplets/3RScan/0ad2d3a1-79e2-2212-9b99-a96495d9f7fe/triplets.txt"

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
with open(scene_path, "r", encoding="utf-8") as file:
    file_content = file.readlines()  # Read all lines in the file
    lines = [
                f"[SUB] {split_if_needed(line.split()[0])} [/SUB] "
                f"[OBJ] {split_if_needed(line.split()[1])} [/OBJ] "
                f"[REL] {line.split()[2]} [/REL]"
                for line in file_content if len(line.split()) >= 3
            ]
input_text = ""
for line in tqdm(lines):
    input_text += " ".join(line.split()[:-2]) + " "
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=1,
        do_sample=True,
        temperature=0.6,
        eos_token_id=tokenizer.eos_token_id
    )
    input_text = tokenizer.decode(generated_ids[0])
    input_text += " [/REL] "

# break the input_text string into triplets and skip [SUB] and [OBJ]
segments = input_text.split(" [/REL] ")

lines = []

for segment in segments:
    segment = segment.strip()
    if not segment:
        continue

    # Extract and process SUB content: join words with underscores.
    sub_match = re.search(r'\[SUB\](.*?)\[/SUB\]', segment, re.DOTALL)
    sub_text = '_'.join(sub_match.group(1).split()) if sub_match else ''

    # Extract and process OBJ content: join words with underscores.
    obj_match = re.search(r'\[OBJ\](.*?)\[/OBJ\]', segment, re.DOTALL)
    obj_text = '_'.join(obj_match.group(1).split()) if obj_match else ''

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


# write to output.txt
with open("output.txt", "w") as f:
    f.write("\n".join(lines))
print(lines)

