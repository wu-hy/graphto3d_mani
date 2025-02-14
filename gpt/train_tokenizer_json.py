import re
import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit  # Only split on whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

# -------------------------------
# Step 1: Load and Prepare the JSON Dataset
# -------------------------------
json_path = "/cluster/project/cvg/students/shangwu/graphto3d/GT/3DSSG_processed_files/relationships_train_clean.json"  # <-- update with your JSON file path
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

scans = data.get("scans", [])
if not scans:
    raise ValueError("No scans found in the JSON file.")

# -------------------------------
# Step 2: (Optional) Define a Cleaning Function
# -------------------------------
# Since we want to keep the appended indices (e.g., "curtain_1"),
# this function is a pass-through.
def clean_text(text):
    return text

# -------------------------------
# Step 3: Initialize the Tokenizer
# -------------------------------
# We use a WordLevel tokenizer with a specified unknown token.
new_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
# Use a whitespace pre-tokenizer so that punctuation and symbols are kept intact.
new_tokenizer.pre_tokenizer = WhitespaceSplit()

# -------------------------------
# Step 4: Prepare the Training Corpus Iterator
# -------------------------------
def get_training_corpus():
    """
    Yields batches of cleaned texts for training the tokenizer.
    For each scan in the JSON, we collect all object names (from the "objects" dict),
    appending an index to each occurrence (e.g., "curtain_1", "curtain_2", etc.).
    Finally, we yield an extra sample that includes the numbers 0 to 50.
    """
    corpus_samples = []
    
    # Process each scan: for each, gather the object names with appended indices.
    for scan in scans:
        objects = scan.get("objects", {})
        # name_count = {}  # track counts for each object name in this scan
        tokens = list(objects.values())
        tokens = [word.replace(" ", "_") for word in tokens]
        # tokens = []
        # for obj_id, name in objects.items():
        #     count = name_count.get(name, 0) + 1
        #     name_count[name] = count
        #     tokens.append(f"{name}_{count}")
        if tokens:
            # Join tokens with spaces to form a single text sample for this scan.
            corpus_samples.append(" ".join(tokens))
    with open("/cluster/project/cvg/students/shangwu/graphto3d/GT/relationships.txt", "r", encoding="utf-8") as f:
        content = f.readlines()
        content = [word.replace(" ", "_") for word in content]
        corpus_samples.append(" ".join(content))
        
    # Yield the samples in batches (batch size 1000 in this example).
    batch_size = 1000
    for i in range(0, len(corpus_samples), batch_size):
        # Apply cleaning if necessary (here clean_text is a pass-through)
        yield [clean_text(text) for text in corpus_samples[i : i + batch_size]]
    
    # Finally, append an extra sample with the numbers 0 to 50 (space-separated)
    extra_sample = " ".join(str(i) for i in range(51))
    yield [extra_sample]

# -------------------------------
# Step 5: Set Up the Trainer and Train the Tokenizer
# -------------------------------
special_tokens = [
    "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]",
    "[SUB]", "[/SUB]", "[OBJ]", "[/OBJ]", "[REL]", "[/REL]",
    "[EOS]"
]

trainer = WordLevelTrainer(special_tokens=special_tokens)
new_tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# -------------------------------
# Step 6: Save the Tokenizer
# -------------------------------
new_tokenizer.save("tokenizer.json")
