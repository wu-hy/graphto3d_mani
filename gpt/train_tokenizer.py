import re
import glob
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit  # Only split on whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, Dataset

# -------------------------------
# Step 1: Load and Prepare the Dataset
# -------------------------------

dataset_files = glob.glob("/cluster/project/cvg/students/shangwu/3RScan/triplets/3RScan/*/triplets.txt")
dataset = load_dataset("text", data_files=dataset_files)

# Convert dataset to a DataFrame (include any cleaning/exploration as needed)
df = dataset["train"].to_pandas()

# Convert the DataFrame back to a Hugging Face Dataset and split it
ds = Dataset.from_pandas(df)
raw_datasets = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)

# -------------------------------
# Step 2: Define a Cleaning Function
# -------------------------------
def clean_text(text):
    """
    Removes a trailing underscore and number (e.g., '_1') from words.
    
    This regex targets words that end with an underscore followed by one or more digits.
    For example:
      "laundry_basket_1"  -> "laundry_basket"
      "desk_2"            -> "desk"
      
    It does not affect words that have underscores elsewhere or include punctuation.
    For instance, "3_o‘clock_direction_near" will not be changed because it does not 
    end with an underscore and digits.
    """
    return re.sub(r"\b(\w+)_\d+\b", r"\1", text)

# -------------------------------
# Step 3: Initialize the Tokenizer
# -------------------------------

# Create a WordLevel tokenizer with an unknown token.
new_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

# Use only a whitespace pre-tokenizer so that punctuation is kept intact.
new_tokenizer.pre_tokenizer = WhitespaceSplit()

# -------------------------------
# Step 4: Prepare the Training Corpus Iterator
# -------------------------------
def get_training_corpus():
    """
    Yields batches of cleaned texts for training.
    Also appends a sample containing numbers 0-50 (space-separated) to ensure these tokens are included.
    """
    dataset = raw_datasets["train"]
    # Process in batches of 1,000 samples
    for i in range(0, len(dataset), 1000):
        texts = dataset[i : i + 1000]["text"]
        # Apply cleaning to each text sample
        cleaned_texts = [clean_text(t) for t in texts]
        yield cleaned_texts

    # Append an extra sample that lists the numbers 0-50 (space-separated)
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
