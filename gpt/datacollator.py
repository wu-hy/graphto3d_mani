# datacollator.py

from transformers import DataCollatorForLanguageModeling

class MyDataCollator(DataCollatorForLanguageModeling):
    """
    A data collator that masks out (sets to -100) any label token that's
    NOT inside [REL] ... [/REL]. This means only tokens within the [REL]
    ... [/REL] region contribute to the language modeling loss.
    """
    def __init__(self, tokenizer, mlm=False, pad_to_multiple_of=None):
        super().__init__(tokenizer=tokenizer, mlm=mlm, pad_to_multiple_of=pad_to_multiple_of)
        # Convert your special marker tokens to IDs:
        self.rel_start_id = tokenizer.convert_tokens_to_ids("[REL]")
        self.rel_end_id   = tokenizer.convert_tokens_to_ids("[/REL]")

    def torch_call(self, examples):
        # Step 1: Let the parent class do its usual job of padding,
        # copying input_ids -> labels, etc.
        batch = super().torch_call(examples)
        
        # Step 2: Now we apply extra label-masking logic.
        labels = batch["labels"]  # shape: (batch_size, seq_len)

        for i in range(labels.size(0)):
            row = labels[i]
            in_rel_region = False

            for j in range(row.size(0)):
                token_id = row[j].item()

                if token_id == self.rel_start_id:
                    # Enter the [REL] region (keep this token's label)
                    in_rel_region = True
                    row[j] = -100
                elif token_id == self.rel_end_id:
                    # This token is [/REL], still inside the region,
                    # but after this, we're out.
                    in_rel_region = False
                    row[j] = -100
                else:
                    # If we're not inside the [REL] region, set -100
                    if not in_rel_region:
                        row[j] = -100

        batch["labels"] = labels
        return batch
