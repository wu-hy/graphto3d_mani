# config.py

config_params = {
    "context_length": 1024,
    "n_layer": 6,
    "n_head": 8,
    "n_emb": 512,
    "dataset_path": "/cluster/project/cvg/students/shangwu/3RScan/triplets/3RScan/*/triplets.txt",
    "tokenizer_file": "tokenizer.json",
    "output_dir": "output_3RScan",
    "learning_rate": 5e-4,
    "steps": 300,
    "seed": 42,
}
