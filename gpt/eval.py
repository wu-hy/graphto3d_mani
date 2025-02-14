
from transformers import AutoConfig, GPT2LMHeadModel, PreTrainedTokenizerFast
from visualize_sg import draw_scene_graph_and_save


# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

chpt_path = "/cluster/project/cvg/students/shangwu/SceneVerse/preprocess/gpt/output_3RScan/checkpoint-8520"
# model = GPT2LMHeadModel.from_pretrained('/cluster/project/cvg/students/shangwu/SceneVerse/preprocess/gpt/output_3RScan/checkpoint-780')
# model = GPT2LMHeadModel.from_pretrained('/cluster/project/cvg/students/shangwu/SceneVerse/preprocess/gpt/output_one_scene/checkpoint-50')
tokenizer_path = chpt_path + "/tokenizer.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.eos_token = "[EOS]"
tokenizer.bos_token = "[CLS]"
tokenizer.sep_token = "[SEP]"
tokenizer.unk_token = "[UNK]"
tokenizer.pad_token = "[PAD]"

model = GPT2LMHeadModel.from_pretrained(chpt_path)
# input_text = "floor_1 shelf_1 support floor_1 bed_1 support"
input_text = "[SUB] shower_curtain 1 [/SUB] [OBJ] wall 1 [/OBJ] [REL]"
# input_text = "[SUB] shower_1 [/SUB] [OBJ] floor_1 [/OBJ] [REL]"
# input_text = "[SUB] chair_2 [/SUB] [OBJ] couch_1 [/OBJ] [REL] 3_o‘clock_direction_near [/REL] [SUB] couch_1 [/SUB] [OBJ] chair_2 [/OBJ] [REL]"
# input_text = "[SUB] pillow_1 [/SUB] [OBJ] couch_1 [/OBJ] [REL] placed_within_the_area_of [/REL] [SUB] pillow_2 [/SUB] [OBJ] couch_1 [/OBJ] [REL]"
#   placed_within_the_area_of
# pillow_2 couch_1 inside
# input_text = "placed_within_the_area_of 3_o‘clock_direction_near"
# chair_2 couch_1
# couch_1 chair_2
# shower_curtain_1 wall hung_on
# water_heater_1 washing_machine_1 higher_than
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Iteratively generate tokens
generated_ids = input_ids
for _ in range(1):  # Generate up to 4 iterations
    generated_ids = model.generate(
        generated_ids,
        max_new_tokens=8,
        do_sample=True,
        temperature=1,
        eos_token_id=tokenizer.eos_token_id#tokenizer.encode("GRAPH_END")[0]
    )

# Decode the final sequence and generate the scene graph
token_sequence = tokenizer.decode(generated_ids[0])
print(token_sequence)
# draw_scene_graph_and_save(token_sequence, "scene_graph.png")

