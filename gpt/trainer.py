from transformers import Trainer
from visualize_sg import draw_scene_graph, draw_scene_graph_and_save
import wandb

class CustomTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, model=None, **kwargs):
        super().__init__(*args, processing_class=tokenizer, model=model, **kwargs)
        self.tokenizer = tokenizer
        self.model = model

    # def evaluation_loop(
    #     self,
    #     dataloader,
    #     description,
    #     prediction_loss_only=None,
    #     ignore_keys=None,
    #     metric_key_prefix="eval",
    # ):
    #     # Call superclass method to get the eval outputs
    #     eval_output = super().evaluation_loop(
    #         dataloader,
    #         description,
    #         prediction_loss_only,
    #         ignore_keys,
    #         metric_key_prefix,
    #     )

    #     # Ensure WandB is initialized
    #     if wandb.run is not None:
    #         input_text = "floor_1 shelf_1 support floor_1 bed_1 support"
    #         input_ids = self.tokenizer.encode(input_text, return_tensors="pt").cuda()

    #         # Iteratively generate tokens
    #         generated_ids = input_ids
    #         for _ in range(4):  # Generate up to 4 iterations
    #             generated_ids = self.model.generate(
    #                 generated_ids,
    #                 max_new_tokens=10,
    #                 do_sample=True,
    #                 temperature=0.75,
    #             )

    #         # Decode the final sequence and generate the scene graph
    #         token_sequence = self.tokenizer.decode(generated_ids[0])
    #         try:
    #             # sg = draw_scene_graph(token_sequence)
    #             draw_scene_graph_and_save(token_sequence, "scene_graph.png")

    #             # Log the generated scene graph to WandB
    #             wandb.log({"Generated_scene_graph": wandb.Image("scene_graph.png")})

    #         except Exception as e:
    #             print(f"Error generating scene graph: {e}")
    #     return eval_output