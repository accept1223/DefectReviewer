from transformers import Trainer
import torch

class CausalLMTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """
        Custom prediction step for generative evaluation using model.generate().
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        generation_kwargs = {
            "max_new_tokens": 128,
            "num_beams": 2,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        # Generate predictions
        generated_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

        if generated_tokens.shape[-1] < input_ids.shape[-1]:
            pad_length = input_ids.shape[-1] - generated_tokens.shape[-1]
            generated_tokens = torch.cat(
                [
                    generated_tokens,
                    torch.full(
                        (generated_tokens.shape[0], pad_length),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long,
                        device=generated_tokens.device,
                    ),
                ],
                dim=-1,
            )

        with torch.no_grad():
            loss = None
            if "labels" in inputs:
                loss = self.compute_loss(model, inputs, return_outputs=False)

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels", None)
        return (loss, generated_tokens, labels)
