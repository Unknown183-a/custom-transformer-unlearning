from trainer.unlearn.npo import NPO
from trainer.utils import compute_kl_divergence


class CustomV1(NPO):
    """
    Custom V1:
    NPO forgetting + retain NLL + optional retain KL regularization.
    """

    def __init__(self, kl_alpha=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_alpha = kl_alpha

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        # NPO forgetting objective
        forget_loss, forget_outputs = self._compute_npo(
            model, forget_inputs
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

        # Retain NLL
        retain_outputs = model(**retain_inputs)
        retain_nll = retain_outputs.loss

        # Retain KL regularization
        if self.kl_alpha > 0:
            retain_kl, _ = compute_kl_divergence(
                model, self.ref_model, retain_inputs
            )
        else:
            retain_kl = retain_nll.new_zeros(())

        loss = (
            self.gamma * forget_loss
            + self.alpha * retain_nll
            + self.kl_alpha * retain_kl
        )

        return (loss, forget_outputs) if return_outputs else loss

    def _compute_npo(self, model, forget_inputs):
        from trainer.utils import compute_dpo_loss

        loss, outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )
        return loss, outputs[1]
