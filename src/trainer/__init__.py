import torch
import importlib
from typing import Dict, Any
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from trainer.base import FinetuneTrainer
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import NPO
from trainer.unlearn.dpo import DPO
from trainer.unlearn.simnpo import SimNPO
from trainer.unlearn.rmu import RMU
from trainer.unlearn.undial import UNDIAL
from trainer.unlearn.ceu import CEU
from trainer.unlearn.satimp import SatImp
from trainer.unlearn.wga import WGA
from trainer.unlearn.pdu import PDU


import logging

logger = logging.getLogger(__name__)

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def _patch_accelerate_bnb_deepspeed_mapping() -> None:
    """Work around broken bitsandbytes installs (e.g., missing triton.ops).

    accelerate's DeepSpeed path imports bitsandbytes unconditionally while mapping
    optimizers, even when the active optimizer is not a bnb optimizer. If the local
    bitsandbytes install is present but broken, training fails before the first step.
    """
    try:
        import accelerate.utils.deepspeed as acc_ds
        import accelerate.accelerator as acc_accel
    except Exception:
        return

    original = getattr(acc_ds, "map_pytorch_optim_to_deepspeed", None)
    if original is None or getattr(original, "_harmul_safe_bnb_patch", False):
        return

    def _safe_map_pytorch_optim_to_deepspeed(optimizer, *args, **kwargs):
        try:
            return original(optimizer, *args, **kwargs)
        except ModuleNotFoundError as e:
            if e.name != "triton.ops":
                raise
            logger.warning(
                "Skipping accelerate bitsandbytes optimizer mapping because "
                "bitsandbytes is broken (missing triton.ops)."
            )
            return optimizer

    _safe_map_pytorch_optim_to_deepspeed._harmul_safe_bnb_patch = True
    acc_ds.map_pytorch_optim_to_deepspeed = _safe_map_pytorch_optim_to_deepspeed
    if hasattr(acc_accel, "map_pytorch_optim_to_deepspeed"):
        acc_accel.map_pytorch_optim_to_deepspeed = _safe_map_pytorch_optim_to_deepspeed


def _bitsandbytes_usable() -> bool:
    try:
        importlib.import_module("bitsandbytes")
    except Exception as e:
        logger.warning(
            "bitsandbytes is present but unusable (%s: %s).", type(e).__name__, e
        )
        return False
    return True


def load_trainer_args(trainer_args: DictConfig, dataset):
    trainer_args = dict(trainer_args)
    warmup_epochs = trainer_args.pop("warmup_epochs", None)
    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count()
        dataset_len = len(dataset)
        trainer_args["warmup_steps"] = int(
            (warmup_epochs * dataset_len)
            // (batch_size * grad_accum_steps * num_devices)
        )

    optim_name = str(trainer_args.get("optim", ""))
    uses_bnb_optim = "paged_" in optim_name or "8bit" in optim_name or "bnb" in optim_name
    bnb_usable = _bitsandbytes_usable()
    if not bnb_usable:
        _patch_accelerate_bnb_deepspeed_mapping()
    if uses_bnb_optim and not bnb_usable:
        logger.warning(
            "bitsandbytes optimizer '%s' requested but bitsandbytes is unavailable; "
            "falling back to 'adamw_torch'.",
            optim_name,
        )
        trainer_args["optim"] = "adamw_torch"

    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    evaluators=None,
    template_args=None,
):
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get("method_args", {})
    trainer_args = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = trainer_cfg.get("handler")
    assert trainer_handler_name is not None, ValueError(
        f"{trainer_handler_name} handler not set"
    )
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_handler_name} not implemented or not registered"
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        **method_args,
    )
    logger.info(
        f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}"
    )
    return trainer, trainer_args


# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Register Unlearning Trainer
_register_trainer(GradAscent)
_register_trainer(GradDiff)
_register_trainer(NPO)
_register_trainer(DPO)
_register_trainer(SimNPO)
_register_trainer(RMU)
_register_trainer(UNDIAL)
_register_trainer(CEU)
_register_trainer(SatImp)
_register_trainer(WGA)
_register_trainer(PDU)
