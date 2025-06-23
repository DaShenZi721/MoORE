import copy
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from transformers.activations import ACT2FN

from moe_peft.common import LoraConfig

available_routing_strategies = ["moore"]


@dataclass
class MoOREConfig(LoraConfig):
    # expert lora
    expert_config_: LoraConfig = None
    routing_strategy_: str = None
    num_experts_: int = None
    act_fn_: Optional[Union[str, torch.nn.Module]] = None
    ffn_dropout_: float = None
    task_embedding_dim_: int = None

    def check(self) -> "MoOREConfig":
        super().check()
        if self.expert_config_ is not None:
            self.expert_config_.check()
        assert (
            isinstance(self.routing_strategy_, str)
            and self.routing_strategy_ in available_routing_strategies
        )
        assert isinstance(self.num_experts_, int) and self.num_experts_ > 0
        assert self.act_fn_ is None or (
            isinstance(self.act_fn_, str) and self.act_fn_ in ACT2FN
        )

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "MoOREConfig":
        lora_config = MoOREConfig(**LoraConfig.from_config(config).__dict__)
        if "expert_lora" in config:
            expert_config = copy.deepcopy(config)
            expert_config.update(config["expert_lora"])
            lora_config.expert_config_ = LoraConfig().from_config(expert_config)
        lora_config.routing_strategy_ = config["routing_strategy"]
        lora_config.num_experts_ = config["num_experts"]
        # silu for mixtral or gelu_new for switch transformers
        # left blank to automatically use the original act_fn of FFN
        lora_config.act_fn_ = config.get("act_fn", None)
        lora_config.task_embedding_dim_ = config.get("task_embedding_dim", 128)

        return lora_config

    def export(self) -> Dict[str, any]:
        config = super().export()
        config["peft_type"] = "MOORE"
        if self.expert_config_ is not None:
            expert_config = self.expert_config_.export()
            expert_config.pop("peft_type")
            expert_config.pop("target_modules")
            config["expert_lora"] = expert_config
        config["routing_strategy"] = self.routing_strategy_
        config["num_experts"] = self.num_experts_
        if self.act_fn_ is not None and isinstance(self.act_fn_, str):
            config["act_fn"] = self.act_fn_

        return config

    def expert_config(self, expert_idx: int) -> LoraConfig:
        if self.expert_config_ is None:
            config = copy.deepcopy(super())
        else:
            config = copy.deepcopy(self.expert_config_)
        config.adapter_name = f"moe.{self.adapter_name}.experts.{expert_idx}"
        return config
