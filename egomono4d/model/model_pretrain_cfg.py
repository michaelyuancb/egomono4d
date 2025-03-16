from dataclasses import dataclass

@dataclass
class ModelWrapperPretrainCfg:
    lr: float = 5e-5
    cache_track: bool = ""

