import torch
import torch.nn as nn
import torch.nn.functional as F

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

class LoraLayer:
    def __init__(self,
                  r: int, 
                  lora_alpha: int, 
                  lora_dropout: float,
                  adapter_name: str,
                  **kwargs):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.scaling = self.lora_alpha / self.r

        self.adapter_name = adapter_name
        self.kwargs = kwargs

class LoraLinear(nn.Linear, LoraLayer):
    def __init__(self, 
                 adapter_name: str,
                 in_features: int,
                 out_features: int,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.0,
                 fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)

                 # pet parameters
                 **kwargs,):
        nn.Linear.__init__(self, in_features, out_features)
        LoraLayer.__init__(self, r, lora_alpha, lora_dropout, adapter_name, **kwargs)

        if r > 0:
            self.lora_A = nn.Linear(self.in_features, self.out_features, bias=False)
            self.lora_B = nn.Linear(self.out_features, self.in_features, bias=False)
        # freeze PLM weights
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
    
    def forward(self, x: torch.Tensor):        
        # Shared GeMM 1
        previous_dtype = x.dtype
        # result = x

        # no quantization support yet
        # x = x.to(self.lora_A.weight.dtype)

        result = (
            self.lora_B(
                self.lora_A(self.lora_dropout(x))
            )
            * self.scaling
        )

        # result = result.to(previous_dtype)
        return result

class SharedLoraLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                **kwargs):

        super().__init__(in_features, out_features)
        self.loras = nn.ModuleDict()
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout):
        self.loras.update(nn.ModuleDict({adapter_name: LoraLinear(adapter_name, self.in_features, self.out_features, r, lora_alpha, lora_dropout, **self.kwargs)}))
        # if init_lora_weights:
        #     self.reset_lora_parameters(adapter_name)

        self.to(self.weight.device)

    

    def forward(self, x: torch.Tensor, adapter_names: int):
        result = F.linear(x, self.weight, self.bias)
        # split the tensor x based on the logical indexing from the task_ids list
        for i, adapter_name in enumerate(adapter_names):
            result[i] += self.loras[adapter_name](x[i])
        
        return result

