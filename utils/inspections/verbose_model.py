from torch import Tensor, nn
from typing import Union


class VerboseModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

        for layer_name, layer in self.model.named_children():
            layer.__name__ = layer_name
            layer.register_forward_hook(lambda layer, input, output: print(
                f"{layer.__name__}, output_shape:{output.shape} output: {self.norm_(output):.2e}"))
            layer.register_backward_hook(lambda layer, grad_input, grad_output: print(
                f"{layer.__name__} grad_input: {self.format_(self.norm_(grad_input[0]))} grad_output: {self.format_(self.norm_(grad_output[0]))}"))

    @staticmethod
    def norm_(x: Tensor) -> Union[None, Tensor]:
        return x.norm() if x is not None else None

    @staticmethod
    def format_(x: Tensor) -> Union[None, str]:
        return f"{x:.2e}" if x is not None else None

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
