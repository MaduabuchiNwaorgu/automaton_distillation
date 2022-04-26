import abc
import math

import torch
import torch.nn as nn


class MultiAgentLayer(nn.Module, abc.ABC):
    """
    A pytorch module that can handle multiple agents.
    This means that we have multiple different sets of weights, and a parameter in the forward function to select
    which weights. Resizing adds or removes these weights
    """

    @abc.abstractmethod
    def update_num_agents(self, num_agents: int):
        """Add or remove agents from the system"""
        pass

    # Commented out because of annoying inheritance typing rules, but imagine it's like this
    """
    @abc.abstractmethod
    def forward(self, which_agents, *other_params):
        pass
    """


class MultiAgentLinearLayer(MultiAgentLayer):
    """
    A multi-agent layer that performs a linear transformation
    """

    def __init__(self, in_features, out_features, init_num_agents):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_agents = init_num_agents

        self.weight = nn.Parameter(torch.Tensor(init_num_agents, out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(init_num_agents, out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def update_num_agents(self, num_agents: int):
        old_weight = self.weight
        old_bias = self.bias
        old_num_agents = self.num_agents

        self.weight = nn.Parameter(
            torch.zeros((num_agents, self.out_features, self.in_features), device=self.weight.device))
        self.bias = nn.Parameter(torch.zeros((num_agents, self.out_features), device=self.bias.device))
        self.num_agents = num_agents

        self.reset_parameters()

        agents_to_copy = min(num_agents, old_num_agents)

        self.weight.requires_grad = False  # Because of the in-place operations
        self.bias.requires_grad = False

        self.weight[:agents_to_copy] = old_weight[:agents_to_copy].detach()
        self.bias[:agents_to_copy] = old_bias[:agents_to_copy].detach()

        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def forward(self, which_agents: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.weight[which_agents], input.unsqueeze(2)).squeeze(2) + self.bias[which_agents]

    def extra_repr(self) -> str:
        return 'num_linears={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linears, self.in_features, self.out_features, self.bias is not None
        )


class MultiAgentLayerWrapper(MultiAgentLayer):
    """
    Wraps a normal layer into a multi-agent layer, deliberately ignoring which agent a given input is for
    """

    def __init__(self, to_wrap: nn.Module):
        super().__init__()
        self.inner = to_wrap

    def forward(self, which_agents, *others, **other_kwargs):
        return self.inner(*others, **other_kwargs)

    def update_num_agents(self, num_agents: int):
        pass


class MultiAgentLayerSequential(nn.Sequential, MultiAgentLayer):
    def __init__(self, *seq):
        nn.Sequential.__init__(self, *seq)

    # noinspection PyMethodOverriding
    def forward(self, which_agents, input):
        for module in self:
            input = module(which_agents, input)
        return input

    def update_num_agents(self, num_agents: int):
        for module in self:
            module.update_num_agents(num_agents)
