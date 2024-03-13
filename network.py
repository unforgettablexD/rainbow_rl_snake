import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """NoisyNet layer with factorized Gaussian noise."""
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_features ** 0.5)

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / self.out_features ** 0.5)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class Network(nn.Module):
    def __init__(self, in_dim, out_dim, atom_size, support):
        super(Network, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.atom_size = atom_size


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.support = support.to(self.device)

        # Create the support tensor and immediately send it to the determined device
        #self.support = torch.linspace(v_min, v_max, self.atom_size).to(self.device)

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        self.value_layer = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, atom_size),
        )

        self.advantage_layer = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, out_dim * atom_size),
        )

    def forward(self, x):
        x = self.feature_layer(x)
        value = self.value_layer(x).view(-1, 1, self.atom_size)
        advantage = self.advantage_layer(x).view(-1, self.out_dim, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        q_values = torch.sum(dist * self.support, dim=2)

        return q_values

    def dist(self, x):
        x = self.feature_layer(x)
        value = self.value_layer(x).view(-1, 1, self.atom_size)
        advantage = self.advantage_layer(x).view(-1, self.out_dim, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        return dist
