import torch
import numpy as np
from torchdiffeq import odeint
from torch.distributions.categorical import Categorical
from torch import nn

from .egnn import EGNN


def T(t):
    beta_min = 0.1
    beta_max = 20
    return 0.5 * (beta_max - beta_min) * t ** 2 + beta_min * t


def T_hat(t):
    beta_min = 0.1
    beta_max = 20
    return (beta_max - beta_min) * t + beta_min


def remove_mean_with_mask(x, node_mask):
    N = node_mask.sum(1, keepdims=True)
    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked


class Cnflows(torch.nn.Module):
    def __init__(
        self,
        dynamics,
        in_node_nf: int,
        n_dims: int,
        timesteps: int = 10000,
        parametrization="eps",
        time_embed=False,
        noise_schedule="learned",
        noise_precision=1e-4,
        loss_type="ot",
        norm_values=(1.0, 1.0, 1.0),
        norm_biases=(None, 0.0, 0.0),
        include_charges=True,
        discrete_path="OT_path",
        cat_loss="l2",
        cat_loss_step=-1,
        on_hold_batch=-1,
        sampling_method="vanilla",
        weighted_methods="jump",
        ode_method="dopri5",
        without_cat_loss=False,
        angle_penalty=False,
    ):
        super().__init__()

        self.set_odeint(method=ode_method)
        self.loss_type = loss_type
        self.include_charges = include_charges
        self._eps = 0.0
        self.discrete_path = discrete_path
        self.ode_method = ode_method

        self.cat_loss = cat_loss
        self.cat_loss_step = cat_loss_step
        self.on_hold_batch = on_hold_batch
        self.sampling_method = sampling_method
        self.weighted_methods = weighted_methods
        self.without_cat_loss = without_cat_loss
        self.angle_penalty = angle_penalty

        self.dynamics = dynamics
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges
        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.time_embed = time_embed
        self.register_buffer("buffer", torch.zeros(1))

    def set_odeint(self, method="dopri5", rtol=1e-4, atol=1e-4):
        self.method = method
        self._atol = atol
        self._rtol = rtol
        self._atol_test = 1e-7
        self._rtol_test = 1e-7

    def phi(self, t, x, node_mask, edge_mask, context):
        if self.time_embed:
            t = self.frequencies * t[..., None]
            t = torch.cat((t.cos(), t.sin()), dim=-1)
            t = t.expand(*x.shape[:-1], -1)
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)
        return net_out

    def subspace_dimensionality(self, node_mask):
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]
        if self.include_charges:
            h_int = h_int * node_mask
        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):
        x, h_cat = (
            z[:, :, 0: self.n_dims],
            z[:, :, self.n_dims: self.n_dims + self.num_classes],
        )
        h_int = z[:, :, self.n_dims + self.num_classes: self.n_dims + self.num_classes + 1]
        assert h_int.size(2) == self.include_charges
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def sample_p_xh_given_z0(self, dequantizer, z0, node_mask):
        x = z0[:, :, : self.n_dims]
        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(
            x, z0[:, :, self.n_dims: self.n_dims + self.num_classes], h_int, node_mask
        )
        tensor = dequantizer.reverse({"categorical": h_cat, "integer": h_int})
        one_hot, charges = tensor["categorical"], tensor["integer"]
        h = {"integer": charges, "categorical": one_hot}
        return x, h

    def decode(self, z, node_mask, edge_mask, context):
        def wrapper(t, x):
            dx = self.phi(t, x, node_mask, edge_mask, context)
            if self.cat_loss_step > 0:
                if t > self.cat_loss_step:
                    dx[:, :, self.n_dims: -1] = 0
                else:
                    dx[:, :, self.n_dims: -1] = dx[:, :, self.n_dims: -1] / (
                        self.cat_loss_step
                    )
            if self.discrete_path == "VP_path":
                M_para = (-0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)).unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path":
                M_para = (-0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)).unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims:] = dx[:, :, self.n_dims:] * M_para
            return dx

        t_list = torch.linspace(1, 0, 100, dtype=torch.float, device=z.device)
        return odeint(wrapper, z, t_list, method=self.method, rtol=self._rtol, atol=self._atol)

    def forward(self, t, x):
        x = x * self.node_mask
        dx = self.phi(t, x, self.node_mask, self.edge_mask, self.context)
        if self.cat_loss_step > 0:
            if t > self.cat_loss_step:
                dx[:, :, self.n_dims: -1] = 0
            else:
                dx[:, :, self.n_dims: -1] = dx[:, :, self.n_dims: -1] / (self.cat_loss_step)
        if self.discrete_path == "VP_path":
            M_para = (-0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)).unsqueeze(-1)[:, None, None]
            dx = dx * M_para
        elif self.discrete_path == "HB_path":
            M_para = (-0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)).unsqueeze(-1)[:, None, None]
            coeff = torch.ones_like(dx)
            coeff[:, :, self.n_dims:] = M_para
            dx = dx * coeff
        return dx

    def set_conditional_param(self, node_mask=None, edge_mask=None, context=None):
        if isinstance(node_mask, torch.Tensor):
            if hasattr(self, 'node_mask'):
                del self.node_mask
            self.register_buffer('node_mask', node_mask)
        else:
            self.node_mask = node_mask
        if isinstance(node_mask, torch.Tensor):
            if hasattr(self, 'edge_mask'):
                del self.edge_mask
            self.register_buffer('edge_mask', edge_mask)
        else:
            self.edge_mask = edge_mask
        self.context = context

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    @torch.no_grad()
    def sample(self, dequantizer, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        if fix_noise:
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)
        z_ = self.decode(z, node_mask, edge_mask, context)[-1]

        if self.sampling_method == "gradient":
            init = z_[:, :, self.n_dims: -1]
            categorical_steps = np.linspace(0.05, 0, 20)
            for i_ in categorical_steps:
                gradient = self.phi(torch.tensor([i_]), z_, node_mask, edge_mask, context)
                init = init + gradient[:, :, self.n_dims: -1] * (0.05 / 20)
            z_[:, :, self.n_dims: -1] = init
        elif self.sampling_method == "vanilla":
            pass
        else:
            raise NotImplementedError
        x, h = self.sample_p_xh_given_z0(dequantizer, z_, node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            x = remove_mean_with_mask(x, node_mask)
        return x, h


class EGNN_dynamics_QM9(nn.Module):
    def __init__(
        self,
        in_node_nf,
        context_node_nf,
        n_dims,
        hidden_nf=64,
        device='cpu',
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        tanh=False,
        mode='egnn_dynamics',
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs = xh.size(0)
        n_nodes, dims = xh.size(-2), xh.size(-1)
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
        vel = (x_final - x) * node_mask

        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]
        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)
        vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))
        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [
                    torch.LongTensor(rows).to(device),
                    torch.LongTensor(cols).to(device)
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)
        self.prob = torch.from_numpy(prob).float()
        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

