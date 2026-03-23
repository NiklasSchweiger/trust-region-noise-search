"""DistributionProperty for sampling molecular property targets from dataset."""
import torch
from torch.distributions.categorical import Categorical
from typing import Dict, List, Any, Optional


class DistributionProperty:
    """Distribution over molecular properties conditioned on number of nodes.
    
    This class builds histograms of property values from the training dataset,
    conditioned on the number of nodes. It can then sample normalized property
    targets for optimization.
    """
    
    def __init__(
        self,
        dataloader,
        properties: List[str],
        num_bins: int = 1000,
        normalizer: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(
                dataloader.dataset.data['num_atoms'],
                dataloader.dataset.data[prop],
                self.distributions[prop]
            )
        
        self.normalizer = normalizer

    def set_normalizer(self, normalizer: Dict[str, Dict[str, float]]) -> None:
        """Set the normalizer (mean/MAD) for each property."""
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        """Create probability distribution for each number of nodes."""
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        """Create probability distribution for a given number of nodes."""
        n_bins = self.num_bins
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            # Handle edge case due to numerical precision
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        
        probs = histogram / torch.sum(histogram)
        probs = Categorical(probs)
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor: torch.Tensor, prop: str) -> torch.Tensor:
        """Normalize property value using mean and MAD."""
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes: int = 19) -> torch.Tensor:
        """Sample normalized property values for a given number of nodes."""
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample: torch.Tensor) -> torch.Tensor:
        """Sample normalized property values for a batch of node counts."""
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins) -> torch.Tensor:
        """Convert histogram index back to property value."""
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


def compute_mean_mad_from_dataloader(dataloader, properties: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute mean and MAD (median absolute deviation) for properties.
    
    Args:
        dataloader: PyTorch dataloader with dataset.data containing properties
        properties: List of property names to compute statistics for
        
    Returns:
        Dictionary mapping property names to {'mean': float, 'mad': float}
    """
    property_norms = {}
    for property_key in properties:
        values = dataloader.dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {
            'mean': mean.item(),
            'mad': mad.item()
        }
    return property_norms

