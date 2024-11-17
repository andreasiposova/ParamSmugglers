import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity


def extract_weights(module):
    """
    Extract weights from a PyTorch module and convert to numpy array.

    Args:
        module: PyTorch module (nn.Linear, nn.Conv2d, etc.)

    Returns:
        numpy.ndarray: Weight matrix as numpy array
    """
    if hasattr(module, 'weight'):
        return module.weight.detach().cpu().numpy()
    return None


def measure_parameter_changes(original_module, decorrelated_module):
    """
    Measure various differences between original and decorrelated parameters.

    Args:
        original_module: Original PyTorch module
        decorrelated_module: Decorrelated PyTorch module

    Returns:
        dict: Dictionary containing different metrics
    """
    # Extract weights from modules
    orig = extract_weights(original_module)
    decor = extract_weights(decorrelated_module)

    if orig is None or decor is None:
        return None

    metrics = {}

    # 1. Frobenius norm of the difference
    diff_norm = np.linalg.norm(orig - decor)
    metrics['frobenius_diff'] = float(diff_norm)

    # 2. Relative change (normalized by original norm)
    orig_norm = np.linalg.norm(orig)
    metrics['relative_change'] = float(diff_norm / orig_norm) if orig_norm != 0 else 0.0

    # 3. Average absolute change per parameter
    metrics['mean_abs_change'] = float(np.mean(np.abs(orig - decor)))

    # 4. Maximum absolute change
    metrics['max_abs_change'] = float(np.max(np.abs(orig - decor)))

    # 5. Cosine similarity between flattened matrices
    orig_flat = orig.reshape(-1)
    decor_flat = decor.reshape(-1)
    cos_sim = float(cosine_similarity(orig_flat.reshape(1, -1),
                                      decor_flat.reshape(1, -1))[0, 0])
    metrics['cosine_similarity'] = cos_sim

    # 6. Distribution changes
    try:
        w_distance = wasserstein_distance(orig_flat, decor_flat)
        metrics['wasserstein_distance'] = float(w_distance)
    except Exception:
        metrics['wasserstein_distance'] = float('nan')


    # Add layer size information
    metrics['num_parameters'] = orig.size

    return metrics


def aggregate_metrics(layer_metrics):
    """
    Aggregate metrics across all layers.

    Args:
        layer_metrics: Dictionary of layer names to their metrics

    Returns:
        dict: Aggregated metrics across all layers
    """
    if not layer_metrics:
        return {}

    aggregated = {}
    total_params = 0

    # Get all metric names from the first layer
    first_layer = next(iter(layer_metrics.values()))
    metric_names = [name for name in first_layer.keys() if name != 'num_parameters']

    # Initialize aggregates
    for metric in metric_names:
        aggregated[f'{metric}_mean'] = 0.0
        aggregated[f'{metric}_weighted_mean'] = 0.0
        aggregated[f'{metric}_min'] = float('inf')
        aggregated[f'{metric}_max'] = float('-inf')

    # Collect stats across layers
    for layer_name, metrics in layer_metrics.items():
        layer_size = metrics['num_parameters']
        total_params += layer_size

        for metric in metric_names:
            if metric in metrics:
                value = metrics[metric]
                # Update min/max
                aggregated[f'{metric}_min'] = min(aggregated[f'{metric}_min'], value)
                aggregated[f'{metric}_max'] = max(aggregated[f'{metric}_max'], value)
                # Accumulate for means
                aggregated[f'{metric}_mean'] += value
                aggregated[f'{metric}_weighted_mean'] += value * layer_size

    # Compute final means
    num_layers = len(layer_metrics)
    for metric in metric_names:
        # Simple mean across layers
        aggregated[f'{metric}_mean'] /= num_layers
        # Weighted mean by layer size
        aggregated[f'{metric}_weighted_mean'] /= total_params

    # Add layer count and total parameters
    aggregated['total_layers'] = num_layers
    aggregated['total_parameters'] = total_params

    return aggregated


def analyze_decorrelation_effect(model, modified_model, layer_types=(torch.nn.Linear, torch.nn.Conv2d)):
    """
    Analyze the effect of decorrelation across all layers of a model.

    Args:
        model: Original PyTorch model
        modified_model: Decorrelated PyTorch model
        layer_types: Tuple of PyTorch layer types to analyze (default: Linear and Conv2d layers)

    Returns:
        tuple: (per_layer_metrics, aggregated_metrics)
    """
    per_layer_metrics = {}

    for (name, orig_module), (_, mod_module) in zip(model.named_modules(),
                                                    modified_model.named_modules()):
        if isinstance(orig_module, layer_types):
            metrics = measure_parameter_changes(orig_module, mod_module)
            if metrics is not None:
                per_layer_metrics[name] = metrics

    # Compute aggregated metrics
    aggregated_metrics = aggregate_metrics(per_layer_metrics)

    return per_layer_metrics, aggregated_metrics