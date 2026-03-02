"""
EGGROLL: Evolution Strategies with Low-Rank Perturbations

Implementation based on:
"Evolution Strategies at the Hyperscale" (Sarkar et al., 2025)

Key concepts:
- Low-rank perturbations: E = (1/√r) * A @ B^T where r << min(m,n)
- Memory efficient: O(r(m+n)) instead of O(mn)
- Full-rank updates: N*r >= min(m,n) ensures effective full-rank exploration
- Gaussian score approximation for faster convergence

This is a PyTorch port of the JAX EGGROLL implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import copy
from tqdm import tqdm


@dataclass
class EGGROLLConfig:
    """Configuration for EGGROLL optimizer."""
    
    # Core hyperparameters
    population_size: int = 64          # N: number of perturbations per step
    rank: int = 4                      # r: rank of perturbation matrices
    sigma: float = 0.01                # σ: noise scale for perturbations
    learning_rate: float = 0.1         # α: step size for parameter updates
    
    # Training settings
    num_generations: int = 100         # Number of evolution steps
    batch_size: int = 32               # Batch size for fitness evaluation
    
    # Advanced settings
    use_antithetic: bool = True        # Use mirrored sampling (±E)
    normalize_fitness: bool = True     # Normalize fitness scores (z-score)
    rank_transform: bool = False       # Use rank-based fitness shaping (more stable)
    centered_rank: bool = True         # Center ranks around 0 (for rank_transform)
    weight_decay: float = 0.0          # L2 regularization
    lr_decay: float = 1.0              # Learning rate decay per generation
    sigma_decay: float = 1.0           # Sigma decay per generation
    
    # Constraint: N*r >= min(m,n) for full-rank updates
    enforce_rank_constraint: bool = True
    
    # Random seed
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.population_size > 0, "Population size must be positive"
        assert self.rank > 0, "Rank must be positive"
        assert self.sigma > 0, "Sigma must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"


class LowRankPerturbation:
    """
    Low-rank perturbation for a parameter tensor.
    
    For a parameter W of shape (m, n), the perturbation is:
    E = (1/√r) * A @ B^T
    
    where A ∈ R^(m×r) and B ∈ R^(n×r)
    
    Memory: O(r(m+n)) instead of O(mn)
    """
    
    def __init__(
        self, 
        shape: Tuple[int, ...], 
        rank: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ):
        self.shape = shape
        self.rank = rank
        self.device = device
        self.dtype = dtype
        
        # Handle different tensor dimensions
        if len(shape) == 1:
            # 1D tensor (bias): treat as (m, 1)
            self.m = shape[0]
            self.n = 1
            self.is_1d = True
        elif len(shape) == 2:
            # 2D tensor (weight matrix)
            self.m, self.n = shape
            self.is_1d = False
        else:
            # Higher dimensional (conv kernels etc): flatten to 2D
            self.m = shape[0]
            self.n = int(np.prod(shape[1:]))
            self.is_1d = False
            self.original_shape = shape
        
        # Effective rank (can't exceed min dimension)
        self.effective_rank = min(rank, self.m, self.n)
        
        # Scaling factor
        self.scale = 1.0 / np.sqrt(self.effective_rank)
    
    def sample(self, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample low-rank factors A and B.
        
        Returns:
            A: Shape (m, r)
            B: Shape (n, r)
        """
        A = torch.randn(
            self.m, self.effective_rank, 
            generator=rng, device=self.device, dtype=self.dtype
        )
        B = torch.randn(
            self.n, self.effective_rank,
            generator=rng, device=self.device, dtype=self.dtype
        )
        return A, B
    
    def construct_perturbation(
        self, 
        A: torch.Tensor, 
        B: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct full perturbation E = (1/√r) * A @ B^T
        
        Args:
            A: Shape (m, r)
            B: Shape (n, r)
            
        Returns:
            E: Shape matching original parameter shape
        """
        # E = scale * A @ B^T
        E = self.scale * torch.mm(A, B.t())  # (m, n)
        
        # Reshape to original shape
        if self.is_1d:
            E = E.squeeze(-1)  # (m,)
        elif hasattr(self, 'original_shape'):
            E = E.view(self.original_shape)
        
        return E
    
    def compute_update(
        self,
        A_list: List[torch.Tensor],
        B_list: List[torch.Tensor],
        fitness_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute parameter update from weighted perturbations (vectorized).
        
        Update = (1/N) * Σ_i E_i * f_i
               = (1/N) * Σ_i (scale * A_i @ B_i^T) * f_i
        
        Uses batched operations for GPU efficiency.
        
        Args:
            A_list: List of A matrices for each perturbation
            B_list: List of B matrices for each perturbation
            fitness_scores: Fitness score for each perturbation
            
        Returns:
            Parameter update tensor
        """
        N = len(A_list)
        
        # Vectorized computation using batched matmul
        # Stack all factors: (N, m, r) and (N, n, r)
        A_stack = torch.stack(A_list)  # (N, m, r)
        B_stack = torch.stack(B_list)  # (N, n, r)
        
        # Reshape fitness for broadcasting: (N, 1, 1)
        f_stack = fitness_scores.view(N, 1, 1)
        
        # Batched matmul: (N, m, r) @ (N, r, n) -> (N, m, n)
        E_all = self.scale * torch.bmm(A_stack, B_stack.transpose(-1, -2))
        
        # Weighted sum: (N, m, n) * (N, 1, 1) -> sum -> (m, n)
        update = (E_all * f_stack).sum(dim=0) / N
        
        # Reshape to original shape
        if self.is_1d:
            update = update.squeeze(-1)
        elif hasattr(self, 'original_shape'):
            update = update.view(self.original_shape)
        
        return update


class EGGROLL:
    """
    EGGROLL Optimizer: Evolution Strategies with Low-Rank Perturbations
    
    This optimizer replaces gradient-based training with evolution strategies.
    It applies low-rank perturbations to model parameters and updates based
    on fitness scores.
    
    Usage:
        config = EGGROLLConfig(population_size=64, rank=4, sigma=0.01)
        optimizer = EGGROLL(model, config)
        
        for gen in range(num_generations):
            fitness = optimizer.step(fitness_fn, data_batch)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: EGGROLLConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize EGGROLL optimizer.
        
        Args:
            model: PyTorch model to optimize
            config: EGGROLL configuration
            device: Device to use (default: auto-detect)
        """
        self.model = model
        self.config = config
        self.device = device or next(model.parameters()).device
        
        # Set random seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        
        # Create RNG for reproducibility
        self.rng = torch.Generator(device=self.device)
        if config.seed is not None:
            self.rng.manual_seed(config.seed)
        
        # Get trainable parameters
        self.param_names = []
        self.param_shapes = {}
        self.perturbations = {}
        
        self._setup_parameters()
        
        # Current learning rate and sigma (for decay)
        self.current_lr = config.learning_rate
        self.current_sigma = config.sigma
        
        # Statistics
        self.generation = 0
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def _setup_parameters(self):
        """Setup perturbation objects for each trainable parameter."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.param_shapes[name] = param.shape
                
                # Create low-rank perturbation object
                self.perturbations[name] = LowRankPerturbation(
                    shape=param.shape,
                    rank=self.config.rank,
                    device=self.device,
                    dtype=param.dtype
                )
                
                # Check rank constraint if enabled
                if self.config.enforce_rank_constraint:
                    pert = self.perturbations[name]
                    min_dim = min(pert.m, pert.n)
                    required_N = int(np.ceil(min_dim / self.config.rank))
                    
                    if self.config.population_size * self.config.rank < min_dim:
                        print(f"Warning: Parameter '{name}' shape {param.shape}")
                        print(f"  N*r = {self.config.population_size}*{self.config.rank} = "
                              f"{self.config.population_size * self.config.rank} < min(m,n) = {min_dim}")
                        print(f"  Consider N >= {required_N} or r >= {int(np.ceil(min_dim / self.config.population_size))}")
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"EGGROLL initialized with {len(self.param_names)} parameter groups")
        print(f"Total trainable parameters: {total_params:,}")
        print(f"Population size: {self.config.population_size}, Rank: {self.config.rank}")
        print(f"Sigma: {self.config.sigma}, Learning rate: {self.config.learning_rate}")
        if self.config.rank_transform:
            print(f"Fitness shaping: rank_transform (centered={self.config.centered_rank})")
        elif self.config.normalize_fitness:
            print(f"Fitness shaping: z-score normalization")
    
    def _sample_perturbations(self) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Sample low-rank perturbation factors for all parameters.
        
        Returns:
            List of dicts mapping param_name -> (A, B) tuples
        """
        perturbation_samples = []
        
        N = self.config.population_size
        if self.config.use_antithetic:
            N = N // 2  # Will mirror each sample
        
        for _ in range(N):
            sample = {}
            for name in self.param_names:
                A, B = self.perturbations[name].sample(self.rng)
                sample[name] = (A, B)
            perturbation_samples.append(sample)
        
        return perturbation_samples
    
    def _apply_perturbation(
        self,
        param_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        sign: float = 1.0
    ):
        """
        Apply perturbation to model parameters.
        
        Args:
            param_factors: Dict mapping param_name -> (A, B)
            sign: +1 or -1 for antithetic sampling
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in param_factors:
                    A, B = param_factors[name]
                    E = self.perturbations[name].construct_perturbation(A, B)
                    # Ensure E is on the same device as param
                    E = E.to(param.device)
                    param.add_(sign * self.current_sigma * E)
    
    def _remove_perturbation(
        self,
        param_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        sign: float = 1.0
    ):
        """Remove previously applied perturbation."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in param_factors:
                    A, B = param_factors[name]
                    E = self.perturbations[name].construct_perturbation(A, B)
                    # Ensure E is on the same device as param
                    E = E.to(param.device)
                    param.add_(-sign * self.current_sigma * E)
    
    def _evaluate_fitness(
        self,
        fitness_fn: Callable,
        data: Any,
        perturbation_samples: List[Dict],
    ) -> Tuple[torch.Tensor, List[Dict], List[float]]:
        """
        Evaluate fitness for all perturbations.
        
        Args:
            fitness_fn: Function(model, data) -> fitness score (higher is better)
            data: Data batch for fitness evaluation
            perturbation_samples: List of perturbation factor dicts
            
        Returns:
            fitness_scores: Tensor of fitness values
            all_factors: Complete list of (A, B) factors (including antithetic)
            signs: List of signs (+1 or -1)
        """
        fitness_scores = []
        all_factors = []
        signs = []
        
        self.model.eval()
        
        for factors in perturbation_samples:
            # Positive perturbation
            self._apply_perturbation(factors, sign=1.0)
            
            with torch.no_grad():
                fitness_pos = fitness_fn(self.model, data)
            
            self._remove_perturbation(factors, sign=1.0)
            
            fitness_scores.append(fitness_pos)
            all_factors.append(factors)
            signs.append(1.0)
            
            # Antithetic (negative) perturbation
            if self.config.use_antithetic:
                self._apply_perturbation(factors, sign=-1.0)
                
                with torch.no_grad():
                    fitness_neg = fitness_fn(self.model, data)
                
                self._remove_perturbation(factors, sign=-1.0)
                
                fitness_scores.append(fitness_neg)
                all_factors.append(factors)
                signs.append(-1.0)
        
        return torch.tensor(fitness_scores, device=self.device), all_factors, signs
    
    def _normalize_fitness(self, fitness: torch.Tensor) -> torch.Tensor:
        """Normalize fitness scores to have zero mean and unit variance."""
        mean = fitness.mean()
        std = fitness.std()
        if std > 1e-8:
            return (fitness - mean) / std
        return fitness - mean
    
    def _rank_transform(self, fitness: torch.Tensor) -> torch.Tensor:
        """
        Rank-based fitness shaping (from paper).
        
        Converts fitness values to ranks, then normalizes.
        More robust to outliers than z-score normalization.
        """
        N = len(fitness)
        
        # Get ranks (0 to N-1)
        ranks = torch.zeros_like(fitness)
        sorted_indices = torch.argsort(fitness)
        ranks[sorted_indices] = torch.arange(N, device=fitness.device, dtype=fitness.dtype)
        
        if self.config.centered_rank:
            # Center ranks around 0: [-0.5, 0.5]
            ranks = ranks / (N - 1) - 0.5
        else:
            # Normalize to [0, 1]
            ranks = ranks / (N - 1)
        
        return ranks
    
    def _compute_updates(
        self,
        all_factors: List[Dict],
        signs: List[float],
        fitness_scores: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute parameter updates from fitness-weighted perturbations.
        
        Update rule (Eq. 8 from paper):
        μ_{t+1} = μ_t + (α/N) * Σ_i E_i * f_i
        
        Args:
            all_factors: List of (A, B) factor dicts
            signs: List of signs for each perturbation
            fitness_scores: Normalized fitness scores
            
        Returns:
            Dict mapping param_name -> update tensor
        """
        updates = {}
        N = len(all_factors)
        
        for name in self.param_names:
            # Collect A, B matrices and apply signs
            A_list = []
            B_list = []
            
            for factors, sign in zip(all_factors, signs):
                A, B = factors[name]
                # Apply sign to one of the factors
                A_list.append(sign * A)
                B_list.append(B)
            
            # Compute update
            update = self.perturbations[name].compute_update(
                A_list, B_list, fitness_scores
            )
            updates[name] = update
        
        return updates
    
    def _apply_updates(self, updates: Dict[str, torch.Tensor]):
        """Apply computed updates to model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in updates:
                    # Ensure update is on the same device as param
                    update = updates[name].to(param.device)
                    # Update: μ = μ + α * update
                    param.add_(self.current_lr * update)
                    
                    # Weight decay
                    if self.config.weight_decay > 0:
                        param.mul_(1 - self.config.weight_decay)
    
    def step(
        self,
        fitness_fn: Callable,
        data: Any,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Perform one EGGROLL update step.
        
        Args:
            fitness_fn: Function(model, data) -> fitness (higher is better)
            data: Data for fitness evaluation
            verbose: Print progress
            
        Returns:
            Dict with statistics (mean_fitness, max_fitness, etc.)
        """
        # Sample perturbations
        perturbation_samples = self._sample_perturbations()
        
        # Evaluate fitness
        fitness_scores, all_factors, signs = self._evaluate_fitness(
            fitness_fn, data, perturbation_samples
        )
        
        # Record statistics before normalization
        mean_fitness = fitness_scores.mean().item()
        max_fitness = fitness_scores.max().item()
        min_fitness = fitness_scores.min().item()
        std_fitness = fitness_scores.std().item()
        
        # Update best fitness
        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
        
        # Shape fitness (rank_transform takes priority over normalize_fitness)
        if self.config.rank_transform:
            fitness_scores = self._rank_transform(fitness_scores)
        elif self.config.normalize_fitness:
            fitness_scores = self._normalize_fitness(fitness_scores)
        
        # Compute updates
        updates = self._compute_updates(all_factors, signs, fitness_scores)
        
        # Apply updates
        self._apply_updates(updates)
        
        # Decay learning rate and sigma
        self.current_lr *= self.config.lr_decay
        self.current_sigma *= self.config.sigma_decay
        
        # Update generation counter
        self.generation += 1
        
        # Record history
        stats = {
            'generation': self.generation,
            'mean_fitness': mean_fitness,
            'max_fitness': max_fitness,
            'min_fitness': min_fitness,
            'std_fitness': std_fitness,
            'best_fitness': self.best_fitness,
            'learning_rate': self.current_lr,
            'sigma': self.current_sigma
        }
        self.fitness_history.append(stats)
        
        if verbose:
            print(f"Gen {self.generation}: "
                  f"mean={mean_fitness:.4f}, max={max_fitness:.4f}, "
                  f"best={self.best_fitness:.4f}")
        
        return stats
    
    def train(
        self,
        fitness_fn: Callable,
        data_loader: Any,
        num_generations: Optional[int] = None,
        callback: Optional[Callable] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Run full EGGROLL training loop.
        
        Args:
            fitness_fn: Fitness function
            data_loader: Iterable of data batches (will cycle)
            num_generations: Override config.num_generations
            callback: Called after each generation with (gen, stats, model)
            verbose: Print progress
            
        Returns:
            List of stats dicts for each generation
        """
        num_gens = num_generations or self.config.num_generations
        
        # Create iterator
        if hasattr(data_loader, '__iter__'):
            data_iter = iter(data_loader)
        else:
            data_iter = iter([data_loader])
        
        history = []
        
        pbar = tqdm(range(num_gens), desc="EGGROLL Training", disable=not verbose)
        
        for gen in pbar:
            # Get next batch (cycle if needed)
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                data = next(data_iter)
            
            # Perform update step
            stats = self.step(fitness_fn, data, verbose=False)
            history.append(stats)
            
            # Update progress bar
            pbar.set_postfix({
                'mean': f"{stats['mean_fitness']:.4f}",
                'best': f"{stats['best_fitness']:.4f}"
            })
            
            # Callback
            if callback is not None:
                callback(gen, stats, self.model)
        
        return history
    
    def get_best_model(self) -> nn.Module:
        """Return current model (which should be the best after training)."""
        return self.model
    
    def state_dict(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'current_lr': self.current_lr,
            'current_sigma': self.current_sigma,
            'fitness_history': self.fitness_history,
            'rng_state': self.rng.get_state()
        }
    
    def load_state_dict(self, state: Dict):
        """Load optimizer state from checkpoint."""
        self.generation = state['generation']
        self.best_fitness = state['best_fitness']
        self.current_lr = state['current_lr']
        self.current_sigma = state['current_sigma']
        self.fitness_history = state['fitness_history']
        self.rng.set_state(state['rng_state'])


def compute_optimal_hyperparameters(
    model: nn.Module,
    target_full_rank: bool = True
) -> Dict[str, Any]:
    """
    Compute optimal EGGROLL hyperparameters for a model.
    
    Based on the constraint: N * r >= min(m, n) for full-rank updates.
    
    Args:
        model: PyTorch model
        target_full_rank: If True, ensure N*r >= min(m,n) for all layers
        
    Returns:
        Dict with suggested hyperparameters
    """
    max_min_dim = 0
    param_info = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape = param.shape
            if len(shape) >= 2:
                m, n = shape[0], int(np.prod(shape[1:]))
            else:
                m, n = shape[0], 1
            
            min_dim = min(m, n)
            max_min_dim = max(max_min_dim, min_dim)
            param_info.append({
                'name': name,
                'shape': shape,
                'm': m,
                'n': n,
                'min_dim': min_dim
            })
    
    # Suggest hyperparameters
    suggestions = {}
    
    if target_full_rank:
        # Option 1: r=1, N=max_min_dim
        suggestions['option_1'] = {
            'rank': 1,
            'population_size': max_min_dim,
            'note': 'Minimum rank, large population'
        }
        
        # Option 2: r=4, N=ceil(max_min_dim/4)
        suggestions['option_2'] = {
            'rank': 4,
            'population_size': int(np.ceil(max_min_dim / 4)),
            'note': 'Balanced rank and population'
        }
        
        # Option 3: r=16, N=ceil(max_min_dim/16)
        suggestions['option_3'] = {
            'rank': 16,
            'population_size': int(np.ceil(max_min_dim / 16)),
            'note': 'Higher rank, smaller population'
        }
    else:
        suggestions['default'] = {
            'rank': 4,
            'population_size': 64,
            'note': 'Default settings (may not achieve full-rank updates)'
        }
    
    return {
        'max_min_dim': max_min_dim,
        'param_info': param_info,
        'suggestions': suggestions
    }
