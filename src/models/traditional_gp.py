"""
Traditional Genetic Programming (GP) Module

This module implements traditional GP with:
- Discrete operator selection
- Sparse feature selection (only uses few of 512 features)
- Mutation and crossover operations
- Interpretable expressions
"""

import torch
import numpy as np
import copy
import random
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Types of nodes in GP tree."""
    OPERATOR = "operator"
    FEATURE = "feature"
    CONSTANT = "constant"


@dataclass
class GPNode:
    """A node in the GP tree."""
    node_type: NodeType
    value: Any  # operator name, feature index, or constant value
    left: Optional['GPNode'] = None
    right: Optional['GPNode'] = None
    
    def copy(self) -> 'GPNode':
        """Deep copy of node and subtree."""
        new_node = GPNode(
            node_type=self.node_type,
            value=self.value if self.node_type != NodeType.CONSTANT else float(self.value),
            left=self.left.copy() if self.left else None,
            right=self.right.copy() if self.right else None
        )
        return new_node
    
    def depth(self) -> int:
        """Compute depth of subtree."""
        if self.node_type != NodeType.OPERATOR:
            return 1
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)
    
    def size(self) -> int:
        """Count number of nodes in subtree."""
        if self.node_type != NodeType.OPERATOR:
            return 1
        left_size = self.left.size() if self.left else 0
        right_size = self.right.size() if self.right else 0
        return 1 + left_size + right_size


class TraditionalGPTree:
    """
    A traditional GP tree with discrete operators.
    
    Operators:
    - Binary: add, sub, mul, div (protected)
    - Unary: neg, abs, sqrt, square, sin, cos, exp, log, tanh
    
    Terminals:
    - Features: x_i where i ∈ [0, 511]
    - Constants: random floats in [-5, 5]
    """
    
    # Available operators
    BINARY_OPS = ['add', 'sub', 'mul', 'div']
    UNARY_OPS = ['neg', 'abs', 'sqrt', 'square', 'sin', 'cos', 'exp', 'tanh']
    ALL_OPS = BINARY_OPS + UNARY_OPS
    
    def __init__(
        self,
        input_dim: int = 512,
        max_depth: int = 4,
        root: Optional[GPNode] = None
    ):
        """
        Initialize GP tree.
        
        Args:
            input_dim: Number of input features (512 for UniMol)
            max_depth: Maximum tree depth
            root: Optional pre-built root node
        """
        self.input_dim = input_dim
        self.max_depth = max_depth
        
        if root is not None:
            self.root = root
        else:
            self.root = self._create_random_tree(max_depth)
    
    def _create_random_tree(self, max_depth: int, current_depth: int = 0) -> GPNode:
        """Create a random GP tree using grow method."""
        # Force terminal at max depth
        if current_depth >= max_depth:
            return self._create_terminal()
        
        # At shallow depths, prefer operators to build complex trees
        if current_depth < 2:
            op_prob = 0.9
        else:
            op_prob = 0.5
        
        if random.random() < op_prob:
            # Create operator node
            if random.random() < 0.6:  # Prefer binary ops
                op = random.choice(self.BINARY_OPS)
                return GPNode(
                    node_type=NodeType.OPERATOR,
                    value=op,
                    left=self._create_random_tree(max_depth, current_depth + 1),
                    right=self._create_random_tree(max_depth, current_depth + 1)
                )
            else:
                op = random.choice(self.UNARY_OPS)
                return GPNode(
                    node_type=NodeType.OPERATOR,
                    value=op,
                    left=self._create_random_tree(max_depth, current_depth + 1),
                    right=None
                )
        else:
            return self._create_terminal()
    
    def _create_terminal(self) -> GPNode:
        """Create a terminal node (feature or constant)."""
        if random.random() < 0.7:  # 70% feature, 30% constant
            # Feature node - select random feature index
            feature_idx = random.randint(0, self.input_dim - 1)
            return GPNode(node_type=NodeType.FEATURE, value=feature_idx)
        else:
            # Constant node
            constant = random.uniform(-5, 5)
            return GPNode(node_type=NodeType.CONSTANT, value=constant)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate GP tree on input.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Output tensor (batch_size,)
        """
        return self._evaluate_node(self.root, x)
    
    def _evaluate_node(self, node: GPNode, x: torch.Tensor) -> torch.Tensor:
        """Recursively evaluate a node."""
        if node.node_type == NodeType.FEATURE:
            # Return feature value
            return x[:, node.value]
        
        elif node.node_type == NodeType.CONSTANT:
            # Return constant (broadcast to batch size)
            return torch.full((x.shape[0],), node.value, dtype=x.dtype, device=x.device)
        
        elif node.node_type == NodeType.OPERATOR:
            # Evaluate children
            left_val = self._evaluate_node(node.left, x)
            
            if node.right is not None:
                right_val = self._evaluate_node(node.right, x)
            else:
                right_val = None
            
            # Apply operator
            return self._apply_operator(node.value, left_val, right_val)
    
    def _apply_operator(
        self, 
        op: str, 
        left: torch.Tensor, 
        right: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply operator with protection against numerical issues."""
        eps = 1e-6
        max_val = 100.0
        
        # Clamp inputs
        left = torch.clamp(left, -max_val, max_val)
        if right is not None:
            right = torch.clamp(right, -max_val, max_val)
        
        if op == 'add':
            result = left + right
        elif op == 'sub':
            result = left - right
        elif op == 'mul':
            result = left * right
        elif op == 'div':
            # Protected division
            safe_right = torch.where(torch.abs(right) < eps, torch.ones_like(right), right)
            result = left / safe_right
        elif op == 'neg':
            result = -left
        elif op == 'abs':
            result = torch.abs(left)
        elif op == 'sqrt':
            result = torch.sqrt(torch.abs(left) + eps)
        elif op == 'square':
            result = left * left
        elif op == 'sin':
            result = torch.sin(left)
        elif op == 'cos':
            result = torch.cos(left)
        elif op == 'exp':
            result = torch.exp(torch.clamp(left, -10, 10))
        elif op == 'tanh':
            result = torch.tanh(left)
        else:
            raise ValueError(f"Unknown operator: {op}")
        
        # Clamp output
        return torch.clamp(result, -max_val, max_val)
    
    def copy(self) -> 'TraditionalGPTree':
        """Create a deep copy of this tree."""
        new_tree = TraditionalGPTree(
            input_dim=self.input_dim,
            max_depth=self.max_depth,
            root=self.root.copy()
        )
        return new_tree
    
    def get_expression(self) -> str:
        """Get human-readable expression."""
        return self._node_to_string(self.root)
    
    def _node_to_string(self, node: GPNode) -> str:
        """Convert node to string."""
        if node.node_type == NodeType.FEATURE:
            return f"x{node.value}"
        elif node.node_type == NodeType.CONSTANT:
            return f"{node.value:.3f}"
        elif node.node_type == NodeType.OPERATOR:
            left_str = self._node_to_string(node.left)
            if node.right is not None:
                right_str = self._node_to_string(node.right)
                if node.value in ['add', 'sub', 'mul', 'div']:
                    op_symbol = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}
                    op_symbol = {v: k for k, v in op_symbol.items()}[node.value]
                    return f"({left_str} {op_symbol} {right_str})"
                else:
                    return f"{node.value}({left_str}, {right_str})"
            else:
                return f"{node.value}({left_str})"
        return "?"
    
    def get_all_nodes(self) -> List[Tuple[GPNode, Optional[GPNode], str]]:
        """Get all nodes with their parent and position (left/right)."""
        nodes = []
        self._collect_nodes(self.root, None, None, nodes)
        return nodes
    
    def _collect_nodes(
        self, 
        node: GPNode, 
        parent: Optional[GPNode], 
        position: Optional[str],
        nodes: List
    ):
        """Recursively collect all nodes."""
        nodes.append((node, parent, position))
        if node.left:
            self._collect_nodes(node.left, node, 'left', nodes)
        if node.right:
            self._collect_nodes(node.right, node, 'right', nodes)
    
    def depth(self) -> int:
        """Get tree depth."""
        return self.root.depth()
    
    def size(self) -> int:
        """Get number of nodes."""
        return self.root.size()
    
    def get_used_features(self) -> List[int]:
        """Get list of feature indices used in tree."""
        features = []
        self._collect_features(self.root, features)
        return list(set(features))
    
    def _collect_features(self, node: GPNode, features: List[int]):
        """Recursively collect feature indices."""
        if node.node_type == NodeType.FEATURE:
            features.append(node.value)
        if node.left:
            self._collect_features(node.left, features)
        if node.right:
            self._collect_features(node.right, features)


class GPEvolution:
    """
    Traditional GP evolution with selection, crossover, and mutation.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        input_dim: int = 512,
        max_depth: int = 4,
        tournament_size: int = 5,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        elitism: int = 5
    ):
        """
        Initialize GP evolution.
        
        Args:
            population_size: Number of individuals
            input_dim: Input dimension
            max_depth: Maximum tree depth
            tournament_size: Tournament selection size
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            elitism: Number of best individuals to keep
        """
        self.population_size = population_size
        self.input_dim = input_dim
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        
        # Initialize population
        self.population = [
            TraditionalGPTree(input_dim=input_dim, max_depth=max_depth)
            for _ in range(population_size)
        ]
        
        self.fitness_cache = {}
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_tree = None
    
    def evaluate_population(
        self, 
        fitness_fn: Callable[[TraditionalGPTree], float]
    ) -> List[float]:
        """Evaluate fitness for all individuals."""
        fitness_scores = []
        for tree in self.population:
            fitness = fitness_fn(tree)
            fitness_scores.append(fitness)
            
            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_tree = tree.copy()
        
        return fitness_scores
    
    def evolve(self, fitness_scores: List[float]) -> None:
        """
        Perform one generation of evolution.
        
        Args:
            fitness_scores: Fitness for each individual
        """
        new_population = []
        
        # Elitism: keep best individuals
        sorted_indices = np.argsort(fitness_scores)[::-1]
        for i in range(self.elitism):
            new_population.append(self.population[sorted_indices[i]].copy())
        
        # Fill rest with crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_prob:
                # Crossover
                parent1 = self._tournament_select(fitness_scores)
                parent2 = self._tournament_select(fitness_scores)
                child = self._crossover(parent1, parent2)
            else:
                # Clone
                parent = self._tournament_select(fitness_scores)
                child = parent.copy()
            
            # Mutation
            if random.random() < self.mutation_prob:
                child = self._mutate(child)
            
            # Depth check
            if child.depth() <= self.max_depth:
                new_population.append(child)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def _tournament_select(self, fitness_scores: List[float]) -> TraditionalGPTree:
        """Tournament selection."""
        indices = random.sample(range(len(self.population)), self.tournament_size)
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return self.population[best_idx]
    
    def _crossover(
        self, 
        parent1: TraditionalGPTree, 
        parent2: TraditionalGPTree
    ) -> TraditionalGPTree:
        """Subtree crossover."""
        child = parent1.copy()
        
        # Get random node from parent2
        nodes2 = parent2.get_all_nodes()
        _, _, _ = random.choice(nodes2)
        donor_node = random.choice([n for n, _, _ in nodes2])
        
        # Get random node from child to replace
        child_nodes = child.get_all_nodes()
        if len(child_nodes) > 1:
            # Don't replace root
            non_root_nodes = [(n, p, pos) for n, p, pos in child_nodes if p is not None]
            if non_root_nodes:
                _, parent, position = random.choice(non_root_nodes)
                
                # Replace
                if position == 'left':
                    parent.left = donor_node.copy()
                else:
                    parent.right = donor_node.copy()
        
        return child
    
    def _mutate(self, tree: TraditionalGPTree) -> TraditionalGPTree:
        """Apply random mutation."""
        tree = tree.copy()
        
        mutation_type = random.choice(['point', 'subtree', 'constant'])
        
        nodes = tree.get_all_nodes()
        if not nodes:
            return tree
        
        node, parent, position = random.choice(nodes)
        
        if mutation_type == 'point':
            # Change operator or terminal
            if node.node_type == NodeType.OPERATOR:
                if node.right is not None:
                    # Binary op -> different binary op
                    node.value = random.choice(self.population[0].BINARY_OPS)
                else:
                    # Unary op -> different unary op
                    node.value = random.choice(self.population[0].UNARY_OPS)
            elif node.node_type == NodeType.FEATURE:
                # Change feature index
                node.value = random.randint(0, self.input_dim - 1)
            elif node.node_type == NodeType.CONSTANT:
                # Perturb constant
                node.value += random.gauss(0, 1)
                node.value = max(-10, min(10, node.value))
        
        elif mutation_type == 'subtree':
            # Replace with new random subtree
            new_subtree = tree._create_random_tree(
                max_depth=max(1, self.max_depth - (tree.depth() - node.depth())),
                current_depth=0
            )
            if parent is None:
                tree.root = new_subtree
            elif position == 'left':
                parent.left = new_subtree
            else:
                parent.right = new_subtree
        
        elif mutation_type == 'constant':
            # Find and mutate a constant
            constants = [(n, p, pos) for n, p, pos in nodes if n.node_type == NodeType.CONSTANT]
            if constants:
                const_node, _, _ = random.choice(constants)
                const_node.value += random.gauss(0, 0.5)
                const_node.value = max(-10, min(10, const_node.value))
        
        return tree
    
    def get_best(self) -> TraditionalGPTree:
        """Get best tree found so far."""
        return self.best_tree if self.best_tree else self.population[0]
    
    def get_population_stats(self, fitness_scores: List[float]) -> Dict[str, float]:
        """Get population statistics."""
        return {
            'mean_fitness': np.mean(fitness_scores),
            'max_fitness': np.max(fitness_scores),
            'min_fitness': np.min(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'mean_depth': np.mean([t.depth() for t in self.population]),
            'mean_size': np.mean([t.size() for t in self.population]),
        }


# Test
if __name__ == '__main__':
    print("Testing Traditional GP Module")
    print("=" * 50)
    
    # Create random tree
    tree = TraditionalGPTree(input_dim=512, max_depth=4)
    print(f"Tree expression: {tree.get_expression()}")
    print(f"Tree depth: {tree.depth()}")
    print(f"Tree size: {tree.size()}")
    print(f"Used features: {tree.get_used_features()}")
    
    # Test evaluation
    x = torch.randn(8, 512)
    output = tree.evaluate(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Test evolution
    print("\n" + "=" * 50)
    print("Testing GP Evolution")
    
    def dummy_fitness(tree: TraditionalGPTree) -> float:
        x = torch.randn(32, 512)
        y = tree.evaluate(x)
        return -y.std().item()  # Minimize variance
    
    evolution = GPEvolution(population_size=20, max_depth=4)
    
    for gen in range(5):
        fitness_scores = evolution.evaluate_population(dummy_fitness)
        stats = evolution.get_population_stats(fitness_scores)
        print(f"Gen {gen}: mean={stats['mean_fitness']:.4f}, max={stats['max_fitness']:.4f}")
        evolution.evolve(fitness_scores)
    
    best = evolution.get_best()
    print(f"\nBest tree: {best.get_expression()}")