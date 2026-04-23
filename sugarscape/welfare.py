"""Welfare evaluation metrics for Sugarscape simulations."""

import numpy as np
from typing import List, Dict, Any
from sugarscape.agent import SugarAgent


class WelfareCalculator:
    """Calculates various social welfare metrics for agent populations."""
    
    @staticmethod
    def calculate_individual_welfares(agents: List[SugarAgent]) -> List[float]:
        """Calculate individual welfare for each agent.
        
        Args:
            agents: List of SugarAgent instances
            
        Returns:
            List of welfare values (one per agent)
        """
        return [agent.welfare for agent in agents if agent.alive]
    
    @staticmethod
    def calculate_utilitarian_welfare(agents: List[SugarAgent]) -> float:
        """Calculate utilitarian social welfare (sum of all individual utilities).
        
        This metric maximizes total welfare without regard to distribution.
        
        Args:
            agents: List of SugarAgent instances
            
        Returns:
            Total welfare across all agents
        """
        welfares = WelfareCalculator.calculate_individual_welfares(agents)
        return float(np.sum(welfares)) if welfares else 0.0
    
    @staticmethod
    def calculate_average_welfare(agents: List[SugarAgent]) -> float:
        """Calculate average (mean) welfare across agents.
        
        Args:
            agents: List of SugarAgent instances
            
        Returns:
            Mean welfare value
        """
        welfares = WelfareCalculator.calculate_individual_welfares(agents)
        return float(np.mean(welfares)) if welfares else 0.0
    
    @staticmethod
    def calculate_rawlsian_welfare(agents: List[SugarAgent]) -> float:
        """Calculate Rawlsian social welfare (welfare of worst-off agent).
        
        Follows Rawls' maximin principle - maximize the minimum welfare.
        
        Args:
            agents: List of SugarAgent instances
            
        Returns:
            Minimum welfare value (welfare of worst-off agent)
        """
        welfares = WelfareCalculator.calculate_individual_welfares(agents)
        return float(np.min(welfares)) if welfares else 0.0
    
    @staticmethod
    def calculate_nash_welfare(agents: List[SugarAgent]) -> float:
        """Calculate Nash social welfare (geometric mean of utilities).
        
        Nash welfare balances efficiency and equity. Uses log form to avoid overflow.
        
        Args:
            agents: List of SugarAgent instances
            
        Returns:
            Nash welfare (geometric mean of individual welfares)
        """
        welfares = WelfareCalculator.calculate_individual_welfares(agents)
        if not welfares:
            return 0.0
        
        # Use log form for numerical stability
        # Nash = (∏ w_i)^(1/n) = exp(mean(log(w_i)))
        # Add small epsilon to avoid log(0)
        log_welfares = [np.log(max(w, 1e-10)) for w in welfares]
        log_nash = np.mean(log_welfares)
        return float(np.exp(log_nash))
    
    @staticmethod
    def calculate_inequality_adjusted_welfare(agents: List[SugarAgent]) -> Dict[str, float]:
        """Calculate inequality-adjusted welfare metrics.
        
        Combines efficiency (mean welfare) with equity (distribution).
        
        Args:
            agents: List of SugarAgent instances
            
        Returns:
            Dictionary with various inequality-adjusted metrics
        """
        welfares = WelfareCalculator.calculate_individual_welfares(agents)
        if not welfares:
            return {
                "mean_welfare": 0.0,
                "welfare_gini": 0.0,
                "gini_adjusted_welfare": 0.0,
                "atkinson_index_05": 0.0,
                "atkinson_adjusted_05": 0.0
            }
        
        mean_welfare = float(np.mean(welfares))
        gini = WelfareCalculator._gini_coefficient(welfares)
        
        # Gini-adjusted welfare: mean × (1 - Gini)
        gini_adjusted = mean_welfare * (1 - gini)
        
        # Atkinson index with ε=0.5 (moderate inequality aversion)
        atkinson_05 = WelfareCalculator._atkinson_index(welfares, epsilon=0.5)
        atkinson_adjusted_05 = mean_welfare * (1 - atkinson_05)
        
        return {
            "mean_welfare": mean_welfare,
            "welfare_gini": gini,
            "gini_adjusted_welfare": float(gini_adjusted),
            "atkinson_index_05": float(atkinson_05),
            "atkinson_adjusted_05": float(atkinson_adjusted_05)
        }
    
    @staticmethod
    def calculate_survival_metrics(agents: List[SugarAgent], initial_population: int) -> Dict[str, float]:
        """Calculate population survival metrics.
        
        Args:
            agents: List of SugarAgent instances
            initial_population: Initial number of agents at start
            
        Returns:
            Dictionary with survival metrics
        """
        alive_count = sum(1 for a in agents if a.alive)
        survival_rate = alive_count / initial_population if initial_population > 0 else 0.0
        
        # Calculate average age of living agents
        living_agents = [a for a in agents if a.alive]
        mean_age = float(np.mean([a.age for a in living_agents])) if living_agents else 0.0
        
        # Calculate lifespan utilization (age/max_age)
        lifespan_utils = [a.age / a.max_age for a in living_agents if a.max_age > 0]
        mean_lifespan_util = float(np.mean(lifespan_utils)) if lifespan_utils else 0.0
        
        return {
            "alive_count": alive_count,
            "survival_rate": float(survival_rate),
            "mean_age": mean_age,
            "mean_lifespan_utilization": mean_lifespan_util
        }
    
    @staticmethod
    def calculate_all_welfare_metrics(agents: List[SugarAgent], initial_population: int = None) -> Dict[str, Any]:
        """Calculate comprehensive welfare metrics.
        
        Args:
            agents: List of SugarAgent instances
            initial_population: Initial population count (defaults to current if not provided)
            
        Returns:
            Dictionary containing all welfare metrics
        """
        if initial_population is None:
            initial_population = len(agents)
        
        # Core welfare metrics
        utilitarian = WelfareCalculator.calculate_utilitarian_welfare(agents)
        average = WelfareCalculator.calculate_average_welfare(agents)
        rawlsian = WelfareCalculator.calculate_rawlsian_welfare(agents)
        nash = WelfareCalculator.calculate_nash_welfare(agents)
        
        # Inequality-adjusted metrics
        inequality_metrics = WelfareCalculator.calculate_inequality_adjusted_welfare(agents)
        
        # Survival metrics
        survival_metrics = WelfareCalculator.calculate_survival_metrics(agents, initial_population)
        
        # Welfare distribution statistics
        welfares = WelfareCalculator.calculate_individual_welfares(agents)
        welfare_stats = {
            "welfare_std": float(np.std(welfares)) if welfares else 0.0,
            "welfare_median": float(np.median(welfares)) if welfares else 0.0,
            "welfare_max": float(np.max(welfares)) if welfares else 0.0,
            "welfare_min": float(np.min(welfares)) if welfares else 0.0,
            "welfare_q25": float(np.percentile(welfares, 25)) if welfares else 0.0,
            "welfare_q75": float(np.percentile(welfares, 75)) if welfares else 0.0,
        }
        
        return {
            # Primary welfare metrics
            "utilitarian_welfare": utilitarian,
            "average_welfare": average,
            "rawlsian_welfare": rawlsian,
            "nash_welfare": nash,
            
            # Inequality-adjusted metrics
            **inequality_metrics,
            
            # Survival metrics
            **survival_metrics,
            
            # Distribution statistics
            **welfare_stats
        }
    
    @staticmethod
    def _gini_coefficient(values: List[float]) -> float:
        """Calculate Gini coefficient for a distribution.
        
        Args:
            values: List of values (e.g., wealth or welfare)
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        if not values:
            return 0.0
        
        sorted_vals = sorted(values)
        n = len(values)
        total = sum(sorted_vals)
        
        if total == 0:
            return 0.0
        
        # Gini = (2 * Σ(i * x_i) / (n * Σx_i)) - (n+1)/n
        weighted_sum = sum((i + 1) * val for i, val in enumerate(sorted_vals))
        return (2 * weighted_sum) / (n * total) - (n + 1) / n
    
    @staticmethod
    def _atkinson_index(values: List[float], epsilon: float = 0.5) -> float:
        """Calculate Atkinson inequality index.
        
        Args:
            values: List of values (e.g., wealth or welfare)
            epsilon: Inequality aversion parameter (0 = no aversion, ∞ = Rawlsian)
            
        Returns:
            Atkinson index (0 = perfect equality, 1 = perfect inequality)
        """
        if not values or len(values) == 0:
            return 0.0
        
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
        
        n = len(values)
        
        if epsilon == 1.0:
            # Special case: geometric mean
            log_vals = [np.log(max(v, 1e-10)) for v in values]
            ede = np.exp(np.mean(log_vals))  # Equally Distributed Equivalent
        else:
            # General case
            powered = [(max(v, 1e-10) / mean_val) ** (1 - epsilon) for v in values]
            ede = mean_val * (np.mean(powered) ** (1 / (1 - epsilon)))
        
        atkinson = 1 - (ede / mean_val)
        return float(max(0.0, min(1.0, atkinson)))  # Clamp to [0, 1]


