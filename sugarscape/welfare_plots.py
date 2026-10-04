"""Visualization tools for welfare analysis in Sugarscape simulations."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class WelfarePlotter:
    """Generate comprehensive welfare visualizations for Sugarscape experiments."""
    
    @staticmethod
    def plot_welfare_timeseries(
        metrics_df: pd.DataFrame,
        save_path: str,
        title_prefix: str = ""
    ):
        """Plot time series of all welfare metrics.
        
        Args:
            metrics_df: DataFrame with metrics over time
            save_path: Path to save the plot
            title_prefix: Optional prefix for plot title (e.g., "LLM Agents" or "Rule-Based")
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        title = f"{title_prefix} Welfare Metrics Over Time" if title_prefix else "Welfare Metrics Over Time"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Primary Welfare Measures
        ax1 = fig.add_subplot(gs[0, :])
        if 'utilitarian_welfare' in metrics_df.columns:
            ax1.plot(metrics_df['tick'], metrics_df['utilitarian_welfare'], 
                    label='Utilitarian (Sum)', linewidth=2, alpha=0.8)
        if 'average_welfare' in metrics_df.columns:
            ax1.plot(metrics_df['tick'], metrics_df['average_welfare'], 
                    label='Average (Mean)', linewidth=2, alpha=0.8)
        if 'nash_welfare' in metrics_df.columns:
            ax1.plot(metrics_df['tick'], metrics_df['nash_welfare'], 
                    label='Nash (Geometric Mean)', linewidth=2, alpha=0.8)
        if 'rawlsian_welfare' in metrics_df.columns:
            ax1.plot(metrics_df['tick'], metrics_df['rawlsian_welfare'], 
                    label='Rawlsian (Min)', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Welfare Value')
        ax1.set_title('Primary Welfare Measures')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Inequality-Adjusted Welfare
        ax2 = fig.add_subplot(gs[1, 0])
        if 'gini_adjusted_welfare' in metrics_df.columns:
            ax2.plot(metrics_df['tick'], metrics_df['gini_adjusted_welfare'], 
                    label='Gini-Adjusted', linewidth=2, color='green', alpha=0.8)
        if 'atkinson_adjusted_05' in metrics_df.columns:
            ax2.plot(metrics_df['tick'], metrics_df['atkinson_adjusted_05'], 
                    label='Atkinson-Adjusted (ε=0.5)', linewidth=2, color='purple', alpha=0.8)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Adjusted Welfare')
        ax2.set_title('Inequality-Adjusted Welfare')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. Inequality Metrics
        ax3 = fig.add_subplot(gs[1, 1])
        if 'welfare_gini' in metrics_df.columns:
            ax3.plot(metrics_df['tick'], metrics_df['welfare_gini'], 
                    label='Welfare Gini', linewidth=2, color='red', alpha=0.8)
        if 'gini' in metrics_df.columns:
            ax3.plot(metrics_df['tick'], metrics_df['gini'], 
                    label='Wealth Gini', linewidth=2, color='orange', alpha=0.8)
        if 'atkinson_index_05' in metrics_df.columns:
            ax3.plot(metrics_df['tick'], metrics_df['atkinson_index_05'], 
                    label='Atkinson Index (ε=0.5)', linewidth=2, color='brown', alpha=0.8)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Inequality Index')
        ax3.set_title('Inequality Measures (0=equal, 1=unequal)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. Survival Metrics
        ax4 = fig.add_subplot(gs[2, 0])
        if 'survival_rate' in metrics_df.columns:
            ax4.plot(metrics_df['tick'], metrics_df['survival_rate'], 
                    linewidth=2, color='darkblue', alpha=0.8)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Survival Rate')
        ax4.set_title('Population Survival Rate')
        ax4.set_ylim([0, 1.05])
        ax4.grid(True, alpha=0.3)
        
        # 5. Population
        ax5 = fig.add_subplot(gs[2, 1])
        if 'population' in metrics_df.columns:
            ax5.plot(metrics_df['tick'], metrics_df['population'], 
                    linewidth=2, color='darkgreen', alpha=0.8)
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Living Agents')
        ax5.set_title('Population Over Time')
        ax5.grid(True, alpha=0.3)
        
        # 6. Welfare Distribution
        ax6 = fig.add_subplot(gs[3, 0])
        if all(col in metrics_df.columns for col in ['welfare_min', 'welfare_max', 'welfare_median']):
            ax6.fill_between(metrics_df['tick'], 
                           metrics_df['welfare_min'], 
                           metrics_df['welfare_max'],
                           alpha=0.3, label='Min-Max Range')
            ax6.plot(metrics_df['tick'], metrics_df['welfare_median'], 
                    linewidth=2, color='darkred', label='Median')
            if 'average_welfare' in metrics_df.columns:
                ax6.plot(metrics_df['tick'], metrics_df['average_welfare'], 
                        linewidth=2, color='blue', linestyle='--', label='Mean')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Welfare')
        ax6.set_title('Welfare Distribution (Min, Median, Max)')
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3)
        
        # 7. Lifespan Utilization
        ax7 = fig.add_subplot(gs[3, 1])
        if 'mean_lifespan_utilization' in metrics_df.columns:
            ax7.plot(metrics_df['tick'], metrics_df['mean_lifespan_utilization'], 
                    linewidth=2, color='teal', alpha=0.8)
        ax7.set_xlabel('Time Step')
        ax7.set_ylabel('Lifespan Utilization')
        ax7.set_title('Mean Lifespan Utilization (age/max_age)')
        ax7.set_ylim([0, 1.05])
        ax7.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved welfare time series plot to: {save_path}")
    
    @staticmethod
    def plot_welfare_comparison(
        llm_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        save_path: str
    ):
        """Compare welfare metrics between LLM and baseline agents.
        
        Args:
            llm_df: DataFrame with LLM agent metrics
            baseline_df: DataFrame with baseline agent metrics
            save_path: Path to save the comparison plot
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('LLM vs Baseline Agent Welfare Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_compare = [
            ('utilitarian_welfare', 'Utilitarian Welfare (Total)'),
            ('average_welfare', 'Average Welfare'),
            ('nash_welfare', 'Nash Welfare'),
            ('rawlsian_welfare', 'Rawlsian Welfare (Min)'),
            ('survival_rate', 'Survival Rate'),
            ('welfare_gini', 'Welfare Inequality (Gini)')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_compare):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            
            if metric in llm_df.columns:
                ax.plot(llm_df['tick'], llm_df[metric], 
                       label='LLM Agents', linewidth=2, alpha=0.8)
            
            if metric in baseline_df.columns:
                ax.plot(baseline_df['tick'], baseline_df[metric], 
                       label='Baseline Agents', linewidth=2, alpha=0.8, linestyle='--')
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved welfare comparison plot to: {save_path}")
    
    @staticmethod
    def plot_welfare_summary(
        metrics_df: pd.DataFrame,
        save_path: str,
        title_prefix: str = ""
    ):
        """Plot summary statistics of welfare metrics.
        
        Args:
            metrics_df: DataFrame with metrics over time
            save_path: Path to save the plot
            title_prefix: Optional prefix for plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        title = f"{title_prefix} Welfare Summary Statistics" if title_prefix else "Welfare Summary Statistics"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Calculate summary statistics for key metrics
        welfare_cols = [
            'utilitarian_welfare', 'average_welfare', 
            'nash_welfare', 'rawlsian_welfare'
        ]
        
        # 1. Final values comparison
        ax = axes[0, 0]
        final_values = []
        labels = []
        for col in welfare_cols:
            if col in metrics_df.columns and len(metrics_df[col]) > 0:
                final_values.append(metrics_df[col].iloc[-1])
                labels.append(col.replace('_welfare', '').title())
        
        if final_values:
            bars = ax.bar(range(len(final_values)), final_values, alpha=0.7)
            ax.set_xticks(range(len(final_values)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Welfare Value')
            ax.set_title('Final Welfare Values')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Color bars
            colors = ['blue', 'green', 'orange', 'red']
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # 2. Mean values over time
        ax = axes[0, 1]
        mean_values = []
        for col in welfare_cols:
            if col in metrics_df.columns:
                mean_values.append(metrics_df[col].mean())
        
        if mean_values:
            bars = ax.bar(range(len(mean_values)), mean_values, alpha=0.7)
            ax.set_xticks(range(len(mean_values)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Mean Welfare Value')
            ax.set_title('Average Welfare Over Time')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # 3. Inequality metrics comparison
        ax = axes[1, 0]
        inequality_metrics = {
            'Wealth Gini': metrics_df['gini'].mean() if 'gini' in metrics_df.columns else 0,
            'Welfare Gini': metrics_df['welfare_gini'].mean() if 'welfare_gini' in metrics_df.columns else 0,
            'Atkinson (ε=0.5)': metrics_df['atkinson_index_05'].mean() if 'atkinson_index_05' in metrics_df.columns else 0
        }
        
        ineq_labels = list(inequality_metrics.keys())
        ineq_values = list(inequality_metrics.values())
        
        bars = ax.bar(range(len(ineq_values)), ineq_values, alpha=0.7, color='red')
        ax.set_xticks(range(len(ineq_values)))
        ax.set_xticklabels(ineq_labels, rotation=45, ha='right')
        ax.set_ylabel('Inequality Index')
        ax.set_title('Mean Inequality Metrics')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Key performance indicators
        ax = axes[1, 1]
        
        kpis = {}
        if 'survival_rate' in metrics_df.columns:
            kpis['Final Survival Rate'] = metrics_df['survival_rate'].iloc[-1]
        if 'mean_lifespan_utilization' in metrics_df.columns:
            kpis['Mean Lifespan Use'] = metrics_df['mean_lifespan_utilization'].mean()
        if 'gini_adjusted_welfare' in metrics_df.columns:
            kpis['Mean Adjusted Welfare'] = metrics_df['gini_adjusted_welfare'].mean() / 100  # Normalize
        
        if kpis:
            kpi_labels = list(kpis.keys())
            kpi_values = list(kpis.values())
            
            bars = ax.bar(range(len(kpi_values)), kpi_values, alpha=0.7, color='teal')
            ax.set_xticks(range(len(kpi_values)))
            ax.set_xticklabels(kpi_labels, rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.set_title('Key Performance Indicators')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved welfare summary plot to: {save_path}")
    
    @staticmethod
    def generate_all_plots(
        csv_path: str,
        plots_dir: str,
        title_prefix: str = ""
    ):
        """Generate all welfare plots from a metrics CSV file.
        
        Args:
            csv_path: Path to metrics CSV file
            plots_dir: Directory to save plots
            title_prefix: Optional prefix for plot titles
        """
        # Read metrics
        df = pd.read_csv(csv_path)
        
        # Create plots directory if needed
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        WelfarePlotter.plot_welfare_timeseries(
            df,
            f"{plots_dir}/welfare_timeseries.png",
            title_prefix
        )
        
        WelfarePlotter.plot_welfare_summary(
            df,
            f"{plots_dir}/welfare_summary.png",
            title_prefix
        )
        
        print(f"Generated all welfare plots in: {plots_dir}")
    
    @staticmethod
    def generate_comparison_plots(
        llm_csv_path: str,
        baseline_csv_path: str,
        output_dir: str
    ):
        """Generate comparison plots between LLM and baseline experiments.
        
        Args:
            llm_csv_path: Path to LLM experiment metrics CSV
            baseline_csv_path: Path to baseline experiment metrics CSV
            output_dir: Directory to save comparison plots
        """
        # Read both datasets
        llm_df = pd.read_csv(llm_csv_path)
        baseline_df = pd.read_csv(baseline_csv_path)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate comparison plot
        WelfarePlotter.plot_welfare_comparison(
            llm_df,
            baseline_df,
            f"{output_dir}/llm_vs_baseline_comparison.png"
        )
        
        print(f"Generated comparison plots in: {output_dir}")


