#!/usr/bin/env python3
"""
Script to fit the optimal β parameter for the conservation law:
momentum²/2 + β*log(1/px) ≈ constant

This script finds the β value that minimizes the coefficient of variation
of the conserved quantity across and within trajectories.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from scipy.optimize import minimize_scalar, minimize
import pandas as pd
from datetime import datetime


class BetaParameterFitter:
    """
    Fits the optimal β parameter for the conservation law.
    """
    
    def __init__(self, trajectories_file: str, output_dir: str = None):
        """
        Initialize the fitter.
        
        Args:
            trajectories_file: Path to the JSON file containing trajectory data
            output_dir: Directory to save results
        """
        self.trajectories_file = Path(trajectories_file)
        if output_dir:
            output_dir = output_dir + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.trajectories_file.parent / f"beta_fitting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load trajectory data
        try:
            with open(self.trajectories_file, 'r') as f:
                self.trajectories = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Trajectories file not found: {self.trajectories_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {self.trajectories_file}. Error: {e}. "
                           f"Make sure you're providing a trajectories.json file, not a Python script.")
        
        print(f"Loaded {len(self.trajectories)} trajectories from {self.trajectories_file}")
        
        # Extract all data for fitting
        self.all_momentum_squared_half = []
        self.all_neg_log_perplexity = []
        self.trajectory_data = []
        
        self._extract_all_data()
    
    def _extract_all_data(self):
        """Extract momentum²/2 and -log(px) from all trajectories."""
        
        for traj_idx, trajectory in enumerate(self.trajectories):
            steps = trajectory.get('steps', [])
            
            momentum_squared_half = []
            neg_log_perplexity = []
            
            for step in steps:
                # Extract momentum
                momentum = step.get('momentum', 0.0)
                if momentum == 0:
                    continue
                if isinstance(momentum, str):
                    try:
                        momentum = float(momentum)
                    except ValueError:
                        momentum = 0.0
                
                # Extract point perplexity
                point_ppl = step.get('point_perplexity', 1.0)
                
                # Calculate momentum²/2 and -log(point_perplexity) = log(1/px)
                mom_sq_half = np.log(0.5*momentum**2)
                neg_log_ppl = np.log(point_ppl)  # This is log(1/px)
                
                momentum_squared_half.append(mom_sq_half)
                neg_log_perplexity.append(neg_log_ppl)
                
                # Store for global optimization
                self.all_momentum_squared_half.append(mom_sq_half)
                self.all_neg_log_perplexity.append(neg_log_ppl)
            
            # Store trajectory-level data
            if len(momentum_squared_half) > 0:
                self.trajectory_data.append({
                    'trajectory_idx': traj_idx,
                    'momentum_squared_half': np.array(momentum_squared_half),
                    'neg_log_perplexity': np.array(neg_log_perplexity),
                    'num_steps': len(momentum_squared_half)
                })
        
        # Convert to numpy arrays
        self.all_momentum_squared_half = np.array(self.all_momentum_squared_half)
        self.all_neg_log_perplexity = np.array(self.all_neg_log_perplexity)
        
        print(f"Extracted {len(self.all_momentum_squared_half)} data points from {len(self.trajectory_data)} trajectories")
    
    def compute_cv_for_beta(self, beta: float, method: str = 'global') -> float:
        """
        Compute coefficient of variation for given β.
        
        Args:
            beta: The β parameter to test
            method: 'global' (all points), 'trajectory_avg' (trajectory averages), 
                   or 'within_trajectory' (average CV within trajectories)
            
        Returns:
            Coefficient of variation
        """
        
        if method == 'global':
            # Compute CV using all data points
            conserved_quantity = self.all_momentum_squared_half + beta * self.all_neg_log_perplexity
            
        elif method == 'trajectory_avg':
            # Compute CV of trajectory averages
            trajectory_averages = []
            for traj_data in self.trajectory_data:
                avg_conserved = traj_data['momentum_squared_half'] + beta * traj_data['neg_log_perplexity']
                avg_conserved = np.mean(avg_conserved)
                trajectory_averages.append(avg_conserved)
            conserved_quantity = np.array(trajectory_averages)
            
        elif method == 'within_trajectory':
            # Compute average CV within individual trajectories
            cvs = []
            for traj_data in self.trajectory_data:
                conserved = traj_data['momentum_squared_half'] + beta * traj_data['neg_log_perplexity']
                if len(conserved) > 1:
                    mean_val = np.mean(conserved)
                    std_val = np.std(conserved)
                    if mean_val != 0:
                        cv = std_val / abs(mean_val)
                        cvs.append(cv)
            conserved_quantity = np.array(cvs)
            return np.mean(conserved_quantity) if len(conserved_quantity) > 0 else float('inf')
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate CV
        if len(conserved_quantity) == 0:
            return float('inf')
        
        mean_val = np.mean(conserved_quantity)
        std_val = np.std(conserved_quantity)
        
        if abs(mean_val) < 1e-10:  # Avoid division by very small numbers
            return float('inf')
        
        cv = std_val / abs(mean_val)
        return cv
    
    def fit_beta_optimization(self, method: str = 'global', beta_range: Tuple[float, float] = (-10, 10)) -> Dict:
        """
        Find optimal β using scipy optimization.
        
        Args:
            method: CV computation method
            beta_range: Range to search for β
            
        Returns:
            Dictionary with optimization results
        """
        print(f"Fitting β using {method} method...")
        
        def objective(beta):
            return self.compute_cv_for_beta(beta, method=method)
        
        # Use scalar minimization
        result = minimize_scalar(objective, bounds=beta_range, method='bounded')
        
        optimal_beta = result.x
        optimal_cv = result.fun
        
        print(f"Optimal β ({method}): {optimal_beta:.6f}")
        print(f"Minimum CV ({method}): {optimal_cv:.6f}")
        
        return {
            'method': method,
            'optimal_beta': optimal_beta,
            'optimal_cv': optimal_cv,
            'optimization_result': result
        }
    
    def scan_beta_range(self, beta_range: Tuple[float, float] = (-5, 5), 
                       num_points: int = 1000, methods: List[str] = None) -> pd.DataFrame:
        """
        Scan a range of β values to visualize CV landscape.
        
        Args:
            beta_range: Range to scan
            num_points: Number of β values to test
            methods: List of methods to test
            
        Returns:
            DataFrame with results
        """
        if methods is None:
            methods = ['global', 'trajectory_avg']
        
        print(f"Scanning β range {beta_range} with {num_points} points...")
        
        beta_values = np.linspace(beta_range[0], beta_range[1], num_points)
        results = []
        
        for i, beta in enumerate(beta_values):
            if i % 100 == 0:
                print(f"Progress: {i+1}/{num_points}")
            
            result = {'beta': beta}
            for method in methods:
                try:
                    cv = self.compute_cv_for_beta(beta, method=method)
                    result[f'cv_{method}'] = cv
                except Exception as e:
                    print(f"Error at β={beta}, method={method}: {e}")
                    result[f'cv_{method}'] = float('inf')
            
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Save results
        results_file = self.output_dir / "beta_scan_results.csv"
        df.to_csv(results_file, index=False)
        print(f"Saved β scan results to {results_file}")
        
        return df
    
    def plot_beta_scan(self, df: pd.DataFrame):
        """
        Plot CV vs β for different methods.
        
        Args:
            df: DataFrame from scan_beta_range
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        methods = ['global', 'trajectory_avg', 'within_trajectory']
        method_names = ['Global CV', 'Trajectory Averages CV', 'Within-Trajectory Average CV']
        colors = ['blue', 'green', 'red']
        
        for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
            cv_col = f'cv_{method}'
            if cv_col in df.columns:
                # Filter out infinite values for plotting
                valid_mask = np.isfinite(df[cv_col]) & (df[cv_col] < 10)  # Reasonable upper bound
                
                if valid_mask.sum() > 0:
                    axes[i].plot(df.loc[valid_mask, 'beta'], df.loc[valid_mask, cv_col], 
                               color=color, linewidth=2)
                    
                    # Find and mark minimum
                    min_idx = df.loc[valid_mask, cv_col].idxmin()
                    min_beta = df.loc[min_idx, 'beta']
                    min_cv = df.loc[min_idx, cv_col]
                    
                    axes[i].scatter([min_beta], [min_cv], color='red', s=100, zorder=5,
                                   label=f'Min: β={min_beta:.3f}, CV={min_cv:.4f}')
                    
                    axes[i].set_xlabel('β')
                    axes[i].set_ylabel('Coefficient of Variation')
                    axes[i].set_title(f'{name}')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].legend()
                else:
                    axes[i].text(0.5, 0.5, 'No valid data', transform=axes[i].transAxes, 
                               ha='center', va='center')
        
        plt.tight_layout()
        
        plot_file = self.output_dir / "beta_scan_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved β scan plot to {plot_file}")
    
    def analyze_conservation_quality(self, beta: float) -> Dict:
        """
        Analyze how well the conservation law works with given β.
        
        Args:
            beta: The β parameter to analyze
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing conservation quality with β = {beta:.6f}")
        
        # Global analysis
        global_conserved = self.all_momentum_squared_half + beta * self.all_neg_log_perplexity
        global_cv = np.std(global_conserved) / abs(np.mean(global_conserved)) if np.mean(global_conserved) != 0 else float('inf')
        
        # Trajectory-level analysis
        trajectory_stats = []
        trajectory_averages = []
        
        for traj_data in self.trajectory_data:
            conserved = traj_data['momentum_squared_half'] + beta * traj_data['neg_log_perplexity']
            avg_conserved = np.mean(conserved)
            
            if len(conserved) > 1:
                within_traj_cv = np.std(conserved) / abs(avg_conserved) if avg_conserved != 0 else float('inf')
            else:
                within_traj_cv = 0.0
            
            trajectory_stats.append({
                'trajectory_idx': traj_data['trajectory_idx'],
                'num_steps': traj_data['num_steps'],
                'avg_conserved_quantity': avg_conserved,
                'within_trajectory_cv': within_traj_cv
            })
            
            trajectory_averages.append(avg_conserved)
        
        # Cross-trajectory CV
        trajectory_averages = np.array(trajectory_averages)
        cross_traj_cv = np.std(trajectory_averages) / abs(np.mean(trajectory_averages)) if np.mean(trajectory_averages) != 0 else float('inf')
        
        # Average within-trajectory CV
        within_traj_cvs = [stat['within_trajectory_cv'] for stat in trajectory_stats if np.isfinite(stat['within_trajectory_cv'])]
        avg_within_traj_cv = np.mean(within_traj_cvs) if len(within_traj_cvs) > 0 else float('inf')
        
        return {
            'beta': beta,
            'global_cv': global_cv,
            'cross_trajectory_cv': cross_traj_cv,
            'avg_within_trajectory_cv': avg_within_traj_cv,
            'trajectory_stats': trajectory_stats,
            'global_mean': np.mean(global_conserved),
            'global_std': np.std(global_conserved),
            'num_trajectories': len(trajectory_stats),
            'total_points': len(global_conserved)
        }
    
    def generate_report(self, optimization_results: List[Dict], best_beta: float):
        """
        Generate a comprehensive report of the β fitting results.
        
        Args:
            optimization_results: Results from different optimization methods
            best_beta: The recommended β value
        """
        report_file = self.output_dir / "beta_fitting_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("BETA PARAMETER FITTING REPORT\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Input file: {self.trajectories_file}\n")
            f.write(f"Number of trajectories: {len(self.trajectory_data)}\n")
            f.write(f"Total data points: {len(self.all_momentum_squared_half)}\n\n")
            
            # Add component averages
            f.write("COMPONENT STATISTICS\n")
            f.write("-" * 20 + "\n\n")
            f.write(f"Average momentum²/2: {np.mean(self.all_momentum_squared_half):.6f}\n")
            f.write(f"Std momentum²/2: {np.std(self.all_momentum_squared_half):.6f}\n")
            f.write(f"Average log(perplexity): {np.mean(self.all_neg_log_perplexity):.6f}\n")
            f.write(f"Std log(perplexity): {np.std(self.all_neg_log_perplexity):.6f}\n")
            f.write(f"Ratio of averages (momentum²/2) / log(perplexity): {np.mean(self.all_momentum_squared_half) / np.mean(self.all_neg_log_perplexity):.6f}\n\n")
            
            f.write("OPTIMIZATION RESULTS\n")
            f.write("-" * 20 + "\n\n")
            
            for result in optimization_results:
                f.write(f"Method: {result['method']}\n")
                f.write(f"  Optimal β: {result['optimal_beta']:.6f}\n")
                f.write(f"  Minimum CV: {result['optimal_cv']:.6f}\n")
                f.write(f"  Optimization success: {result['optimization_result'].success}\n\n")
            
            f.write(f"RECOMMENDED β VALUE: {best_beta:.6f}\n\n")
            
            # Analyze conservation quality with recommended β
            analysis = self.analyze_conservation_quality(best_beta)
            
            f.write("CONSERVATION ANALYSIS WITH RECOMMENDED β\n")
            f.write("-" * 45 + "\n\n")
            f.write(f"Global CV: {analysis['global_cv']:.6f}\n")
            f.write(f"Cross-trajectory CV: {analysis['cross_trajectory_cv']:.6f}\n")
            f.write(f"Average within-trajectory CV: {analysis['avg_within_trajectory_cv']:.6f}\n\n")
            
            f.write(f"Global mean conserved quantity: {analysis['global_mean']:.4f}\n")
            f.write(f"Global std conserved quantity: {analysis['global_std']:.4f}\n\n")
            
            # Add component contributions to conserved quantity
            f.write("CONSERVED QUANTITY BREAKDOWN\n")
            f.write("-" * 30 + "\n\n")
            momentum_contribution = np.mean(self.all_momentum_squared_half)
            log_contribution = best_beta * np.mean(self.all_neg_log_perplexity)
            f.write(f"Average momentum²/2 contribution: {momentum_contribution:.6f}\n")
            f.write(f"Average β*log(perplexity) contribution: {log_contribution:.6f}\n")
            f.write(f"Total average conserved quantity: {momentum_contribution + log_contribution:.6f}\n")
            f.write(f"Contribution ratio (momentum/log): {abs(momentum_contribution / log_contribution):.6f}\n")
        
        print(f"Generated β fitting report: {report_file}")
    
    def run_full_fitting(self) -> float:
        """
        Run the complete β fitting pipeline.
        
        Returns:
            Recommended β value
        """
        print("Starting β parameter fitting...")
        
        # Perform optimization with different methods
        methods = ['global', 'trajectory_avg', 'within_trajectory']
        optimization_results = []
        
        for method in methods:
            try:
                result = self.fit_beta_optimization(method=method, beta_range=(1, 1.01))
                optimization_results.append(result)
            except Exception as e:
                print(f"Error in {method} optimization: {e}")
        
        # Perform detailed β scan
        scan_df = self.scan_beta_range(beta_range=(1.0, 1.01), num_points=100, methods=methods)
        self.plot_beta_scan(scan_df)
        
        # Choose best β (prioritize global method)
        best_beta = None
        for result in optimization_results:
            if result['method'] == 'global':
                best_beta = result['optimal_beta']
                break
        
        if best_beta is None and len(optimization_results) > 0:
            # Fallback to first available result
            best_beta = optimization_results[0]['optimal_beta']
        
        if best_beta is None:
            print("Warning: Could not find optimal β, using default value")
            best_beta = 1.0
        
        # Generate comprehensive report
        self.generate_report(optimization_results, best_beta)
        
        print(f"\nβ fitting complete! Recommended β value: {best_beta:.6f}")
        print(f"Results saved to: {self.output_dir}")
        
        # Print component statistics
        print(f"\nComponent Statistics:")
        print(f"  Average momentum²/2: {np.mean(self.all_momentum_squared_half):.6f}")
        print(f"  Average log(perplexity): {np.mean(self.all_neg_log_perplexity):.6f}")
        print(f"  Ratio (momentum²/2) / log(perplexity): {np.mean(self.all_momentum_squared_half) / np.mean(self.all_neg_log_perplexity):.6f}")
        
        # Print conserved quantity breakdown
        momentum_contribution = np.mean(self.all_momentum_squared_half)
        log_contribution = best_beta * np.mean(self.all_neg_log_perplexity)
        print(f"\nConserved Quantity Breakdown:")
        print(f"  Momentum²/2 contribution: {momentum_contribution:.6f}")
        print(f"  β*log(perplexity) contribution: {log_contribution:.6f}")
        print(f"  Total: {momentum_contribution + log_contribution:.6f}")
        print(f"  Contribution ratio: {abs(momentum_contribution / log_contribution):.6f}")
        
        return best_beta


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Fit optimal β parameter for conservation law')
    parser.add_argument('trajectories_file', help='Path to trajectories JSON file')
    parser.add_argument('--output_dir', help='Output directory for results', default='./beta_fitting')
    parser.add_argument('--beta_range', nargs=2, type=float, default=[-10, 10],
                       help='Range to search for β (default: -10 10)')
    
    args = parser.parse_args()
    
    # Create fitter and run analysis
    fitter = BetaParameterFitter(
        trajectories_file=args.trajectories_file,
        output_dir=args.output_dir
    )
    
    best_beta = fitter.run_full_fitting()
    
    print(f"\nRecommended β parameter: {best_beta:.6f}")
    print("Use this value in your main experiment!")


if __name__ == "__main__":
    main() 