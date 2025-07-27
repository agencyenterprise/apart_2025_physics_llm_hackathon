#!/usr/bin/env python3
"""
Phase Space Visualization Library
=================================

This module contains all visualization methods for momentum-perplexity phase space analysis.
Extracted from the main experiment file to improve code organization and reusability.

The PhaseSpaceVisualizer class provides comprehensive visualization capabilities including:
- Individual trajectory plots
- Phase portraits
- Density maps and flow analysis  
- Vector fields
- Attractor analysis
- Temporal evolution analysis
- Token type analysis
- Velocity/acceleration fields
- Energy analysis (spatial and temporal)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches

class PhaseSpaceVisualizer:
    """Comprehensive visualization suite for phase space analysis"""
    
    def __init__(self):
        """Initialize the visualizer"""
        # Define colors for token types
        self.token_type_colors = {
            'punctuation': 'red',
            'function_word': 'blue', 
            'capitalized': 'green',
            'long_content': 'purple',
            'short_content': 'orange',
            'number': 'brown',
            'empty': 'gray',
            'other': 'pink'
        }
    
    def create_all_visualizations(self, trajectories: List[Dict], output_dir: str, perplexity_cutoff: Optional[float] = None) -> None:
        """Create comprehensive phase space visualizations"""
        
        print(f"🎨 Creating phase space visualizations...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        all_momentum = []
        all_point_perp = []
        all_dist_perp = []
        
        for traj in trajectories:
            momentum = traj['momentum'][1:]  # Skip first step (momentum=0)
            point_perp = traj['point_perplexity'][1:]
            dist_perp = traj['distribution_perplexity'][1:]
            
            all_momentum.extend(momentum)
            all_point_perp.extend(point_perp)
            all_dist_perp.extend(dist_perp)
        
        # Apply perplexity cutoff if specified
        if perplexity_cutoff is not None:
            all_point_perp = np.clip(all_point_perp, None, perplexity_cutoff)
            all_dist_perp = np.clip(all_dist_perp, None, perplexity_cutoff)
            cutoff_suffix = f" (cutoff: {perplexity_cutoff})"
            print(f"  🔪 Applied perplexity cutoff: {perplexity_cutoff}")
        else:
            cutoff_suffix = ""
        
        # Convert to numpy arrays
        all_momentum = np.array(all_momentum)
        all_point_perp = np.array(all_point_perp)
        all_dist_perp = np.array(all_dist_perp)
        
        print(f"  📊 Plotting {len(all_momentum)} phase space points from {len(trajectories)} trajectories")
        
        # Create all visualizations
        self.plot_individual_trajectories(trajectories, output_dir, cutoff_suffix, perplexity_cutoff)
        self.plot_phase_portraits(trajectories, output_dir, cutoff_suffix, perplexity_cutoff)
        self.plot_density_and_flow(all_momentum, all_point_perp, all_dist_perp, output_dir, cutoff_suffix)
        self.plot_vector_fields(trajectories, output_dir, cutoff_suffix, perplexity_cutoff)
        self.plot_attractor_analysis(all_momentum, all_point_perp, all_dist_perp, output_dir, cutoff_suffix)
        self.plot_temporal_evolution(trajectories, output_dir, cutoff_suffix, perplexity_cutoff)
        self.plot_token_type_analysis(trajectories, output_dir, cutoff_suffix, perplexity_cutoff)
        self.plot_velocity_acceleration_fields(trajectories, output_dir, cutoff_suffix, perplexity_cutoff)
        self.plot_energy_analysis(trajectories, output_dir, cutoff_suffix, perplexity_cutoff)
        self.plot_average_dynamics(trajectories, output_dir, cutoff_suffix, perplexity_cutoff)
        
        print(f"  ✅ Phase space visualizations saved to {output_dir}/")
    
    def plot_individual_trajectories(self, trajectories: List[Dict], output_dir: str, cutoff_suffix: str, perplexity_cutoff: Optional[float]) -> None:
        """Plot first few individual trajectories to show detailed paths"""
        
        num_to_plot = min(6, len(trajectories))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(num_to_plot):
            traj = trajectories[i]
            ax = axes[i]
            
            momentum = np.array(traj['momentum'][1:])  # Skip first step
            point_perp = np.array(traj['point_perplexity'][1:])
            
            if perplexity_cutoff is not None:
                point_perp = np.clip(point_perp, None, perplexity_cutoff)
            
            # Plot trajectory path with arrows
            if len(momentum) > 1:
                ax.plot(momentum, point_perp, 'b-', alpha=0.7, linewidth=2, label='Point Perplexity')
                ax.scatter(momentum, point_perp, c=range(len(momentum)), cmap='Blues', s=50, zorder=5)
                
                # Add arrows to show direction
                for j in range(len(momentum)-1):
                    ax.annotate('', xy=(momentum[j+1], point_perp[j+1]), xytext=(momentum[j], point_perp[j]),
                               arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
            
            ax.set_title(f'Trajectory {i+1}: Point Perplexity\n"{traj["text"][:40]}..."', fontsize=10)
            ax.set_xlabel('Momentum')
            ax.set_ylabel('Point Perplexity')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_to_plot, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/individual_trajectories_point_perplexity{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Repeat for distribution perplexity
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(num_to_plot):
            traj = trajectories[i]
            ax = axes[i]
            
            momentum = np.array(traj['momentum'][1:])
            dist_perp = np.array(traj['distribution_perplexity'][1:])
            
            if perplexity_cutoff is not None:
                dist_perp = np.clip(dist_perp, None, perplexity_cutoff)
            
            if len(momentum) > 1:
                ax.plot(momentum, dist_perp, 'g-', alpha=0.7, linewidth=2, label='Distribution Perplexity')
                ax.scatter(momentum, dist_perp, c=range(len(momentum)), cmap='Greens', s=50, zorder=5)
                
                for j in range(len(momentum)-1):
                    ax.annotate('', xy=(momentum[j+1], dist_perp[j+1]), xytext=(momentum[j], dist_perp[j]),
                               arrowprops=dict(arrowstyle='->', color='green', alpha=0.6))
            
            ax.set_title(f'Trajectory {i+1}: Distribution Perplexity\n"{traj["text"][:40]}..."', fontsize=10)
            ax.set_xlabel('Momentum')
            ax.set_ylabel('Distribution Perplexity')
            ax.grid(True, alpha=0.3)
        
        for i in range(num_to_plot, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/individual_trajectories_distribution_perplexity{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_phase_portraits(self, trajectories: List[Dict], output_dir: str, cutoff_suffix: str, perplexity_cutoff: Optional[float]) -> None:
        """Plot phase portraits showing all trajectories together"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Color palette for different trajectories
        colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))
        
        for i, traj in enumerate(trajectories):
            momentum = np.array(traj['momentum'][1:])
            point_perp = np.array(traj['point_perplexity'][1:])
            dist_perp = np.array(traj['distribution_perplexity'][1:])
            
            if perplexity_cutoff is not None:
                point_perp = np.clip(point_perp, None, perplexity_cutoff)
                dist_perp = np.clip(dist_perp, None, perplexity_cutoff)
            
            if len(momentum) > 1:
                # Point perplexity phase portrait
                ax1.plot(momentum, point_perp, '-', color=colors[i], alpha=0.6, linewidth=1)
                ax1.scatter(momentum[0], point_perp[0], color=colors[i], s=50, marker='o', alpha=0.8)  # Start
                ax1.scatter(momentum[-1], point_perp[-1], color=colors[i], s=50, marker='s', alpha=0.8)  # End
                
                # Distribution perplexity phase portrait
                ax2.plot(momentum, dist_perp, '-', color=colors[i], alpha=0.6, linewidth=1)
                ax2.scatter(momentum[0], dist_perp[0], color=colors[i], s=50, marker='o', alpha=0.8)
                ax2.scatter(momentum[-1], dist_perp[-1], color=colors[i], s=50, marker='s', alpha=0.8)
        
        ax1.set_title(f'Phase Portrait: Momentum vs Point Perplexity{cutoff_suffix}')
        ax1.set_xlabel('Momentum')
        ax1.set_ylabel('Point Perplexity')
        ax1.grid(True, alpha=0.3)
        ax1.legend(['○ Start', '□ End'], loc='upper right')
        
        ax2.set_title(f'Phase Portrait: Momentum vs Distribution Perplexity{cutoff_suffix}')
        ax2.set_xlabel('Momentum')
        ax2.set_ylabel('Distribution Perplexity')
        ax2.grid(True, alpha=0.3)
        ax2.legend(['○ Start', '□ End'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/phase_portraits{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_density_and_flow(self, momentum: np.ndarray, point_perp: np.ndarray, dist_perp: np.ndarray, output_dir: str, cutoff_suffix: str) -> None:
        """Plot density maps and advanced flow analysis"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 2D density histogram - Point perplexity
        H1, xedges, yedges = np.histogram2d(momentum, point_perp, bins=50)
        im1 = ax1.imshow(H1.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                        cmap='viridis', aspect='auto')
        ax1.set_title(f'Density Map: Momentum vs Point Perplexity{cutoff_suffix}')
        ax1.set_xlabel('Momentum')
        ax1.set_ylabel('Point Perplexity')
        plt.colorbar(im1, ax=ax1, label='Density')
        
        # 2. 2D density histogram - Distribution perplexity
        H2, xedges, yedges = np.histogram2d(momentum, dist_perp, bins=50)
        im2 = ax2.imshow(H2.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                        cmap='plasma', aspect='auto')
        ax2.set_title(f'Density Map: Momentum vs Distribution Perplexity{cutoff_suffix}')
        ax2.set_xlabel('Momentum')
        ax2.set_ylabel('Distribution Perplexity')
        plt.colorbar(im2, ax=ax2, label='Density')
        
        # 3. Contour plot - Point perplexity
        ax3.scatter(momentum, point_perp, alpha=0.5, s=5, c='blue')
        
        # Create smooth density contours
        try:
            # Create grid for interpolation
            xi = np.linspace(momentum.min(), momentum.max(), 50)
            yi = np.linspace(point_perp.min(), point_perp.max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Interpolate density
            zi = griddata((momentum, point_perp), np.ones_like(momentum), (Xi, Yi), method='cubic')
            zi = gaussian_filter(zi, sigma=1.0)
            
            # Plot contours
            contour = ax3.contour(Xi, Yi, zi, levels=10, colors='red', alpha=0.6)
            ax3.clabel(contour, inline=True, fontsize=8)
        except:
            pass  # Skip contours if interpolation fails
            
        ax3.set_title(f'Phase Space Contours: Point Perplexity{cutoff_suffix}')
        ax3.set_xlabel('Momentum')
        ax3.set_ylabel('Point Perplexity')
        
        # 4. Combined visualization
        scatter = ax4.scatter(momentum, point_perp, c=dist_perp, cmap='viridis', s=20, alpha=0.6)
        ax4.set_title(f'Combined Phase Space: Color = Distribution Perplexity{cutoff_suffix}')
        ax4.set_xlabel('Momentum')
        ax4.set_ylabel('Point Perplexity')
        plt.colorbar(scatter, ax=ax4, label='Distribution Perplexity')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/density_and_flow_analysis{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _clean_suffix(self, suffix: str) -> str:
        """Clean suffix for filename"""
        return suffix.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '') 

    def plot_vector_fields(self, trajectories: List[Dict], output_dir: str, cutoff_suffix: str, perplexity_cutoff: Optional[float]) -> None:
        """Plot vector fields showing trajectory flow directions"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Calculate velocity vectors for each trajectory
        all_positions = []
        all_velocities_point = []
        all_velocities_dist = []
        
        for traj in trajectories:
            momentum = np.array(traj['momentum'][1:])
            point_perp = np.array(traj['point_perplexity'][1:])
            dist_perp = np.array(traj['distribution_perplexity'][1:])
            
            if perplexity_cutoff is not None:
                point_perp = np.clip(point_perp, None, perplexity_cutoff)
                dist_perp = np.clip(dist_perp, None, perplexity_cutoff)
            
            if len(momentum) > 1:
                # Calculate velocity vectors
                for i in range(len(momentum)-1):
                    pos = [momentum[i], point_perp[i]]
                    vel_point = [momentum[i+1] - momentum[i], point_perp[i+1] - point_perp[i]]
                    vel_dist = [momentum[i+1] - momentum[i], dist_perp[i+1] - dist_perp[i]]
                    
                    all_positions.append(pos)
                    all_velocities_point.append(vel_point)
                    all_velocities_dist.append(vel_dist)
        
        if all_positions:
            positions = np.array(all_positions)
            velocities_point = np.array(all_velocities_point)
            velocities_dist = np.array(all_velocities_dist)
            
            # Vector field for point perplexity
            ax1.quiver(positions[:, 0], positions[:, 1], 
                      velocities_point[:, 0], velocities_point[:, 1],
                      alpha=0.6, scale_units='xy', scale=1, width=0.003)
            ax1.set_title(f'Vector Field: Flow in Point Perplexity Phase Space{cutoff_suffix}')
            ax1.set_xlabel('Momentum')
            ax1.set_ylabel('Point Perplexity')
            ax1.grid(True, alpha=0.3)
            
            # Vector field for distribution perplexity (using same x-positions, different y)
            dist_positions = positions.copy()
            for i, traj in enumerate(trajectories):
                if i < len(dist_positions):
                    dist_perp = np.array(traj['distribution_perplexity'][1:])
                    if perplexity_cutoff is not None:
                        dist_perp = np.clip(dist_perp, None, perplexity_cutoff)
                    if len(dist_perp) > 0:
                        dist_positions[i, 1] = dist_perp[0]  # Use first distribution perplexity value
            
            ax2.quiver(dist_positions[:, 0], dist_positions[:, 1], 
                      velocities_dist[:, 0], velocities_dist[:, 1],
                      alpha=0.6, scale_units='xy', scale=1, width=0.003)
            ax2.set_title(f'Vector Field: Flow in Distribution Perplexity Phase Space{cutoff_suffix}')
            ax2.set_xlabel('Momentum')
            ax2.set_ylabel('Distribution Perplexity')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vector_fields{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attractor_analysis(self, momentum: np.ndarray, point_perp: np.ndarray, dist_perp: np.ndarray, output_dir: str, cutoff_suffix: str) -> None:
        """Plot attractor analysis and phase space regions"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Identify high-density regions (potential attractors)
        H1, xedges, yedges = np.histogram2d(momentum, point_perp, bins=30)
        high_density_threshold = np.percentile(H1[H1 > 0], 80)  # Top 20% of density
        
        # Plot density with attractor regions highlighted
        im1 = ax1.imshow(H1.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                        cmap='viridis', aspect='auto')
        
        # Highlight potential attractor regions
        for i in range(len(xedges)-1):
            for j in range(len(yedges)-1):
                if H1[i, j] > high_density_threshold:
                    rect = patches.Rectangle((xedges[i], yedges[j]), 
                                           xedges[i+1] - xedges[i], yedges[j+1] - yedges[j],
                                           linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
                    ax1.add_patch(rect)
        
        ax1.set_title(f'Attractor Regions: Point Perplexity{cutoff_suffix}')
        ax1.set_xlabel('Momentum')
        ax1.set_ylabel('Point Perplexity')
        plt.colorbar(im1, ax=ax1, label='Density')
        
        # 2. Same for distribution perplexity
        H2, xedges, yedges = np.histogram2d(momentum, dist_perp, bins=30)
        high_density_threshold2 = np.percentile(H2[H2 > 0], 80)
        
        im2 = ax2.imshow(H2.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                        cmap='plasma', aspect='auto')
        
        for i in range(len(xedges)-1):
            for j in range(len(yedges)-1):
                if H2[i, j] > high_density_threshold2:
                    rect = patches.Rectangle((xedges[i], yedges[j]), 
                                           xedges[i+1] - xedges[i], yedges[j+1] - yedges[j],
                                           linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
                    ax2.add_patch(rect)
        
        ax2.set_title(f'Attractor Regions: Distribution Perplexity{cutoff_suffix}')
        ax2.set_xlabel('Momentum')
        ax2.set_ylabel('Distribution Perplexity')
        plt.colorbar(im2, ax=ax2, label='Density')
        
        # 3. Phase space regions analysis
        # Divide phase space into quadrants and analyze
        momentum_median = np.median(momentum)
        point_perp_median = np.median(point_perp)
        
        # Define regions
        regions = {
            'Low Mom, Low Perp': (momentum < momentum_median) & (point_perp < point_perp_median),
            'High Mom, Low Perp': (momentum >= momentum_median) & (point_perp < point_perp_median),
            'Low Mom, High Perp': (momentum < momentum_median) & (point_perp >= point_perp_median),
            'High Mom, High Perp': (momentum >= momentum_median) & (point_perp >= point_perp_median)
        }
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (region_name, mask) in enumerate(regions.items()):
            ax3.scatter(momentum[mask], point_perp[mask], c=colors[i], alpha=0.6, s=10, label=region_name)
        
        ax3.axvline(momentum_median, color='black', linestyle='--', alpha=0.5)
        ax3.axhline(point_perp_median, color='black', linestyle='--', alpha=0.5)
        ax3.set_title(f'Phase Space Regions: Point Perplexity{cutoff_suffix}')
        ax3.set_xlabel('Momentum')
        ax3.set_ylabel('Point Perplexity')
        ax3.legend()
        
        # 4. Distribution of points in each region
        region_counts = [np.sum(mask) for mask in regions.values()]
        ax4.bar(range(len(regions)), region_counts, color=colors)
        ax4.set_title(f'Point Distribution by Phase Space Region{cutoff_suffix}')
        ax4.set_xlabel('Region')
        ax4.set_ylabel('Number of Points')
        ax4.set_xticks(range(len(regions)))
        ax4.set_xticklabels(list(regions.keys()), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/attractor_analysis{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_evolution(self, trajectories: List[Dict], output_dir: str, cutoff_suffix: str, perplexity_cutoff: Optional[float]) -> None:
        """Plot trajectories colored by temporal evolution (time steps)"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        for traj_idx, traj in enumerate(trajectories[:10]):  # Show first 10 trajectories
            momentum = np.array(traj['momentum'][1:])
            point_perp = np.array(traj['point_perplexity'][1:])
            dist_perp = np.array(traj['distribution_perplexity'][1:])
            
            if perplexity_cutoff is not None:
                point_perp = np.clip(point_perp, None, perplexity_cutoff)
                dist_perp = np.clip(dist_perp, None, perplexity_cutoff)
            
            if len(momentum) > 1:
                # Time steps as colors
                time_steps = np.arange(len(momentum))
                
                # 1. Point perplexity with temporal coloring
                scatter1 = ax1.scatter(momentum, point_perp, c=time_steps, cmap='viridis', 
                                     s=50, alpha=0.7, label=f'Traj {traj_idx+1}')
                ax1.plot(momentum, point_perp, '-', alpha=0.3, color='gray')
        
        ax1.set_title(f'Temporal Evolution: Point Perplexity{cutoff_suffix}')
        ax1.set_xlabel('Momentum')
        ax1.set_ylabel('Point Perplexity')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1, label='Time Step')
        
        # 2. Distribution perplexity with temporal coloring
        for traj_idx, traj in enumerate(trajectories[:10]):
            momentum = np.array(traj['momentum'][1:])
            dist_perp = np.array(traj['distribution_perplexity'][1:])
            
            if perplexity_cutoff is not None:
                dist_perp = np.clip(dist_perp, None, perplexity_cutoff)
            
            if len(momentum) > 1:
                time_steps = np.arange(len(momentum))
                scatter2 = ax2.scatter(momentum, dist_perp, c=time_steps, cmap='plasma', 
                                     s=50, alpha=0.7)
                ax2.plot(momentum, dist_perp, '-', alpha=0.3, color='gray')
        
        ax2.set_title(f'Temporal Evolution: Distribution Perplexity{cutoff_suffix}')
        ax2.set_xlabel('Momentum')
        ax2.set_ylabel('Distribution Perplexity')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Time Step')
        
        # 3. Time evolution of momentum vs perplexity correlation
        all_correlations = []
        trajectory_lengths = []
        
        for traj in trajectories:
            momentum = np.array(traj['momentum'][1:])
            point_perp = np.array(traj['point_perplexity'][1:])
            
            if len(momentum) > 2:
                corr = np.corrcoef(momentum, point_perp)[0, 1]
                all_correlations.append(corr)
                trajectory_lengths.append(len(momentum))
        
        if all_correlations:
            ax3.scatter(trajectory_lengths, all_correlations, alpha=0.6, s=50)
            ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax3.set_title(f'Momentum-Perplexity Correlation vs Trajectory Length{cutoff_suffix}')
            ax3.set_xlabel('Trajectory Length')
            ax3.set_ylabel('Momentum-Point Perplexity Correlation')
            ax3.grid(True, alpha=0.3)
        
        # 4. Average temporal evolution across all trajectories
        max_length = max(len(traj['momentum'][1:]) for traj in trajectories if len(traj['momentum']) > 1)
        avg_momentum = np.zeros(max_length)
        avg_point_perp = np.zeros(max_length)
        counts = np.zeros(max_length)
        
        for traj in trajectories:
            momentum = traj['momentum'][1:]
            point_perp = traj['point_perplexity'][1:]
            
            for i in range(min(len(momentum), max_length)):
                avg_momentum[i] += momentum[i]
                avg_point_perp[i] += point_perp[i]
                counts[i] += 1
        
        # Normalize by counts
        for i in range(max_length):
            if counts[i] > 0:
                avg_momentum[i] /= counts[i]
                avg_point_perp[i] /= counts[i]
        
        valid_steps = counts > 0
        time_steps = np.arange(max_length)[valid_steps]
        
        if len(time_steps) > 1:
            scatter4 = ax4.scatter(avg_momentum[valid_steps], avg_point_perp[valid_steps], 
                                 c=time_steps, cmap='coolwarm', s=100, alpha=0.8)
            ax4.plot(avg_momentum[valid_steps], avg_point_perp[valid_steps], '-', 
                    color='black', alpha=0.5, linewidth=2)
            
            # Add step numbers as annotations
            for i, step in enumerate(time_steps[::2]):  # Every other step
                ax4.annotate(f'{step}', (avg_momentum[step], avg_point_perp[step]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax4.set_title(f'Average Temporal Evolution Path{cutoff_suffix}')
            ax4.set_xlabel('Average Momentum')
            ax4.set_ylabel('Average Point Perplexity')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter4, ax=ax4, label='Time Step')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/temporal_evolution{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close() 

    def plot_token_type_analysis(self, trajectories: List[Dict], output_dir: str, cutoff_suffix: str, perplexity_cutoff: Optional[float]) -> None:
        """Plot phase space colored by token types"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Collect all data points by token type
        token_type_data = {tt: {'momentum': [], 'point_perp': [], 'dist_perp': []} 
                          for tt in self.token_type_colors.keys()}
        
        for traj in trajectories:
            momentum = traj['momentum'][1:]
            point_perp = traj['point_perplexity'][1:]
            dist_perp = traj['distribution_perplexity'][1:]
            token_types = traj['token_types'][1:]
            
            if perplexity_cutoff is not None:
                point_perp = np.clip(point_perp, None, perplexity_cutoff)
                dist_perp = np.clip(dist_perp, None, perplexity_cutoff)
            
            for i in range(min(len(momentum), len(token_types))):
                tt = token_types[i]
                if tt in token_type_data:
                    token_type_data[tt]['momentum'].append(momentum[i])
                    token_type_data[tt]['point_perp'].append(point_perp[i])
                    token_type_data[tt]['dist_perp'].append(dist_perp[i])
        
        # 1. Point perplexity by token type
        for token_type, data in token_type_data.items():
            if data['momentum']:  # Only plot if we have data
                ax1.scatter(data['momentum'], data['point_perp'], 
                          c=self.token_type_colors[token_type], alpha=0.6, s=30, label=token_type)
        
        ax1.set_title(f'Token Types in Phase Space: Point Perplexity{cutoff_suffix}')
        ax1.set_xlabel('Momentum')
        ax1.set_ylabel('Point Perplexity')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution perplexity by token type
        for token_type, data in token_type_data.items():
            if data['momentum']:
                ax2.scatter(data['momentum'], data['dist_perp'], 
                          c=self.token_type_colors[token_type], alpha=0.6, s=30, label=token_type)
        
        ax2.set_title(f'Token Types in Phase Space: Distribution Perplexity{cutoff_suffix}')
        ax2.set_xlabel('Momentum')
        ax2.set_ylabel('Distribution Perplexity')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Token type statistics
        token_type_stats = {}
        for token_type, data in token_type_data.items():
            if data['momentum']:
                token_type_stats[token_type] = {
                    'count': len(data['momentum']),
                    'avg_momentum': np.mean(data['momentum']),
                    'avg_point_perp': np.mean(data['point_perp']),
                    'avg_dist_perp': np.mean(data['dist_perp'])
                }
        
        # Bar plot of counts
        if token_type_stats:
            types = list(token_type_stats.keys())
            counts = [token_type_stats[tt]['count'] for tt in types]
            colors = [self.token_type_colors[tt] for tt in types]
            
            bars = ax3.bar(range(len(types)), counts, color=colors, alpha=0.7)
            ax3.set_title(f'Token Type Distribution{cutoff_suffix}')
            ax3.set_xlabel('Token Type')
            ax3.set_ylabel('Count')
            ax3.set_xticks(range(len(types)))
            ax3.set_xticklabels(types, rotation=45, ha='right')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
        
        # 4. Average momentum by token type
        if token_type_stats:
            avg_momentums = [token_type_stats[tt]['avg_momentum'] for tt in types]
            bars = ax4.bar(range(len(types)), avg_momentums, color=colors, alpha=0.7)
            ax4.set_title(f'Average Momentum by Token Type{cutoff_suffix}')
            ax4.set_xlabel('Token Type')
            ax4.set_ylabel('Average Momentum')
            ax4.set_xticks(range(len(types)))
            ax4.set_xticklabels(types, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, val in zip(bars, avg_momentums):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/token_type_analysis{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_velocity_acceleration_fields(self, trajectories: List[Dict], output_dir: str, cutoff_suffix: str, perplexity_cutoff: Optional[float]) -> None:
        """Plot velocity and acceleration vector fields in phase space"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Collect velocity and acceleration data
        all_positions = []
        all_velocities = []
        all_accelerations = []
        
        for traj in trajectories:
            momentum = traj['momentum'][1:]
            point_perp = traj['point_perplexity'][1:]
            velocities = traj['velocities'][1:]
            accelerations = traj['accelerations'][1:]
            
            if perplexity_cutoff is not None:
                point_perp = np.clip(point_perp, None, perplexity_cutoff)
            
            for i in range(min(len(momentum), len(velocities))):
                if i < len(momentum) and i < len(point_perp):
                    all_positions.append([momentum[i], point_perp[i]])
                    all_velocities.append(velocities[i])
                    if i < len(accelerations):
                        all_accelerations.append(accelerations[i])
        
        if all_positions and all_velocities:
            positions = np.array(all_positions)
            
            # 1. Velocity magnitude field
            velocity_magnitudes = [v['velocity_magnitude'] for v in all_velocities]
            scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], c=velocity_magnitudes, 
                                 cmap='viridis', s=50, alpha=0.7)
            ax1.set_title(f'Velocity Magnitude Field{cutoff_suffix}')
            ax1.set_xlabel('Momentum')
            ax1.set_ylabel('Point Perplexity')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter1, ax=ax1, label='Velocity Magnitude')
            
            # 2. Velocity direction arrows (subsample for clarity)
            step = max(1, len(positions) // 50)  # Show ~50 arrows
            pos_subset = positions[::step]
            vel_subset = all_velocities[::step]
            
            momentum_vels = [v['momentum_velocity'] for v in vel_subset]
            perp_vels = [v['point_perp_velocity'] for v in vel_subset]
            
            ax2.quiver(pos_subset[:, 0], pos_subset[:, 1], 
                      momentum_vels, perp_vels,
                      alpha=0.6, scale_units='xy', scale=1, width=0.003)
            ax2.scatter(positions[:, 0], positions[:, 1], c='lightblue', s=10, alpha=0.3)
            ax2.set_title(f'Velocity Vector Field{cutoff_suffix}')
            ax2.set_xlabel('Momentum')
            ax2.set_ylabel('Point Perplexity')
            ax2.grid(True, alpha=0.3)
            
            # 3. Acceleration magnitude field
            if all_accelerations:
                acc_magnitudes = [a['acceleration_magnitude'] for a in all_accelerations]
                scatter3 = ax3.scatter(positions[:len(acc_magnitudes), 0], 
                                     positions[:len(acc_magnitudes), 1], 
                                     c=acc_magnitudes, cmap='plasma', s=50, alpha=0.7)
                ax3.set_title(f'Acceleration Magnitude Field{cutoff_suffix}')
                ax3.set_xlabel('Momentum')
                ax3.set_ylabel('Point Perplexity')
                ax3.grid(True, alpha=0.3)
                plt.colorbar(scatter3, ax=ax3, label='Acceleration Magnitude')
            
            # 4. Phase space flow analysis
            # Show regions of high vs low velocity
            velocity_magnitudes = np.array(velocity_magnitudes)
            high_vel_threshold = np.percentile(velocity_magnitudes, 75)
            
            high_vel_mask = velocity_magnitudes > high_vel_threshold
            low_vel_mask = velocity_magnitudes <= high_vel_threshold
            
            ax4.scatter(positions[high_vel_mask, 0], positions[high_vel_mask, 1], 
                       c='red', s=30, alpha=0.7, label='High Velocity')
            ax4.scatter(positions[low_vel_mask, 0], positions[low_vel_mask, 1], 
                       c='blue', s=30, alpha=0.7, label='Low Velocity')
            
            ax4.set_title(f'Velocity Regions{cutoff_suffix}')
            ax4.set_xlabel('Momentum')
            ax4.set_ylabel('Point Perplexity')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/velocity_acceleration_fields{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_energy_analysis(self, trajectories: List[Dict], output_dir: str, cutoff_suffix: str, perplexity_cutoff: Optional[float]) -> None:
        """Plot energy-like measures in phase space"""
        
        # Create two separate figures: spatial energy fields and temporal energy evolution
        
        # FIGURE 1: Spatial Energy Fields in Phase Space
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Collect energy data
        all_momentum = []
        all_point_perp = []
        all_kinetic = []
        all_potential = []
        all_total_energy = []
        all_energy_ratio = []
        
        for traj in trajectories:
            momentum = traj['momentum'][1:]
            point_perp = traj['point_perplexity'][1:]
            energy_measures = traj['energy_measures'][1:]
            
            if perplexity_cutoff is not None:
                point_perp = np.clip(point_perp, None, perplexity_cutoff)
            
            for i in range(min(len(momentum), len(energy_measures))):
                all_momentum.append(momentum[i])
                all_point_perp.append(point_perp[i])
                all_kinetic.append(energy_measures[i]['kinetic_energy'])
                all_potential.append(energy_measures[i]['point_potential'])
                all_total_energy.append(energy_measures[i]['total_energy_point'])
                all_energy_ratio.append(energy_measures[i]['energy_ratio'])
        
        if all_momentum:
            all_momentum = np.array(all_momentum)
            all_point_perp = np.array(all_point_perp)
            all_kinetic = np.array(all_kinetic)
            all_potential = np.array(all_potential)
            all_total_energy = np.array(all_total_energy)
            all_energy_ratio = np.array(all_energy_ratio)
            
            # DEBUG: Check for kinetic energy calculation consistency
            print("🔍 DEBUGGING KINETIC ENERGY CALCULATION:")
            calculated_kinetic = (all_momentum ** 2) / 2.0
            kinetic_diff = np.abs(all_kinetic - calculated_kinetic)
            max_diff = np.max(kinetic_diff)
            print(f"  Max difference between stored and calculated kinetic energy: {max_diff}")
            
            if max_diff > 1e-10:
                print("  ❌ ERROR: Kinetic energy calculation inconsistency detected!")
                print("  First 10 problematic points:")
                problem_indices = np.where(kinetic_diff > 1e-10)[0][:10]
                for idx in problem_indices:
                    print(f"    Momentum: {all_momentum[idx]:.3f}, Stored: {all_kinetic[idx]:.3f}, Calculated: {calculated_kinetic[idx]:.3f}")
            else:
                print("  ✅ Kinetic energy calculation is consistent")
            
            # Check for duplicate momentum values with different kinetic energies
            print("🔍 CHECKING FOR MOMENTUM DUPLICATES:")
            unique_momentums, unique_indices = np.unique(all_momentum, return_index=True)
            if len(unique_momentums) < len(all_momentum):
                print(f"  Found {len(all_momentum) - len(unique_momentums)} duplicate momentum values")
                # Find duplicates
                for momentum_val in unique_momentums:
                    indices = np.where(np.abs(all_momentum - momentum_val) < 1e-10)[0]
                    if len(indices) > 1:
                        kinetic_vals = all_kinetic[indices]
                        if not np.allclose(kinetic_vals, kinetic_vals[0]):
                            print(f"    ❌ Momentum {momentum_val:.3f} has different kinetic energies: {kinetic_vals}")
            else:
                print("  ✅ No duplicate momentum values found")
            
            # 1. Kinetic energy (from momentum)
            scatter1 = ax1.scatter(all_momentum, all_point_perp, c=all_kinetic, 
                                 cmap='viridis', s=30, alpha=0.7)
            ax1.set_title(f'Kinetic Energy Field{cutoff_suffix}')
            ax1.set_xlabel('Momentum')
            ax1.set_ylabel('Point Perplexity')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter1, ax=ax1, label='Kinetic Energy')
            
            # 2. Potential energy (from perplexity)
            scatter2 = ax2.scatter(all_momentum, all_point_perp, c=all_potential, 
                                 cmap='plasma', s=30, alpha=0.7)
            ax2.set_title(f'Potential Energy Field{cutoff_suffix}')
            ax2.set_xlabel('Momentum')
            ax2.set_ylabel('Point Perplexity')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label='Potential Energy (log perplexity)')
            
            # 3. Total energy contours
            scatter3 = ax3.scatter(all_momentum, all_point_perp, c=all_total_energy, 
                                 cmap='coolwarm', s=30, alpha=0.7)
            ax3.set_title(f'Total Energy Field{cutoff_suffix}')
            ax3.set_xlabel('Momentum')
            ax3.set_ylabel('Point Perplexity')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter3, ax=ax3, label='Total Energy')
            
            # Add energy contour lines
            try:
                # Create grid for contour plot
                xi = np.linspace(all_momentum.min(), all_momentum.max(), 30)
                yi = np.linspace(all_point_perp.min(), all_point_perp.max(), 30)
                Xi, Yi = np.meshgrid(xi, yi)
                
                # Interpolate total energy onto grid
                Zi = griddata((all_momentum, all_point_perp), all_total_energy, 
                            (Xi, Yi), method='cubic')
                
                # Plot contour lines
                contour = ax3.contour(Xi, Yi, Zi, levels=8, colors='black', alpha=0.4, linewidths=0.5)
                ax3.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
            except:
                pass  # Skip contours if interpolation fails
            
            # 4. Energy ratio analysis (kinetic vs potential)
            scatter4 = ax4.scatter(all_momentum, all_point_perp, c=all_energy_ratio, 
                                 cmap='RdYlBu', s=30, alpha=0.7)
            ax4.set_title(f'Energy Ratio (Kinetic/Total){cutoff_suffix}')
            ax4.set_xlabel('Momentum')
            ax4.set_ylabel('Point Perplexity')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter4, ax=ax4, label='Kinetic Energy Ratio')
            
            # Add lines for equal energy
            momentum_range = np.linspace(all_momentum.min(), all_momentum.max(), 100)
            # Line where kinetic = potential (energy_ratio = 0.5)
            equal_energy_perp = np.exp(momentum_range**2 / 2)  # kinetic = potential
            valid_mask = equal_energy_perp <= all_point_perp.max()
            if np.any(valid_mask):
                ax4.plot(momentum_range[valid_mask], equal_energy_perp[valid_mask], 
                        'k--', alpha=0.7, linewidth=2, label='Kinetic = Potential')
                ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_spatial_fields{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # FIGURE 2: Temporal Energy Evolution
        fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Individual trajectory energy evolution (show first 6 trajectories)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, traj in enumerate(trajectories[:6]):
            energy_measures = traj['energy_measures']
            steps = range(len(energy_measures))
            
            kinetic_energies = [em['kinetic_energy'] for em in energy_measures]
            potential_energies = [em['point_potential'] for em in energy_measures]
            total_energies = [em['total_energy_point'] for em in energy_measures]
            
            # Use log scale for kinetic energy to better see patterns
            log_kinetic_energies = [np.log(max(ke, 1e-8)) for ke in kinetic_energies]
            
            ax5.plot(steps, log_kinetic_energies, '-', color=colors[i], alpha=0.7, linewidth=2, 
                    label=f'Traj {i+1} Log Kinetic')
            ax5.plot(steps, potential_energies, '--', color=colors[i], alpha=0.7, linewidth=2, 
                    label=f'Traj {i+1} Potential')
        
        ax5.set_title(f'Individual Trajectory Energy Evolution (Log Kinetic){cutoff_suffix}')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Energy (Log scale for kinetic)')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 2. Total energy evolution for individual trajectories
        for i, traj in enumerate(trajectories[:6]):
            energy_measures = traj['energy_measures']
            steps = range(len(energy_measures))
            total_energies = [em['total_energy_point'] for em in energy_measures]
            
            ax6.plot(steps, total_energies, '-', color=colors[i], alpha=0.8, linewidth=2, 
                    label=f'Trajectory {i+1}')
        
        ax6.set_title(f'Total Energy Evolution{cutoff_suffix}')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Total Energy')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 3. Average energy evolution across all trajectories
        max_steps = max(len(traj['energy_measures']) for traj in trajectories)
        avg_kinetic = np.zeros(max_steps)
        avg_potential = np.zeros(max_steps)
        avg_total = np.zeros(max_steps)
        avg_ratio = np.zeros(max_steps)
        step_counts = np.zeros(max_steps)
        
        for traj in trajectories:
            energy_measures = traj['energy_measures']
            for step, em in enumerate(energy_measures):
                if step < max_steps:
                    avg_kinetic[step] += em['kinetic_energy']
                    avg_potential[step] += em['point_potential']
                    avg_total[step] += em['total_energy_point']
                    avg_ratio[step] += em['energy_ratio']
                    step_counts[step] += 1
        
        # Normalize by counts
        for step in range(max_steps):
            if step_counts[step] > 0:
                avg_kinetic[step] /= step_counts[step]
                avg_potential[step] /= step_counts[step] 
                avg_total[step] /= step_counts[step]
                avg_ratio[step] /= step_counts[step]
        
        valid_steps = step_counts > 0
        step_range = np.arange(max_steps)[valid_steps]
        
        if len(step_range) > 1:
            # Use log scale for average kinetic energy
            log_avg_kinetic = np.log(np.maximum(avg_kinetic[valid_steps], 1e-8))
            
            ax7.plot(step_range, log_avg_kinetic, 'b-', linewidth=3, label='Average Log Kinetic')
            ax7.plot(step_range, avg_potential[valid_steps], 'r-', linewidth=3, label='Average Potential')
            ax7.plot(step_range, avg_total[valid_steps], 'g-', linewidth=3, label='Average Total')
            
            # Add error bars showing standard deviation
            std_kinetic = np.zeros(max_steps)
            std_potential = np.zeros(max_steps)
            std_total = np.zeros(max_steps)
            
            for traj in trajectories:
                energy_measures = traj['energy_measures']
                for step, em in enumerate(energy_measures):
                    if step < max_steps and step_counts[step] > 0:
                        std_kinetic[step] += (em['kinetic_energy'] - avg_kinetic[step])**2
                        std_potential[step] += (em['point_potential'] - avg_potential[step])**2
                        std_total[step] += (em['total_energy_point'] - avg_total[step])**2
            
            for step in range(max_steps):
                if step_counts[step] > 1:
                    std_kinetic[step] = np.sqrt(std_kinetic[step] / (step_counts[step] - 1))
                    std_potential[step] = np.sqrt(std_potential[step] / (step_counts[step] - 1))
                    std_total[step] = np.sqrt(std_total[step] / (step_counts[step] - 1))
            
            # For log kinetic energy, compute error bars in log space
            log_kinetic_upper = np.log(np.maximum(avg_kinetic[valid_steps] + std_kinetic[valid_steps], 1e-8))
            log_kinetic_lower = np.log(np.maximum(avg_kinetic[valid_steps] - std_kinetic[valid_steps], 1e-8))
            
            ax7.fill_between(step_range, log_kinetic_lower, log_kinetic_upper, 
                           alpha=0.2, color='blue')
            ax7.fill_between(step_range, 
                           (avg_potential[valid_steps] - std_potential[valid_steps]), 
                           (avg_potential[valid_steps] + std_potential[valid_steps]), 
                           alpha=0.2, color='red')
            
            ax7.set_title(f'Average Energy Evolution with Standard Deviation (Log Kinetic){cutoff_suffix}')
            ax7.set_xlabel('Time Step')
            ax7.set_ylabel('Energy (Log scale for kinetic)')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 4. Energy ratio evolution (kinetic vs potential dominance)
        if len(step_range) > 1:
            ax8.plot(step_range, avg_ratio[valid_steps], 'purple', linewidth=3, label='Kinetic/Total Ratio')
            ax8.axhline(0.5, color='black', linestyle='--', alpha=0.7, label='Equal Energy Line')
            
            # Shade regions
            ax8.fill_between(step_range, 0, avg_ratio[valid_steps], 
                           where=(avg_ratio[valid_steps] < 0.5), 
                           alpha=0.3, color='red', label='Potential Dominated')
            ax8.fill_between(step_range, avg_ratio[valid_steps], 1, 
                           where=(avg_ratio[valid_steps] > 0.5), 
                           alpha=0.3, color='blue', label='Kinetic Dominated')
            
            ax8.set_title(f'Energy Balance Evolution{cutoff_suffix}')
            ax8.set_xlabel('Time Step')
            ax8.set_ylabel('Kinetic Energy Ratio')
            ax8.set_ylim(0, 1)
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_temporal_evolution{self._clean_suffix(cutoff_suffix)}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close() 

    def plot_average_dynamics(self, trajectories: List[Dict], output_dir: str, cutoff_suffix: str, perplexity_cutoff: Optional[float]) -> None:
        """Plot average perplexity for each momentum value across all sequences"""
        
        # Collect all momentum and perplexity data points
        all_momentum = []
        all_point_perp = []
        all_dist_perp = []
        
        for traj in trajectories:
            momentum = traj['momentum'][1:]  # Skip first step (momentum=0)
            point_perp = traj['point_perplexity'][1:]
            dist_perp = traj['distribution_perplexity'][1:]
            
            for i in range(len(momentum)):
                mom = momentum[i]
                pp = point_perp[i]
                dp = dist_perp[i]
                
                if perplexity_cutoff is not None:
                    pp = min(pp, perplexity_cutoff)
                    dp = min(dp, perplexity_cutoff)
                
                all_momentum.append(mom)
                all_point_perp.append(pp)
                all_dist_perp.append(dp)
        
        if not all_momentum:
            print("  ⚠️ No data points for average dynamics plot")
            return
        
        # Convert to numpy arrays
        all_momentum = np.array(all_momentum)
        all_point_perp = np.array(all_point_perp)
        all_dist_perp = np.array(all_dist_perp)
        
        # Create momentum bins
        num_bins = 30
        momentum_min = np.min(all_momentum)
        momentum_max = np.max(all_momentum)
        momentum_bins = np.linspace(momentum_min, momentum_max, num_bins + 1)
        momentum_centers = (momentum_bins[:-1] + momentum_bins[1:]) / 2
        
        # Calculate average perplexity for each momentum bin
        avg_point_perp_by_momentum = []
        avg_dist_perp_by_momentum = []
        std_point_perp_by_momentum = []
        std_dist_perp_by_momentum = []
        bin_counts = []
        
        for i in range(num_bins):
            # Find data points in this momentum bin
            bin_mask = (all_momentum >= momentum_bins[i]) & (all_momentum < momentum_bins[i + 1])
            
            if i == num_bins - 1:  # Include the maximum value in the last bin
                bin_mask = (all_momentum >= momentum_bins[i]) & (all_momentum <= momentum_bins[i + 1])
            
            bin_point_perp = all_point_perp[bin_mask]
            bin_dist_perp = all_dist_perp[bin_mask]
            
            if len(bin_point_perp) > 0:
                avg_point_perp_by_momentum.append(np.mean(bin_point_perp))
                avg_dist_perp_by_momentum.append(np.mean(bin_dist_perp))
                std_point_perp_by_momentum.append(np.std(bin_point_perp))
                std_dist_perp_by_momentum.append(np.std(bin_dist_perp))
                bin_counts.append(len(bin_point_perp))
            else:
                avg_point_perp_by_momentum.append(np.nan)
                avg_dist_perp_by_momentum.append(np.nan)
                std_point_perp_by_momentum.append(np.nan)
                std_dist_perp_by_momentum.append(np.nan)
                bin_counts.append(0)
        
        # Filter out bins with no data
        valid_bins = np.array(bin_counts) > 0
        momentum_centers_valid = momentum_centers[valid_bins]
        avg_point_perp_valid = np.array(avg_point_perp_by_momentum)[valid_bins]
        avg_dist_perp_valid = np.array(avg_dist_perp_by_momentum)[valid_bins]
        std_point_perp_valid = np.array(std_point_perp_by_momentum)[valid_bins]
        std_dist_perp_valid = np.array(std_dist_perp_by_momentum)[valid_bins]
        bin_counts_valid = np.array(bin_counts)[valid_bins]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        if len(momentum_centers_valid) > 0:
            # Plot 1: Average Point Perplexity vs Momentum
            ax1.plot(momentum_centers_valid, avg_point_perp_valid, 'r-', linewidth=3, 
                    label='Average Point Perplexity', marker='o', markersize=4)
            ax1.fill_between(momentum_centers_valid, 
                            avg_point_perp_valid - std_point_perp_valid,
                            avg_point_perp_valid + std_point_perp_valid,
                            alpha=0.3, color='red', label='±1 Std Dev')
            
            ax1.set_xlabel('Momentum')
            ax1.set_ylabel('Average Point Perplexity')
            ax1.set_title('Average Point Perplexity for Each Momentum Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add a second y-axis for bin counts
            ax1_twin = ax1.twinx()
            ax1_twin.bar(momentum_centers_valid, bin_counts_valid, alpha=0.2, color='gray', width=(momentum_centers_valid[1] - momentum_centers_valid[0]) * 0.8, label='Data Point Count')
            ax1_twin.set_ylabel('Number of Data Points per Bin', color='gray')
            ax1_twin.tick_params(axis='y', labelcolor='gray')
            
            # Plot 2: Average Distribution Perplexity vs Momentum
            ax2.plot(momentum_centers_valid, avg_dist_perp_valid, 'g-', linewidth=3, 
                    label='Average Distribution Perplexity', marker='o', markersize=4)
            ax2.fill_between(momentum_centers_valid, 
                            avg_dist_perp_valid - std_dist_perp_valid,
                            avg_dist_perp_valid + std_dist_perp_valid,
                            alpha=0.3, color='green', label='±1 Std Dev')
            
            ax2.set_xlabel('Momentum')
            ax2.set_ylabel('Average Distribution Perplexity')
            ax2.set_title('Average Distribution Perplexity for Each Momentum Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add a second y-axis for bin counts
            ax2_twin = ax2.twinx()
            ax2_twin.bar(momentum_centers_valid, bin_counts_valid, alpha=0.2, color='gray', width=(momentum_centers_valid[1] - momentum_centers_valid[0]) * 0.8, label='Data Point Count')
            ax2_twin.set_ylabel('Number of Data Points per Bin', color='gray')
            ax2_twin.tick_params(axis='y', labelcolor='gray')
            
            # Calculate and display correlation
            point_corr = np.corrcoef(momentum_centers_valid, avg_point_perp_valid)[0,1]
            dist_corr = np.corrcoef(momentum_centers_valid, avg_dist_perp_valid)[0,1]
            
            ax1.text(0.05, 0.95, f'Correlation: {point_corr:.3f}', 
                    transform=ax1.transAxes, fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax2.text(0.05, 0.95, f'Correlation: {dist_corr:.3f}', 
                    transform=ax2.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            print(f"  📊 Momentum-Perplexity Correlations:")
            print(f"    Point Perplexity: {point_corr:.3f}")
            print(f"    Distribution Perplexity: {dist_corr:.3f}")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/average_perplexity_by_momentum{cutoff_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Average perplexity by momentum plot saved") 