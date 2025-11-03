"""
Plot Omega Results

Visualization script for omega sweep results.
Reads timestamped analysis results and generates publication-quality plots.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class OmegaResultsPlotter:
    """Plots results from omega parameter sweep."""

    def __init__(self, run_dir: Path):
        """
        Initialize plotter with run directory.

        Args:
            run_dir: Path to timestamped run directory (e.g., results/run_20250129_143022)
        """
        self.run_dir = Path(run_dir)
        self.summary_file = self.run_dir / "summary.json"
        self.summary_data = None
        self.omega_values = []
        self.flow_rates = []
        self.flow_rate_errors = []
        self.r_squared_values = []

    def load_summary(self) -> None:
        """Load summary JSON file."""
        if not self.summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {self.summary_file}")

        with open(self.summary_file, 'r') as f:
            self.summary_data = json.load(f)

        print(f"\nLoaded results from: {self.run_dir}")
        print(f"Timestamp: {self.summary_data['timestamp']}")
        print(f"Parameters: {self.summary_data['parameters']}\n")

    def extract_results(self) -> None:
        """Extract flow rate data from summary."""
        results = self.summary_data['results']

        omega_data = []
        for omega_str, result in results.items():
            if 'error' in result:
                print(f"Warning: Skipping ω = {omega_str} (analysis failed)")
                continue

            omega = float(omega_str)

            # Handle both binned and linear analysis results
            if 'flow_rate_mean' in result:
                # Binned analysis
                q = result['flow_rate_mean']
                q_err = result['flow_rate_std']
                r2 = None  # Not applicable for binned analysis
            else:
                # Linear analysis
                q = result['flow_rate']
                q_err = result['flow_rate_std_err']
                r2 = result['r_squared']

            omega_data.append((omega, q, q_err, r2))

        # Sort by omega
        omega_data.sort(key=lambda x: x[0])

        self.omega_values = np.array([x[0] for x in omega_data])
        self.flow_rates = np.array([x[1] for x in omega_data])
        self.flow_rate_errors = np.array([x[2] for x in omega_data])
        self.r_squared_values = np.array([x[3] if x[3] is not None else np.nan for x in omega_data])

    def plot_discharge_curves(self, output_file: Optional[Path] = None) -> None:
        """
        Plot all discharge curves on one figure.

        Args:
            output_file: Path to save figure (if None, save to run_dir)
        """
        if output_file is None:
            output_file = self.run_dir / "combined_discharge_curves.png"

        fig, ax = plt.subplots(figsize=(12, 7))

        # Load and plot each omega's discharge curve
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.omega_values)))

        for i, omega in enumerate(self.omega_values):
            omega_dir = self.run_dir / f"omega_{int(omega)}"
            analysis_file = omega_dir / "analysis.json"

            if not analysis_file.exists():
                print(f"Warning: No analysis file for ω = {omega}")
                continue

            with open(analysis_file, 'r') as f:
                data = json.load(f)

            exit_times = np.array(data['exit_times'])
            cumulative_count = np.array(data['cumulative_count'])

            ax.plot(exit_times, cumulative_count, 'o-',
                   color=colors[i], markersize=3, alpha=0.7,
                   label=f'$\\omega$ = {omega:.0f} s⁻¹')

        ax.set_xlabel('Tiempo (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Partículas acumuladas', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Discharge curves plot saved to: {output_file}")
        plt.close()

    def plot_flow_rate_vs_omega(self, output_file: Optional[Path] = None) -> None:
        """
        Plot flow rate Q vs omega with error bars.

        Args:
            output_file: Path to save figure (if None, save to run_dir)
        """
        if output_file is None:
            output_file = self.run_dir / "flow_rate_vs_omega.png"

        # Check if we have R² data (linear analysis) or not (binned analysis)
        has_r_squared = not np.all(np.isnan(self.r_squared_values))

        if has_r_squared:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))

        # Main plot: Q vs omega
        ax1.errorbar(self.omega_values, self.flow_rates, yerr=self.flow_rate_errors,
                    fmt='o-', markersize=8, capsize=5, capthick=2,
                    linewidth=2, elinewidth=2, color='navy', alpha=0.7)

        ax1.set_xlabel(r'$\omega$ (s$^{-1}$)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Caudal Q (partículas/s)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # R² subplot (only for linear analysis)
        if has_r_squared:
            ax2.plot(self.omega_values, self.r_squared_values, 's-',
                    markersize=7, linewidth=2, color='darkgreen', alpha=0.7)
            ax2.set_xlabel('Vibration frequency ω (s⁻¹)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('R² (goodness of fit)', fontsize=12, fontweight='bold')
            ax2.set_title('Linear Fit Quality', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim([0.9, 1.0])

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Flow rate vs omega plot saved to: {output_file}")
        plt.close()

    def plot_all(self) -> None:
        """Generate all plots."""
        if len(self.omega_values) == 0:
            print("\nNo valid results to plot. Skipping plot generation.\n")
            return

        print("\n" + "="*70)
        print("Generating plots...")
        print("="*70 + "\n")

        self.plot_discharge_curves()
        self.plot_flow_rate_vs_omega()

        print("\n" + "="*70)
        print("All plots generated successfully!")
        print("="*70 + "\n")

    def print_summary(self) -> None:
        """Print results summary."""
        print("\n" + "="*70)
        print("FLOW RATE RESULTS")
        print("="*70)

        # Check if we have R² data
        has_r_squared = not np.all(np.isnan(self.r_squared_values))

        if has_r_squared:
            print(f"{'Omega (s⁻¹)':<15} {'Flow Rate (p/s)':<25} {'R²':<10}")
        else:
            print(f"{'Omega (s⁻¹)':<15} {'Flow Rate (p/s)':<25}")

        print("-"*70)

        if len(self.omega_values) == 0:
            print("No valid results to display.")
            print("-"*70)
            print("="*70 + "\n")
            return

        for i, omega in enumerate(self.omega_values):
            q = self.flow_rates[i]
            q_err = self.flow_rate_errors[i]

            if has_r_squared:
                r2 = self.r_squared_values[i]
                print(f"{omega:<15.1f} {q:.4f} ± {q_err:.4f}         {r2:.4f}")
            else:
                print(f"{omega:<15.1f} {q:.4f} ± {q_err:.4f}")

        print("-"*70)

        # Find maximum
        max_idx = np.argmax(self.flow_rates)
        print(f"\nMaximum flow rate:")
        print(f"  ω = {self.omega_values[max_idx]:.1f} s⁻¹")
        print(f"  Q = {self.flow_rates[max_idx]:.4f} ± {self.flow_rate_errors[max_idx]:.4f} particles/s")
        print("="*70 + "\n")


def find_latest_run(results_dir: Path) -> Optional[Path]:
    """
    Find the most recent run directory.

    Args:
        results_dir: Base results directory

    Returns:
        Path to latest run directory, or None if none found
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        return None

    run_dirs = [d for d in results_dir.iterdir()
                if d.is_dir() and d.name.startswith("run_")]

    if not run_dirs:
        return None

    # Sort by timestamp (directory name includes timestamp)
    run_dirs.sort(reverse=True)
    return run_dirs[0]


def find_project_root() -> Path:
    """Find project root by looking for pom.xml."""
    current = Path.cwd().resolve()

    # Check current directory and parents
    for path in [current] + list(current.parents):
        if (path / "pom.xml").exists():
            return path

    # If not found, assume current directory
    print("Warning: Could not find pom.xml, using current directory as project root")
    return current


def main():
    """Main entry point for plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot omega sweep results from timestamped runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot latest run
  python plot_omega_results.py

  # Plot specific run by timestamp
  python plot_omega_results.py --timestamp 20250129_143022

  # Specify custom results directory
  python plot_omega_results.py --results-dir ../results
        """
    )

    parser.add_argument(
        '--timestamp',
        type=str,
        help='Timestamp of run to plot (e.g., 20250129_143022). If not provided, uses latest run.'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='src/main/python/results',
        help='Base results directory relative to project root (default: src/main/python/results)'
    )

    args = parser.parse_args()

    # Find project root and resolve results directory
    project_root = find_project_root()
    results_dir = project_root / args.results_dir

    # Determine which run to plot
    if args.timestamp:
        run_dir = results_dir / f"run_{args.timestamp}"
        if not run_dir.exists():
            print(f"ERROR: Run directory not found: {run_dir}")
            print(f"\nAvailable runs in {results_dir}:")
            for d in sorted(results_dir.glob("run_*"), reverse=True):
                print(f"  - {d.name}")
            return 1
    else:
        run_dir = find_latest_run(results_dir)
        if run_dir is None:
            print(f"ERROR: No run directories found in {results_dir}")
            return 1
        print(f"Using latest run: {run_dir.name}")

    # Create plotter and generate plots
    try:
        plotter = OmegaResultsPlotter(run_dir)
        plotter.load_summary()
        plotter.extract_results()
        plotter.print_summary()
        plotter.plot_all()

        print(f"\nResults plotted from: {run_dir}")

        return 0

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
