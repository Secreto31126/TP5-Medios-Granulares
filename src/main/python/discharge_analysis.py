"""
Discharge Analysis Module

Analyzes particle discharge data from granular media simulations.
Parses exited.txt files, generates discharge curves, and calculates flow rates.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import json
from typing import Tuple, Dict, Optional


class DischargeAnalyzer:
    """Analyzes discharge data from granular media simulations."""

    def __init__(self, exited_file: Path, transient_cutoff: float = 20.0):
        """
        Initialize analyzer with exit data.

        Args:
            exited_file: Path to exited.txt file
            transient_cutoff: Time (seconds) to skip before calculating steady-state flow rate
        """
        self.exited_file = Path(exited_file)
        self.transient_cutoff = transient_cutoff
        self.exit_times = None
        self.cumulative_count = None
        self.flow_rate = None
        self.flow_rate_std_err = None
        self.r_squared = None

    def parse_exit_data(self) -> None:
        """
        Parse exited.txt file and extract exit times.

        File format: "time count" where count is the number of particles
        that exited at that specific time step.
        """
        exit_times = []

        with open(self.exited_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "time count"
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        time = float(parts[0])
                        count = int(parts[1])
                        # Expand: add 'count' particles exiting at this time
                        exit_times.extend([time] * count)
                    except ValueError:
                        continue

        if not exit_times:
            raise ValueError(f"No valid exit data found in {self.exited_file}")

        self.exit_times = np.array(sorted(exit_times))
        self.cumulative_count = np.arange(1, len(self.exit_times) + 1)

    def calculate_flow_rate(self) -> Tuple[float, float, float]:
        """
        Calculate flow rate from linear regression on steady-state region.

        Returns:
            (flow_rate, std_error, r_squared): Flow rate (particles/s), standard error, R² value
        """
        if self.exit_times is None:
            self.parse_exit_data()

        # Filter to steady-state region (after transient)
        mask = self.exit_times >= self.transient_cutoff

        if not np.any(mask):
            # If no data after transient, use all data
            mask = np.ones_like(self.exit_times, dtype=bool)
            print(f"Warning: No data after transient cutoff {self.transient_cutoff}s. Using all data.")

        t_steady = self.exit_times[mask]
        n_steady = self.cumulative_count[mask]

        if len(t_steady) < 2:
            raise ValueError("Insufficient data points for flow rate calculation")

        # Linear regression: N = Q*t + b
        slope, intercept, r_value, p_value, std_err = stats.linregress(t_steady, n_steady)

        self.flow_rate = slope
        self.flow_rate_std_err = std_err
        self.r_squared = r_value ** 2

        return self.flow_rate, self.flow_rate_std_err, self.r_squared

    def plot_discharge_curve(self, output_file: Optional[Path] = None,
                            omega: Optional[float] = None) -> None:
        """
        Plot discharge curve with flow rate fit.

        Args:
            output_file: Path to save plot (if None, display only)
            omega: Frequency value for title (optional)
        """
        if self.exit_times is None:
            self.parse_exit_data()

        if self.flow_rate is None:
            self.calculate_flow_rate()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot raw discharge curve
        ax.plot(self.exit_times, self.cumulative_count, 'o',
                markersize=4, alpha=0.6, label='Discharge data')

        # Plot steady-state fit
        mask = self.exit_times >= self.transient_cutoff
        if np.any(mask):
            t_fit = self.exit_times[mask]
            n_fit = self.flow_rate * t_fit + (self.cumulative_count[mask][0] - self.flow_rate * t_fit[0])
            ax.plot(t_fit, n_fit, 'r-', linewidth=2,
                   label=f'Fit: Q = {self.flow_rate:.3f} ± {self.flow_rate_std_err:.3f} part/s\n' +
                         f'R² = {self.r_squared:.4f}')

        # Mark transient cutoff
        ax.axvline(self.transient_cutoff, color='gray', linestyle='--',
                  alpha=0.5, label=f'Transient cutoff ({self.transient_cutoff}s)')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Cumulative particles exited', fontsize=12)

        title = 'Discharge Curve'
        if omega is not None:
            title += f' (ω = {omega} s⁻¹)'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Discharge curve saved to {output_file}")
        else:
            plt.show()

        plt.close()

    def save_results(self, output_file: Path, omega: Optional[float] = None) -> None:
        """
        Save analysis results to JSON file.

        Args:
            output_file: Path to save JSON results
            omega: Frequency value (optional, for metadata)
        """
        if self.flow_rate is None:
            self.calculate_flow_rate()

        results = {
            'omega': omega,
            'flow_rate': float(self.flow_rate),
            'flow_rate_std_err': float(self.flow_rate_std_err),
            'r_squared': float(self.r_squared),
            'transient_cutoff': float(self.transient_cutoff),
            'total_particles_exited': int(len(self.exit_times)),
            'total_time': float(self.exit_times[-1]) if len(self.exit_times) > 0 else 0.0,
            'exit_times': self.exit_times.tolist(),
            'cumulative_count': self.cumulative_count.tolist()
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Analysis results saved to {output_file}")

    def get_summary(self) -> Dict:
        """Get summary statistics as dictionary."""
        if self.flow_rate is None:
            self.calculate_flow_rate()

        return {
            'flow_rate': self.flow_rate,
            'flow_rate_std_err': self.flow_rate_std_err,
            'r_squared': self.r_squared,
            'total_particles': len(self.exit_times) if self.exit_times is not None else 0
        }


def analyze_discharge(exited_file: Path, output_dir: Path,
                     omega: Optional[float] = None,
                     transient_cutoff: float = 20.0) -> Dict:
    """
    Complete discharge analysis workflow.

    Args:
        exited_file: Path to exited.txt
        output_dir: Directory to save results
        omega: Frequency value (for labeling)
        transient_cutoff: Time to skip before steady-state analysis

    Returns:
        Dictionary with analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = DischargeAnalyzer(exited_file, transient_cutoff=transient_cutoff)

    # Parse and calculate
    analyzer.parse_exit_data()
    analyzer.calculate_flow_rate()

    # Save results
    analyzer.save_results(output_dir / 'analysis.json', omega=omega)

    # Plot discharge curve
    analyzer.plot_discharge_curve(output_dir / 'discharge_curve.png', omega=omega)

    # Print summary
    summary = analyzer.get_summary()
    print(f"\n{'='*60}")
    print(f"Discharge Analysis Summary" + (f" (ω = {omega} s⁻¹)" if omega else ""))
    print(f"{'='*60}")
    print(f"Flow rate (Q):     {summary['flow_rate']:.4f} ± {summary['flow_rate_std_err']:.4f} particles/s")
    print(f"R² (goodness):     {summary['r_squared']:.4f}")
    print(f"Total particles:   {summary['total_particles']}")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python discharge_analysis.py <exited_file> [output_dir] [omega] [transient_cutoff]")
        print("Example: python discharge_analysis.py exited.txt ./results 450 20.0")
        sys.exit(1)

    exited_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./analysis_output")
    omega = float(sys.argv[3]) if len(sys.argv) > 3 else None
    transient_cutoff = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0

    analyze_discharge(exited_file, output_dir, omega, transient_cutoff)
