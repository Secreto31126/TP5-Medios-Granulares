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

    def __init__(self, exited_file: Path, transient_cutoff: float = 20.0,
                 method: str = "linear", bin_size: float = 20.0):
        """
        Initialize analyzer with exit data.

        Args:
            exited_file: Path to exited.txt file
            transient_cutoff: Time (seconds) to skip before calculating steady-state flow rate
            method: Analysis method - "linear" for linear regression or "binned" for bin-based
            bin_size: Size of time bins in seconds (only used for binned method)
        """
        self.exited_file = Path(exited_file)
        self.transient_cutoff = transient_cutoff
        self.method = method
        self.bin_size = bin_size
        self.exit_times = None
        self.cumulative_count = None
        self.flow_rate = None
        self.flow_rate_std_err = None
        self.r_squared = None
        # Binned method attributes
        self.flow_rate_mean = None
        self.flow_rate_std = None
        self.bin_rates = None
        self.bin_centers = None

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

    def calculate_flow_rate_binned(self) -> Tuple[float, float, np.ndarray]:
        """
        Calculate flow rate using non-overlapping time bins.

        Returns:
            (flow_rate_mean, flow_rate_std, bin_rates): Mean flow rate (particles/s),
                                                         standard deviation, and array of per-bin rates
        """
        if self.exit_times is None:
            self.parse_exit_data()

        # Filter to steady-state region (after transient)
        mask = self.exit_times >= self.transient_cutoff

        if not np.any(mask):
            # If no data after transient, use all data
            mask = np.ones_like(self.exit_times, dtype=bool)
            print(f"Warning: No data after transient cutoff {self.transient_cutoff}s. Using all data.")

        steady_exit_times = self.exit_times[mask]

        if len(steady_exit_times) == 0:
            raise ValueError("No exit data available for binned analysis")

        # Determine time range for binning
        start_time = self.transient_cutoff
        end_time = self.exit_times[-1]

        # Create bins
        bin_edges = np.arange(start_time, end_time + self.bin_size, self.bin_size)

        if len(bin_edges) < 2:
            raise ValueError(f"Insufficient time range for binning. Need at least {self.bin_size}s after transient.")

        # Calculate flow rate for each bin
        bin_rates = []
        bin_centers = []

        for i in range(len(bin_edges) - 1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]

            # Count particles in this bin
            particles_in_bin = np.sum((steady_exit_times >= bin_start) & (steady_exit_times < bin_end))

            # Calculate flow rate for this bin
            bin_duration = bin_end - bin_start
            flow_rate_bin = particles_in_bin / bin_duration

            bin_rates.append(flow_rate_bin)
            bin_centers.append((bin_start + bin_end) / 2)

        bin_rates = np.array(bin_rates)
        bin_centers = np.array(bin_centers)

        # Calculate mean and standard deviation
        self.flow_rate_mean = np.mean(bin_rates)
        self.flow_rate_std = np.std(bin_rates, ddof=1) if len(bin_rates) > 1 else 0.0
        self.bin_rates = bin_rates
        self.bin_centers = bin_centers

        return self.flow_rate_mean, self.flow_rate_std, bin_rates

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

        if self.method == "linear":
            if self.flow_rate is None:
                self.calculate_flow_rate()
            self._plot_linear_fit(output_file, omega)
        elif self.method == "binned":
            if self.flow_rate_mean is None:
                self.calculate_flow_rate_binned()
            self._plot_binned_analysis(output_file, omega)
        else:
            raise ValueError(f"Unknown analysis method: {self.method}")

    def _plot_linear_fit(self, output_file: Optional[Path], omega: Optional[float]) -> None:
        """Plot discharge curve with linear regression fit."""
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

        title = 'Discharge Curve (Linear Fit)'
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

    def _plot_binned_analysis(self, output_file: Optional[Path], omega: Optional[float]) -> None:
        """Plot discharge curve with binned flow rate analysis."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Top plot: Raw discharge curve with bin boundaries
        ax1.plot(self.exit_times, self.cumulative_count, 'o',
                markersize=4, alpha=0.6, label='Discharge data')

        # Mark transient cutoff
        ax1.axvline(self.transient_cutoff, color='gray', linestyle='--',
                   alpha=0.5, label=f'Transient cutoff ({self.transient_cutoff}s)')

        # Mark bin boundaries
        start_time = self.transient_cutoff
        end_time = self.exit_times[-1]
        bin_edges = np.arange(start_time, end_time + self.bin_size, self.bin_size)
        for edge in bin_edges[1:-1]:
            ax1.axvline(edge, color='lightblue', linestyle=':', alpha=0.5)

        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Cumulative particles exited', fontsize=12)

        title = 'Discharge Curve (Binned Analysis)'
        if omega is not None:
            title += f' (ω = {omega} s⁻¹)'
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Flow rate per bin
        ax2.plot(self.bin_centers, self.bin_rates, 'o-', markersize=8,
                label=f'Q = {self.flow_rate_mean:.3f} ± {self.flow_rate_std:.3f} part/s')
        ax2.axhline(self.flow_rate_mean, color='red', linestyle='--',
                   alpha=0.7, label=f'Mean: {self.flow_rate_mean:.3f} part/s')
        ax2.fill_between([self.bin_centers[0], self.bin_centers[-1]],
                        self.flow_rate_mean - self.flow_rate_std,
                        self.flow_rate_mean + self.flow_rate_std,
                        alpha=0.2, color='red', label=f'±1 std dev')

        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Flow rate (particles/s)', fontsize=12)
        ax2.set_title(f'Flow Rate per Bin ({self.bin_size}s bins)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

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
        if self.method == "linear":
            if self.flow_rate is None:
                self.calculate_flow_rate()

            results = {
                'method': 'linear',
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

        elif self.method == "binned":
            if self.flow_rate_mean is None:
                self.calculate_flow_rate_binned()

            results = {
                'method': 'binned',
                'omega': omega,
                'flow_rate_mean': float(self.flow_rate_mean),
                'flow_rate_std': float(self.flow_rate_std),
                'bin_size': float(self.bin_size),
                'bin_count': len(self.bin_rates),
                'bin_rates': self.bin_rates.tolist(),
                'bin_centers': self.bin_centers.tolist(),
                'transient_cutoff': float(self.transient_cutoff),
                'total_particles_exited': int(len(self.exit_times)),
                'total_time': float(self.exit_times[-1]) if len(self.exit_times) > 0 else 0.0,
                'exit_times': self.exit_times.tolist(),
                'cumulative_count': self.cumulative_count.tolist()
            }
        else:
            raise ValueError(f"Unknown analysis method: {self.method}")

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Analysis results saved to {output_file}")

    def get_summary(self) -> Dict:
        """Get summary statistics as dictionary."""
        if self.method == "linear":
            if self.flow_rate is None:
                self.calculate_flow_rate()

            return {
                'method': 'linear',
                'flow_rate': self.flow_rate,
                'flow_rate_std_err': self.flow_rate_std_err,
                'r_squared': self.r_squared,
                'total_particles': len(self.exit_times) if self.exit_times is not None else 0
            }

        elif self.method == "binned":
            if self.flow_rate_mean is None:
                self.calculate_flow_rate_binned()

            return {
                'method': 'binned',
                'flow_rate': self.flow_rate_mean,  # Use 'flow_rate' key for compatibility
                'flow_rate_mean': self.flow_rate_mean,
                'flow_rate_std': self.flow_rate_std,
                'bin_count': len(self.bin_rates),
                'bin_rates': self.bin_rates.tolist(),
                'total_particles': len(self.exit_times) if self.exit_times is not None else 0
            }
        else:
            raise ValueError(f"Unknown analysis method: {self.method}")


def analyze_discharge(exited_file: Path, output_dir: Path,
                     omega: Optional[float] = None,
                     transient_cutoff: float = 20.0,
                     method: str = "linear",
                     bin_size: float = 20.0) -> Dict:
    """
    Complete discharge analysis workflow.

    Args:
        exited_file: Path to exited.txt
        output_dir: Directory to save results
        omega: Frequency value (for labeling)
        transient_cutoff: Time to skip before steady-state analysis
        method: Analysis method - "linear" for linear regression or "binned" for bin-based
        bin_size: Size of time bins in seconds (only used for binned method)

    Returns:
        Dictionary with analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = DischargeAnalyzer(exited_file, transient_cutoff=transient_cutoff,
                                method=method, bin_size=bin_size)

    # Parse and calculate
    analyzer.parse_exit_data()
    if method == "linear":
        analyzer.calculate_flow_rate()
    elif method == "binned":
        analyzer.calculate_flow_rate_binned()
    else:
        raise ValueError(f"Unknown analysis method: {method}")

    # Save results
    analyzer.save_results(output_dir / 'analysis.json', omega=omega)

    # Plot discharge curve
    analyzer.plot_discharge_curve(output_dir / 'discharge_curve.png', omega=omega)

    # Print summary
    summary = analyzer.get_summary()
    print(f"\n{'='*60}")
    print(f"Discharge Analysis Summary" + (f" (ω = {omega} s⁻¹)" if omega else ""))
    print(f"{'='*60}")

    if method == "linear":
        print(f"Method:            Linear regression")
        print(f"Flow rate (Q):     {summary['flow_rate']:.4f} ± {summary['flow_rate_std_err']:.4f} particles/s")
        print(f"R² (goodness):     {summary['r_squared']:.4f}")
    elif method == "binned":
        print(f"Method:            Binned analysis")
        print(f"Bin size:          {bin_size} s")
        print(f"Number of bins:    {summary['bin_count']}")
        print(f"Flow rate (Q):     {summary['flow_rate_mean']:.4f} ± {summary['flow_rate_std']:.4f} particles/s")

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
