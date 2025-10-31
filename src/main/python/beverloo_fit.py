"""
Beverloo Law Fitting for 2D Granular Flow

Fits the 2D Beverloo equation to flow rate (Q) vs opening width (d) data:
    Q = B * (d - c*r)^1.5

Where:
    - Q: flow rate (particles/s)
    - d: opening width (m)
    - r: particle mean radius (m)
    - B: proportionality constant (includes gravity and packing effects)
    - c: dimensionless fitting parameter
    - Exponent 1.5: fixed for 2D case

Fitting Method:
    Uses linearization by transforming to: Q^(2/3) = B^(2/3) * (d - c*r)
    Grid search over c values, linear regression for each c, minimize MSE.

Usage:
    python beverloo_fit.py --data Q1,d1 Q2,d2 Q3,d3 ...

Example:
    python beverloo_fit.py --data 0.5,0.02 0.8,0.025 1.2,0.03 1.5,0.035
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class BeverlooFitter:
    """
    Fits the 2D Beverloo law to experimental flow rate data.
    """

    def __init__(self, particle_radius: float = 0.0015):
        """
        Initialize the Beverloo fitter.

        Args:
            particle_radius: Mean particle radius in meters (default: 0.0015 m)
        """
        self.r = particle_radius
        self.exponent = 1.5  # Fixed exponent for 2D case

        # Fitting results
        self.c_optimal: float = None
        self.B_optimal: float = None
        self.mse_optimal: float = None

    def parse_data_pairs(self, data_args: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse command-line data arguments into Q and d arrays.

        Args:
            data_args: List of strings in format "Q,d"

        Returns:
            Tuple of (Q_array, d_array)

        Raises:
            ValueError: If data format is invalid
        """
        Q_values = []
        d_values = []

        for i, pair in enumerate(data_args):
            try:
                parts = pair.split(',')
                if len(parts) != 2:
                    raise ValueError(
                        f"Point {i+1} '{pair}' is not in 'Q,d' format. "
                        f"Expected exactly one comma, got {len(parts)-1}."
                    )

                Q_str, d_str = parts
                Q = float(Q_str)
                d = float(d_str)

                if Q <= 0:
                    raise ValueError(
                        f"Point {i+1}: Flow rate Q={Q} must be positive."
                    )
                if d <= 0:
                    raise ValueError(
                        f"Point {i+1}: Opening width d={d} must be positive."
                    )

                Q_values.append(Q)
                d_values.append(d)

            except ValueError as e:
                if "could not convert" in str(e):
                    raise ValueError(
                        f"Point {i+1} '{pair}' contains non-numeric values."
                    ) from e
                raise

        if len(Q_values) < 3:
            raise ValueError(
                f"Need at least 3 data points for fitting, got {len(Q_values)}."
            )

        return np.array(Q_values), np.array(d_values)

    def beverloo_model(self, d: np.ndarray, B: float, c: float) -> np.ndarray:
        """
        Compute flow rate using Beverloo equation.

        Args:
            d: Opening width(s) in meters
            B: Proportionality constant
            c: Dimensionless parameter

        Returns:
            Predicted flow rate Q
        """
        term = d - c * self.r
        # Prevent negative values under the power
        term = np.maximum(term, 1e-10)
        return B * np.power(term, self.exponent)

    def fit_linear_for_c(self, Q: np.ndarray, d: np.ndarray, c: float) -> Tuple[float, float, float]:
        """
        For a given c, perform linear regression on transformed data.

        Transform: Q^(2/3) = B^(2/3) * (d - c*r)
        This is linear: y = m*x where y = Q^(2/3), x = (d - c*r), m = B^(2/3)

        Args:
            Q: Flow rates
            d: Opening widths
            c: Candidate c value

        Returns:
            Tuple of (B, mse, r_squared)
        """
        # Transform data
        x = d - c * self.r

        # Check if any x values are non-positive (would make model invalid)
        if np.any(x <= 0):
            return None, float('inf'), -float('inf')

        y = np.power(Q, 2.0/3.0)

        # Linear regression: y = m*x (force through origin? or allow intercept?)
        # For physical consistency, we force through origin (no intercept)
        # y = m*x => m = sum(x*y) / sum(x^2)
        slope = np.sum(x * y) / np.sum(x * x)

        # Alternative: allow intercept (more flexible but less physical)
        # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        if slope <= 0:
            return None, float('inf'), -float('inf')

        # Compute B from slope: B = m^(3/2)
        B = np.power(slope, 3.0/2.0)

        # Compute MSE on original Q values (not transformed)
        Q_pred = self.beverloo_model(d, B, c)
        mse = np.mean((Q - Q_pred) ** 2)

        # Compute R^2 for the linear fit
        ss_res = np.sum((y - slope * x) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return B, mse, r_squared

    def fit(self, Q: np.ndarray, d: np.ndarray,
            c_min: float = 0.0, c_max: float = 3.0, c_step: float = 0.01) -> Dict[str, Any]:
        """
        Fit Beverloo law by grid search over c parameter.

        Args:
            Q: Flow rates (particles/s)
            d: Opening widths (m)
            c_min: Minimum c value to test
            c_max: Maximum c value to test
            c_step: Step size for c grid search

        Returns:
            Dictionary with fitting results
        """
        c_candidates = np.arange(c_min, c_max + c_step, c_step)

        best_c = None
        best_B = None
        best_mse = float('inf')
        best_r2 = -float('inf')

        results_list = []

        for c in c_candidates:
            B, mse, r2 = self.fit_linear_for_c(Q, d, c)

            if B is not None and mse < best_mse:
                best_c = c
                best_B = B
                best_mse = mse
                best_r2 = r2

            results_list.append({
                'c': float(c),
                'B': float(B) if B is not None else None,
                'mse': float(mse) if not np.isinf(mse) else None,
                'r2': float(r2) if not np.isinf(r2) else None
            })

        if best_c is None:
            raise RuntimeError(
                "Fitting failed: no valid c value found. "
                "Check that d > c*r for all data points."
            )

        # Store optimal values
        self.c_optimal = best_c
        self.B_optimal = best_B
        self.mse_optimal = best_mse

        return {
            'c_optimal': float(best_c),
            'B_optimal': float(best_B),
            'mse_optimal': float(best_mse),
            'r2_optimal': float(best_r2),
            'particle_radius': float(self.r),
            'exponent': self.exponent,
            'grid_search_results': results_list
        }

    def plot_fit(self, Q: np.ndarray, d: np.ndarray,
                 output_path: Path, show: bool = False):
        """
        Generate and save plot with data and fitted curve.

        Args:
            Q: Flow rates (particles/s)
            d: Opening widths (m)
            output_path: Path to save the plot
            show: Whether to display the plot
        """
        if self.c_optimal is None or self.B_optimal is None:
            raise RuntimeError("Must call fit() before plot_fit()")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot raw data
        ax.scatter(d, Q, s=100, c='blue', marker='o',
                   label='Experimental Data', zorder=3, edgecolors='black')

        # Generate fitted curve (smooth)
        d_min, d_max = d.min(), d.max()
        d_range = d_max - d_min
        d_fit = np.linspace(
            max(d_min - 0.1 * d_range, self.c_optimal * self.r + 0.001),
            d_max + 0.1 * d_range,
            200
        )
        Q_fit = self.beverloo_model(d_fit, self.B_optimal, self.c_optimal)

        ax.plot(d_fit, Q_fit, 'r-', linewidth=2,
                label=f'Beverloo Fit: $Q = B(d - cr)^{{1.5}}$', zorder=2)

        # Add text box with parameters
        textstr = '\n'.join([
            f'$c^* = {self.c_optimal:.3f}$',
            f'$B = {self.B_optimal:.3f}$',
            f'$r = {self.r:.4f}$ m',
            f'MSE $= {self.mse_optimal:.6f}$'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', bbox=props)

        ax.set_xlabel('Opening Width $d$ (m)', fontsize=12)
        ax.set_ylabel('Flow Rate $Q$ (particles/s)', fontsize=12)
        ax.set_title('Beverloo Law Fit: Flow Rate vs Opening Width (2D)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def save_results(self, Q: np.ndarray, d: np.ndarray,
                     fit_results: Dict[str, Any], output_path: Path):
        """
        Save fitting results and data to JSON file.

        Args:
            Q: Flow rates (particles/s)
            d: Opening widths (m)
            fit_results: Dictionary from fit() method
            output_path: Path to save JSON file
        """
        # Generate fitted curve points for replotting
        d_fit = np.linspace(d.min(), d.max(), 100)
        Q_fit = self.beverloo_model(d_fit, self.B_optimal, self.c_optimal)

        output_data = {
            'parameters': {
                'c_optimal': fit_results['c_optimal'],
                'B_optimal': fit_results['B_optimal'],
                'particle_radius': fit_results['particle_radius'],
                'exponent': fit_results['exponent']
            },
            'error_metrics': {
                'mse': fit_results['mse_optimal'],
                'r2': fit_results['r2_optimal']
            },
            'experimental_data': {
                'opening_width_d': d.tolist(),
                'flow_rate_Q': Q.tolist(),
                'num_points': len(Q)
            },
            'fitted_curve': {
                'opening_width_d': d_fit.tolist(),
                'flow_rate_Q': Q_fit.tolist()
            },
            'equation': f"Q = {fit_results['B_optimal']:.4f} * (d - {fit_results['c_optimal']:.4f} * {fit_results['particle_radius']:.4f})^{fit_results['exponent']}"
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {output_path}")


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
    """Main entry point for Beverloo fitting script."""

    parser = argparse.ArgumentParser(
        description='Fit 2D Beverloo law to flow rate vs opening width data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python beverloo_fit.py --data 0.5,0.02 0.8,0.025 1.2,0.03 1.5,0.035

  Each data point is formatted as: FlowRate,OpeningWidth
  - FlowRate (Q): particles per second
  - OpeningWidth (d): meters
        """
    )

    parser.add_argument(
        '--data',
        nargs='+',
        required=True,
        metavar='Q,d',
        help='Data points as space-separated Q,d pairs (e.g., 0.5,0.02 0.8,0.025)'
    )

    parser.add_argument(
        '--radius',
        type=float,
        default=0.0015,
        help='Particle mean radius in meters (default: 0.0015)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='src/main/python/results',
        help='Output directory relative to project root (default: src/main/python/results)'
    )

    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display plot in addition to saving it'
    )

    args = parser.parse_args()

    # Find project root and create absolute output directory
    project_root = find_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize fitter
    fitter = BeverlooFitter(particle_radius=args.radius)

    # Parse input data
    print("Parsing input data...")
    try:
        Q, d = fitter.parse_data_pairs(args.data)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(Q)} data points:")
    for i, (q, dval) in enumerate(zip(Q, d), 1):
        print(f"  Point {i}: Q={q:.4f} particles/s, d={dval:.4f} m")

    # Perform fitting
    print("\nPerforming Beverloo law fitting...")
    print(f"Using particle radius: r = {args.radius} m")
    print("Grid search over c ∈ [0, 3] with step 0.01...")

    try:
        fit_results = fitter.fit(Q, d)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Display results
    print("\n" + "="*60)
    print("FITTING RESULTS")
    print("="*60)
    print(f"Optimal c:       {fit_results['c_optimal']:.6f}")
    print(f"Optimal B:       {fit_results['B_optimal']:.6f}")
    print(f"MSE:             {fit_results['mse_optimal']:.8f}")
    print(f"R² (linear fit): {fit_results['r2_optimal']:.6f}")
    print(f"\nBeverloo equation:")
    print(f"  Q = {fit_results['B_optimal']:.4f} × (d - {fit_results['c_optimal']:.4f} × {args.radius:.4f})^1.5")
    print("="*60)

    # Generate plot
    print("\nGenerating plot...")
    plot_path = output_dir / 'beverloo_fit.png'
    fitter.plot_fit(Q, d, plot_path, show=args.show_plot)

    # Save results
    print("Saving results to JSON...")
    results_path = output_dir / 'beverloo_fit_results.json'
    fitter.save_results(Q, d, fit_results, results_path)

    print("\n✓ Beverloo fitting completed successfully!")


if __name__ == '__main__':
    main()
