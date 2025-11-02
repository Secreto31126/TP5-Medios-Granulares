"""
Beverloo Law Fitting for 2D Granular Flow

Fits the 2D Beverloo equation to flow rate (Q) vs opening width (d) data:
    Q = B * (d - c*r)^1.5

Where:
    - Q: flow rate (particles/s)
    - d: opening width (m)
    - r: particle mean radius (m)
    - B: constant = ρ * sqrt(g) (particle packing density × sqrt(gravity))
    - c: dimensionless fitting parameter (only free parameter)
    - Exponent 1.5: fixed for 2D case

Fitting Method:
    Logarithmic transformation: ln(Q) = ln(B) + 1.5 * ln(d - c*r)
    Grid search over c values to minimize MSE in log space with fixed B.
    B is calculated from particle density and gravity, not fitted.

Usage:
    python beverloo_fit.py --data Q1,d1 Q2,d2 Q3,d3 ... --density <rho> --gravity <g>

Example:
    python beverloo_fit.py --data 0.5,0.02 0.8,0.025 1.2,0.03 1.5,0.035 --density 500 --gravity 9.81
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

    def __init__(self, particle_radius: float = 0.01, particle_density: float = None, gravity: float = 9.81):
        """
        Initialize the Beverloo fitter.

        Args:
            particle_radius: Mean particle radius in meters (default: 0.01 m = 1 cm)
            particle_density: Particle packing density (particles/m² in packing region)
            gravity: Gravitational acceleration in m/s² (default: 9.81)
        """
        self.r = particle_radius
        self.rho = particle_density
        self.g = gravity
        self.exponent = 1.5  # Fixed exponent for 2D case

        # Calculate B constant (fixed, not fitted)
        if particle_density is not None:
            self.B = particle_density * np.sqrt(gravity)
        else:
            self.B = None

        # Fitting results
        self.c_optimal: float = None
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

    def beverloo_model(self, d: np.ndarray, c: float) -> np.ndarray:
        """
        Compute flow rate using Beverloo equation with fixed B.

        Args:
            d: Opening width(s) in meters
            c: Dimensionless parameter

        Returns:
            Predicted flow rate Q
        """
        if self.B is None:
            raise RuntimeError("B constant not initialized. Must provide particle_density.")

        term = d - c * self.r
        # Prevent negative values under the power
        term = np.maximum(term, 1e-10)
        return self.B * np.power(term, self.exponent)

    def evaluate_c(self, Q: np.ndarray, d: np.ndarray, c: float) -> Tuple[float, float]:
        """
        Evaluate a given c value with fixed B using logarithmic transformation.

        Transform: ln(Q) = ln(B) + 1.5 * ln(d - c*r)
        With fixed B, we check how well the data fits this linear relationship in log-log space.

        Args:
            Q: Flow rates
            d: Opening widths
            c: Candidate c value

        Returns:
            Tuple of (mse_log_space, r_squared)
        """
        if self.B is None:
            raise RuntimeError("B constant not initialized. Must provide particle_density.")

        # Transform data
        x = d - c * self.r

        # Check if any values are non-positive (would make log undefined)
        if np.any(x <= 0) or np.any(Q <= 0):
            return float('inf'), -float('inf')

        # Logarithmic transformations
        ln_Q = np.log(Q)
        ln_x = np.log(x)

        # Expected linear relationship in log space: ln(Q) = ln(B) + 1.5 * ln(x)
        a_expected = np.log(self.B)  # ln(B)
        b_expected = 1.5  # Fixed exponent for 2D case

        # Predicted ln(Q) values
        ln_Q_pred = a_expected + b_expected * ln_x

        # Compute MSE in log space
        mse_log_space = np.mean((ln_Q - ln_Q_pred) ** 2)

        # Compute R^2 for the linear fit in log space
        ss_res = np.sum((ln_Q - ln_Q_pred) ** 2)
        ss_tot = np.sum((ln_Q - np.mean(ln_Q)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return mse_log_space, r_squared

    def fit(self, Q: np.ndarray, d: np.ndarray,
            c_min: float = 0.0, c_max: float = 10.0, c_step: float = 0.01) -> Dict[str, Any]:
        """
        Fit Beverloo law by grid search over c parameter with fixed B using logarithmic transformation.

        Args:
            Q: Flow rates (particles/s)
            d: Opening widths (m)
            c_min: Minimum c value to test (default: 0.0)
            c_max: Maximum c value to test (default: 10.0)
            c_step: Step size for c grid search (default: 0.01)

        Returns:
            Dictionary with fitting results (MSE in log space)
        """
        if self.B is None:
            raise RuntimeError("B constant not initialized. Must provide particle_density.")

        c_candidates = np.arange(c_min, c_max + c_step, c_step)

        best_c = None
        best_mse = float('inf')
        best_r2 = -float('inf')

        results_list = []

        for c in c_candidates:
            mse, r2 = self.evaluate_c(Q, d, c)

            if mse < best_mse:
                best_c = c
                best_mse = mse
                best_r2 = r2

            results_list.append({
                'c': float(c),
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
        self.mse_optimal = best_mse

        return {
            'c_optimal': float(best_c),
            'B_constant': float(self.B),
            'particle_density': float(self.rho),
            'gravity': float(self.g),
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
        if self.c_optimal is None or self.B is None:
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
        Q_fit = self.beverloo_model(d_fit, self.c_optimal)

        ax.plot(d_fit, Q_fit, 'r-', linewidth=2,
                label=f'Beverloo Fit: $Q = B(d - cr)^{{1.5}}$', zorder=2)

        # Add text box with parameters
        textstr = '\n'.join([
            f'$c^* = {self.c_optimal:.3f}$',
            f'$B = {self.B:.3f}$ (fixed)',
            f'$\\rho = {self.rho:.1f}$ particles/m²',
            f'$r = {self.r/2.0:.4f}$ m',
            f'MSE (log) $= {self.mse_optimal:.6f}$'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', bbox=props)

        ax.set_xlabel('Opening Width $d$ (m)', fontsize=12)
        ax.set_ylabel('Flow Rate $Q$ (particles/s)', fontsize=12)
        ax.set_title('Beverloo Law Fit (Log-Log): Flow Rate vs Opening Width (2D)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_mse_vs_c(self, fit_results: Dict[str, Any], output_path: Path, show: bool = False):
        """
        Plot MSE as a function of c parameter.

        Args:
            fit_results: Dictionary from fit() method containing grid search results
            output_path: Path to save the plot
            show: Whether to display the plot
        """
        if self.c_optimal is None:
            raise RuntimeError("Must call fit() before plot_mse_vs_c()")

        # Extract grid search results
        grid_results = fit_results['grid_search_results']

        # Filter out invalid results (where MSE is None/inf)
        valid_results = [r for r in grid_results if r['mse'] is not None]

        if not valid_results:
            print("Warning: No valid MSE values to plot")
            return

        c_values = np.array([r['c'] for r in valid_results])
        mse_values = np.array([r['mse'] for r in valid_results])

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot MSE vs c with log scale on Y-axis
        ax.plot(c_values, mse_values, 'b-', linewidth=2, alpha=0.7)

        # Mark optimal c
        ax.plot(self.c_optimal, self.mse_optimal, 'r*', markersize=15,
                label=f'Optimal: c = {self.c_optimal:.3f}, MSE (log) = {self.mse_optimal:.6f}')

        ax.set_xlabel('Parameter c (dimensionless)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Squared Error (log space)', fontsize=12, fontweight='bold')
        ax.set_yscale('log')  # Set Y-axis to logarithmic scale
        ax.set_title('Model Error vs Fitting Parameter c (Log-Log Fit)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')  # Show grid for both major and minor ticks

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"MSE vs c plot saved to: {output_path}")

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
        Q_fit = self.beverloo_model(d_fit, self.c_optimal)

        output_data = {
            'parameters': {
                'c_optimal': fit_results['c_optimal'],
                'B_constant': fit_results['B_constant'],
                'particle_density': fit_results['particle_density'],
                'gravity': fit_results['gravity'],
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
            'equation': f"Q = {fit_results['B_constant']:.4f} * (d - {fit_results['c_optimal']:.4f} * {fit_results['particle_radius']:.4f})^{fit_results['exponent']}"
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
  python beverloo_fit.py --data 0.5,0.02 0.8,0.025 1.2,0.03 1.5,0.035 --density 500

  Each data point is formatted as: FlowRate,OpeningWidth
  - FlowRate (Q): particles per second
  - OpeningWidth (d): meters
  - density: particle packing density (particles/m²) - REQUIRED
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
        default=0.01,
        help='Particle mean radius in meters (default: 0.01 = 1 cm)'
    )

    parser.add_argument(
        '--density',
        type=float,
        required=True,
        help='Particle packing density (particles/m² in packing region)'
    )

    parser.add_argument(
        '--gravity',
        type=float,
        default=9.81,
        help='Gravitational acceleration in m/s² (default: 9.81)'
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

    parser.add_argument(
        '--c-min',
        type=float,
        default=0.0,
        help='Minimum c value for grid search (default: 0.0)'
    )

    parser.add_argument(
        '--c-max',
        type=float,
        default=10.0,
        help='Maximum c value for grid search (default: 10.0)'
    )

    parser.add_argument(
        '--c-step',
        type=float,
        default=0.01,
        help='Step size for c grid search (default: 0.01)'
    )

    args = parser.parse_args()

    # Find project root and create absolute output directory
    project_root = find_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize fitter
    fitter = BeverlooFitter(
        particle_radius=2*args.radius,
        particle_density=args.density,
        gravity=args.gravity
    )

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
    print(f"Using particle density: ρ = {args.density} particles/m²")
    print(f"Using gravity: g = {args.gravity} m/s²")
    print(f"Fixed B constant: B = ρ√g = {fitter.B:.4f}")
    print(f"Grid search over c ∈ [{args.c_min}, {args.c_max}] with step {args.c_step}...")

    try:
        fit_results = fitter.fit(Q, d, c_min=args.c_min, c_max=args.c_max, c_step=args.c_step)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Display results
    print("\n" + "="*60)
    print("FITTING RESULTS (Log-Log Method)")
    print("="*60)
    print(f"Optimal c:           {fit_results['c_optimal']:.6f}")
    print(f"Fixed B:             {fit_results['B_constant']:.6f} (= {args.density:.1f} × √{args.gravity:.2f})")
    print(f"MSE (log space):     {fit_results['mse_optimal']:.8f}")
    print(f"R² (log-log fit):    {fit_results['r2_optimal']:.6f}")
    print(f"\nBeverloo equation:")
    print(f"  Q = {fit_results['B_constant']:.4f} × (d - {fit_results['c_optimal']:.4f} × {args.radius:.4f})^1.5")
    print("="*60)

    # Generate plots
    print("\nGenerating plots...")
    plot_path = output_dir / 'beverloo_fit.png'
    fitter.plot_fit(Q, d, plot_path, show=args.show_plot)

    mse_plot_path = output_dir / 'beverloo_mse_vs_c.png'
    fitter.plot_mse_vs_c(fit_results, mse_plot_path, show=args.show_plot)

    # Save results
    print("\nSaving results to JSON...")
    results_path = output_dir / 'beverloo_fit_results.json'
    fitter.save_results(Q, d, fit_results, results_path)

    print("\n✓ Beverloo fitting completed successfully!")


if __name__ == '__main__':
    main()
