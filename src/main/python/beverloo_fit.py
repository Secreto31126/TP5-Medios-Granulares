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
    # Manual data input
    python beverloo_fit.py --data Q1,d1 Q2,d2 Q3,d3 ... --density <rho> [--gravity <g>] [--use-sse]

    # Load from timestamp directory
    python beverloo_fit.py --timestamp YYYYMMDD_HHMMSS --density <rho> [--gravity <g>] [--use-sse]

Examples:
    # Manual input
    python beverloo_fit.py --data 45.33,0.06 61.73,0.07 82.15,0.08 --density 500 --gravity 9.81

    # Load from timestamp (automatically includes error bars from standard deviations)
    python beverloo_fit.py --timestamp 20251102_140408 --density 500

    # Use SSE instead of MSE
    python beverloo_fit.py --timestamp 20251102_140408 --density 500 --use-sse
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

    def load_from_timestamp(self, timestamp: str, results_dir: Path = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load Q vs d data from a timestamp directory's summary.json file.

        Args:
            timestamp: Timestamp string in format YYYYMMDD_HHMMSS
            results_dir: Path to results directory (default: src/main/python/results)

        Returns:
            Tuple of (Q_array, d_array, Q_std_array)

        Raises:
            FileNotFoundError: If timestamp directory or summary.json doesn't exist
            ValueError: If data format is invalid or insufficient data points
        """
        if results_dir is None:
            # Default to src/main/python/results relative to project root
            project_root = find_project_root()
            results_dir = project_root / "src" / "main" / "python" / "results"

        # Construct path to summary.json
        run_dir = results_dir / f"run_{timestamp}"
        summary_path = run_dir / "summary.json"

        if not summary_path.exists():
            raise FileNotFoundError(
                f"Summary file not found: {summary_path}\n"
                f"Please ensure the timestamp '{timestamp}' exists in {results_dir}"
            )

        # Load summary data
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)

        # Extract results
        if 'results' not in summary_data:
            raise ValueError(f"Invalid summary.json format: missing 'results' key")

        results = summary_data['results']

        if len(results) < 3:
            raise ValueError(
                f"Need at least 3 data points for fitting, got {len(results)} from {summary_path}"
            )

        # Parse data from results
        Q_values = []
        d_values = []
        Q_std_values = []

        for d_str, result_data in results.items():
            try:
                # Parse aperture (d) from key
                d = float(d_str)

                # Extract flow rate (Q) - try different possible keys
                if 'flow_rate_mean' in result_data:
                    Q = float(result_data['flow_rate_mean'])
                elif 'flow_rate' in result_data:
                    Q = float(result_data['flow_rate'])
                else:
                    raise ValueError(f"No flow rate data found for aperture {d_str}")

                # Extract standard deviation - try different possible keys
                if 'flow_rate_std' in result_data:
                    Q_std = float(result_data['flow_rate_std'])
                elif 'flow_rate_std_err' in result_data:
                    Q_std = float(result_data['flow_rate_std_err'])
                else:
                    # If no std available, use 0
                    Q_std = 0.0

                if Q <= 0:
                    raise ValueError(f"Flow rate Q={Q} for aperture {d} must be positive")
                if d <= 0:
                    raise ValueError(f"Aperture d={d} must be positive")

                Q_values.append(Q)
                d_values.append(d)
                Q_std_values.append(Q_std)

            except (ValueError, KeyError) as e:
                raise ValueError(f"Error parsing data for aperture {d_str}: {e}") from e

        # Sort by aperture size
        sorted_indices = np.argsort(d_values)
        Q_array = np.array(Q_values)[sorted_indices]
        d_array = np.array(d_values)[sorted_indices]
        Q_std_array = np.array(Q_std_values)[sorted_indices]

        return Q_array, d_array, Q_std_array

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

    def evaluate_c(self, Q: np.ndarray, d: np.ndarray, c: float, use_sse: bool = False) -> Tuple[float, float]:
        """
        Evaluate a given c value with fixed B using logarithmic transformation.

        Transform: ln(Q) = ln(B) + 1.5 * ln(d - c*r)
        With fixed B, we check how well the data fits this linear relationship in log-log space.

        Args:
            Q: Flow rates
            d: Opening widths
            c: Candidate c value
            use_sse: If True, return SSE instead of MSE (default: False)

        Returns:
            Tuple of (mse_or_sse_log_space, r_squared)
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

        # Compute MSE or SSE in log space
        squared_errors = (ln_Q - ln_Q_pred) ** 2
        if use_sse:
            error_metric = np.sum(squared_errors)
        else:
            error_metric = np.mean(squared_errors)

        # Compute R^2 for the linear fit in log space
        ss_res = np.sum((ln_Q - ln_Q_pred) ** 2)
        ss_tot = np.sum((ln_Q - np.mean(ln_Q)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return error_metric, r_squared

    def fit(self, Q: np.ndarray, d: np.ndarray,
            c_min: float = 0.0, c_max: float = 10.0, c_step: float = 0.01,
            use_sse: bool = False) -> Dict[str, Any]:
        """
        Fit Beverloo law by grid search over c parameter with fixed B using logarithmic transformation.

        Args:
            Q: Flow rates (particles/s)
            d: Opening widths (m)
            c_min: Minimum c value to test (default: 0.0)
            c_max: Maximum c value to test (default: 10.0)
            c_step: Step size for c grid search (default: 0.01)
            use_sse: If True, use SSE instead of MSE for optimization (default: False)

        Returns:
            Dictionary with fitting results (MSE or SSE in log space)
        """
        if self.B is None:
            raise RuntimeError("B constant not initialized. Must provide particle_density.")

        c_candidates = np.arange(c_min, c_max + c_step, c_step)

        best_c = None
        best_error = float('inf')
        best_r2 = -float('inf')

        results_list = []

        for c in c_candidates:
            error, r2 = self.evaluate_c(Q, d, c, use_sse=use_sse)

            if error < best_error:
                best_c = c
                best_error = error
                best_r2 = r2

            error_key = 'sse' if use_sse else 'mse'
            results_list.append({
                'c': float(c),
                error_key: float(error) if not np.isinf(error) else None,
                'r2': float(r2) if not np.isinf(r2) else None
            })

        if best_c is None:
            raise RuntimeError(
                "Fitting failed: no valid c value found. "
                "Check that d > c*r for all data points."
            )

        # Store optimal values
        self.c_optimal = best_c
        self.mse_optimal = best_error

        error_key = 'sse_optimal' if use_sse else 'mse_optimal'
        return {
            'c_optimal': float(best_c),
            'B_constant': float(self.B),
            'particle_density': float(self.rho),
            'gravity': float(self.g),
            error_key: float(best_error),
            'r2_optimal': float(best_r2),
            'particle_radius': float(self.r),
            'exponent': self.exponent,
            'use_sse': use_sse,
            'grid_search_results': results_list
        }

    def plot_fit(self, Q: np.ndarray, d: np.ndarray,
                 output_path: Path, show: bool = False, Q_std: np.ndarray = None):
        """
        Generate and save plot with data and fitted curve.

        Args:
            Q: Flow rates (particles/s)
            d: Opening widths (m)
            output_path: Path to save the plot
            show: Whether to display the plot
            Q_std: Optional standard deviations for Q (for error bars)
        """
        if self.c_optimal is None or self.B is None:
            raise RuntimeError("Must call fit() before plot_fit()")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot raw data with or without error bars
        if Q_std is not None and np.any(Q_std > 0):
            # Use errorbar for data with uncertainties
            ax.errorbar(d, Q, yerr=Q_std, fmt='o', markersize=8, capsize=5,
                       c='blue', markeredgecolor='black', markeredgewidth=1,
                       ecolor='blue', elinewidth=2, capthick=2,
                       zorder=3, label='Resultados experimentales')
        else:
            # Use scatter for data without uncertainties
            ax.scatter(d, Q, s=100, c='blue', marker='o', zorder=3, edgecolors='black',
                      label='Resultados experimentales')

        # Generate fitted curve (smooth)
        d_min, d_max = d.min(), d.max()
        d_range = d_max - d_min
        d_fit = np.linspace(
            max(d_min - 0.1 * d_range, self.c_optimal * self.r + 0.001),
            d_max + 0.1 * d_range,
            200
        )
        Q_fit = self.beverloo_model(d_fit, self.c_optimal)

        ax.plot(d_fit, Q_fit, 'r-', linewidth=2, zorder=2, label='Ajuste de Beverloo')

        ax.set_xlabel(r'Apertura $D$ (m)', fontsize=28, fontweight='bold')
        ax.set_ylabel(r'Caudal $Q$ (partículas/s)', fontsize=28, fontweight='bold')
        ax.legend(fontsize=28)
        ax.tick_params(axis='both', labelsize=28)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_mse_vs_c(self, fit_results: Dict[str, Any], output_path: Path, show: bool = False):
        """
        Plot MSE or SSE as a function of c parameter.

        Args:
            fit_results: Dictionary from fit() method containing grid search results
            output_path: Path to save the plot
            show: Whether to display the plot
        """
        if self.c_optimal is None:
            raise RuntimeError("Must call fit() before plot_mse_vs_c()")

        # Extract grid search results
        grid_results = fit_results['grid_search_results']
        use_sse = fit_results.get('use_sse', False)
        error_key = 'sse' if use_sse else 'mse'

        # Filter out invalid results (where error is None/inf)
        valid_results = [r for r in grid_results if r.get(error_key) is not None]

        if not valid_results:
            print(f"Warning: No valid {error_key.upper()} values to plot")
            return

        c_values = np.array([r['c'] for r in valid_results])
        error_values = np.array([r[error_key] for r in valid_results])

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot error vs c with log scale on Y-axis
        ax.plot(c_values, error_values, 'b-', linewidth=2, alpha=0.7)

        ax.set_xlabel(r'$c$', fontsize=28, fontweight='bold')
        ylabel = 'Suma de Errores Cuadráticos' if use_sse else 'Error Cuadrático Medio'
        ax.set_ylabel(ylabel, fontsize=28, fontweight='bold')
        ax.set_yscale('log')  # Set Y-axis to logarithmic scale
        ax.set_xlim([0.5, 2.0])
        ax.tick_params(axis='both', labelsize=28)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')  # Show grid for both major and minor ticks

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        metric_name = "SSE" if use_sse else "MSE"
        print(f"{metric_name} vs c plot saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def save_results(self, Q: np.ndarray, d: np.ndarray,
                     fit_results: Dict[str, Any], output_path: Path, Q_std: np.ndarray = None):
        """
        Save fitting results and data to JSON file.

        Args:
            Q: Flow rates (particles/s)
            d: Opening widths (m)
            fit_results: Dictionary from fit() method
            output_path: Path to save JSON file
            Q_std: Optional standard deviations for Q
        """
        # Generate fitted curve points for replotting
        d_fit = np.linspace(d.min(), d.max(), 100)
        Q_fit = self.beverloo_model(d_fit, self.c_optimal)

        # Determine which error metric was used
        use_sse = fit_results.get('use_sse', False)
        error_key = 'sse_optimal' if use_sse else 'mse_optimal'
        error_metric_name = 'sse' if use_sse else 'mse'

        # Build experimental data section
        experimental_data = {
            'opening_width_d': d.tolist(),
            'flow_rate_Q': Q.tolist(),
            'num_points': len(Q)
        }

        # Add standard deviations if available
        if Q_std is not None:
            experimental_data['flow_rate_Q_std'] = Q_std.tolist()

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
                error_metric_name: fit_results[error_key],
                'r2': fit_results['r2_optimal'],
                'metric_type': 'SSE' if use_sse else 'MSE'
            },
            'experimental_data': experimental_data,
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
Examples:
  # Manual data input
  python beverloo_fit.py --data 45.33,0.06 61.73,0.07 82.15,0.08 --density 500

  # Load from timestamp directory (with error bars)
  python beverloo_fit.py --timestamp 20251102_140408 --density 500

  # Use SSE instead of MSE
  python beverloo_fit.py --timestamp 20251102_140408 --density 500 --use-sse

Data format:
  - Manual mode: Each data point is formatted as FlowRate,OpeningWidth
  - Timestamp mode: Automatically loads from src/main/python/results/run_TIMESTAMP/summary.json
  - FlowRate (Q): particles per second
  - OpeningWidth (d): meters
  - density: particle packing density (particles/m²) - REQUIRED
        """
    )

    parser.add_argument(
        '--data',
        nargs='+',
        metavar='Q,d',
        help='Data points as space-separated Q,d pairs (e.g., 0.5,0.02 0.8,0.025). Mutually exclusive with --timestamp.'
    )

    parser.add_argument(
        '--timestamp',
        type=str,
        metavar='YYYYMMDD_HHMMSS',
        help='Load data from timestamp directory (e.g., 20251102_140408). Mutually exclusive with --data.'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        help='Path to results directory (default: src/main/python/results). Only used with --timestamp.'
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

    parser.add_argument(
        '--use-sse',
        action='store_true',
        help='Use SSE (Sum of Squared Errors) instead of MSE for optimization'
    )

    args = parser.parse_args()

    # Validate that exactly one of --data or --timestamp is provided
    if args.data and args.timestamp:
        print("Error: Cannot specify both --data and --timestamp. Choose one.", file=sys.stderr)
        sys.exit(1)
    if not args.data and not args.timestamp:
        print("Error: Must specify either --data or --timestamp.", file=sys.stderr)
        sys.exit(1)

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
    Q_std = None  # Will be set if loading from timestamp

    if args.data:
        # Manual data input mode
        print("Parsing input data from command line...")
        try:
            Q, d = fitter.parse_data_pairs(args.data)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Loaded {len(Q)} data points:")
        for i, (q, dval) in enumerate(zip(Q, d), 1):
            print(f"  Point {i}: Q={q:.4f} particles/s, d={dval:.4f} m")

    else:
        # Timestamp mode
        print(f"Loading data from timestamp: {args.timestamp}")
        try:
            results_dir = Path(args.results_dir) if args.results_dir else None
            Q, d, Q_std = fitter.load_from_timestamp(args.timestamp, results_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Loaded {len(Q)} data points from run_{args.timestamp}/summary.json:")
        for i, (q, dval, q_std) in enumerate(zip(Q, d, Q_std), 1):
            if q_std > 0:
                print(f"  Point {i}: Q={q:.4f} ± {q_std:.4f} particles/s, d={dval:.4f} m")
            else:
                print(f"  Point {i}: Q={q:.4f} particles/s, d={dval:.4f} m")

    # Perform fitting
    print("\nPerforming Beverloo law fitting...")
    print(f"Using particle radius: r = {args.radius} m")
    print(f"Using particle density: ρ = {args.density} particles/m²")
    print(f"Using gravity: g = {args.gravity} m/s²")
    print(f"Fixed B constant: B = ρ√g = {fitter.B:.4f}")
    error_metric = "SSE" if args.use_sse else "MSE"
    print(f"Error metric: {error_metric}")
    print(f"Grid search over c ∈ [{args.c_min}, {args.c_max}] with step {args.c_step}...")

    try:
        fit_results = fitter.fit(Q, d, c_min=args.c_min, c_max=args.c_max, c_step=args.c_step, use_sse=args.use_sse)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Display results
    error_key = 'sse_optimal' if args.use_sse else 'mse_optimal'
    error_label = "SSE (log space)" if args.use_sse else "MSE (log space)"
    print("\n" + "="*60)
    print("FITTING RESULTS (Log-Log Method)")
    print("="*60)
    print(f"Optimal c:           {fit_results['c_optimal']:.6f}")
    print(f"Fixed B:             {fit_results['B_constant']:.6f} (= {args.density:.1f} × √{args.gravity:.2f})")
    print(f"{error_label:20} {fit_results[error_key]:.8f}")
    print(f"R² (log-log fit):    {fit_results['r2_optimal']:.6f}")
    print(f"\nBeverloo equation:")
    print(f"  Q = {fit_results['B_constant']:.4f} × (d - {fit_results['c_optimal']:.4f} × {args.radius:.4f})^1.5")
    print("="*60)

    # Generate plots
    print("\nGenerating plots...")
    plot_path = output_dir / 'beverloo_fit.png'
    fitter.plot_fit(Q, d, plot_path, show=args.show_plot, Q_std=Q_std)

    mse_plot_path = output_dir / 'beverloo_mse_vs_c.png'
    fitter.plot_mse_vs_c(fit_results, mse_plot_path, show=args.show_plot)

    # Save results
    print("\nSaving results to JSON...")
    results_path = output_dir / 'beverloo_fit_results.json'
    fitter.save_results(Q, d, fit_results, results_path, Q_std=Q_std)

    print("\n✓ Beverloo fitting completed successfully!")


if __name__ == '__main__':
    main()
