"""
Aperture Sweep Runner

Automates running granular media simulations across multiple aperture values.
Organizes results in timestamped directories for persistence across runs.
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import shutil
from typing import List, Dict
from discharge_analysis import analyze_discharge


class ApertureSweepRunner:
    """Automates aperture parameter sweep simulations."""

    def __init__(self,
                 aperture_values: List[float],
                 omega: float = 400.0,
                 simulation_time: float = 200.0,
                 dt: float = 1e-4,
                 jar_path: str = "target/tp5.jar",
                 output_base_dir: str = "src/main/python/results",
                 transient_cutoff: float = 20.0,
                 bin_size: float = 20.0,
                 project_root: Path = None):
        """
        Initialize aperture sweep runner.

        Args:
            aperture_values: List of aperture sizes to test (meters)
            omega: Fixed vibration frequency (s⁻¹)
            simulation_time: Total simulation time (seconds)
            dt: Integration timestep (seconds)
            jar_path: Path to compiled JAR file (relative to project root)
            output_base_dir: Base directory for all results (relative to project root)
            transient_cutoff: Time to skip before steady-state analysis
            bin_size: Size of time bins for flow rate calculation (seconds)
            project_root: Project root directory (auto-detected if None)
        """
        self.aperture_values = aperture_values
        self.omega = omega
        self.simulation_time = simulation_time
        self.dt = dt
        self.transient_cutoff = transient_cutoff
        self.bin_size = bin_size

        # Find project root (directory containing pom.xml)
        if project_root is None:
            self.project_root = self._find_project_root()
        else:
            self.project_root = Path(project_root)

        self.jar_path = self.project_root / jar_path
        self.output_base_dir = self.project_root / output_base_dir

        # Calculate steps from time and dt
        self.steps = int(simulation_time / dt)

        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_base_dir / f"run_{self.timestamp}"

        # Results storage
        self.results = {}

    def _find_project_root(self) -> Path:
        """Find project root by looking for pom.xml."""
        current = Path.cwd().resolve()

        # Check current directory and parents
        for path in [current] + list(current.parents):
            if (path / "pom.xml").exists():
                return path

        # If not found, assume current directory
        print("Warning: Could not find pom.xml, using current directory as project root")
        return current

    def setup_directories(self) -> None:
        """Create timestamped run directory structure."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"Setting up run directory: {self.run_dir}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Fixed omega: {self.omega} s⁻¹")
        print(f"Simulation time: {self.simulation_time} seconds")
        print(f"Bin size for analysis: {self.bin_size} seconds")
        print(f"{'='*70}\n")

    def run_simulation(self, aperture: float) -> Path:
        """
        Run a single simulation for given aperture value.

        Args:
            aperture: Aperture size (meters)

        Returns:
            Path to output directory for this aperture
        """
        aperture_dir = self.run_dir / f"aperture_{aperture:.3f}"
        aperture_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Running simulation: aperture = {aperture} m")
        print(f"Parameters:")
        print(f"  - Steps: {self.steps:,}")
        print(f"  - Time: {self.simulation_time} s")
        print(f"  - dt: {self.dt} s")
        print(f"  - Omega: {self.omega} s⁻¹")
        print(f"  - Output: {aperture_dir}")
        print(f"{'='*70}")

        # Construct Java command
        cmd = [
            "java",
            "-jar",
            str(self.jar_path),
            str(self.steps),
            str(aperture),
            str(self.omega)
        ]

        try:
            # Run simulation from project root with live output
            print(f"\nExecuting: {' '.join(cmd)}")
            print(f"Working directory: {self.project_root}\n")

            # Run with live output so we can see progress bars
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                print(f"\nERROR: Simulation failed for aperture = {aperture}")
                raise RuntimeError(f"Simulation failed with code {result.returncode}")

            print(f"\nSimulation completed successfully!")

            # Move output files to aperture directory
            self._collect_output_files(aperture_dir)

            return aperture_dir

        except subprocess.TimeoutExpired:
            print(f"ERROR: Simulation timeout for aperture = {aperture}")
            raise
        except Exception as e:
            print(f"ERROR: {str(e)}")
            raise

    def _collect_output_files(self, aperture_dir: Path) -> None:
        """
        Collect simulation output files and move to aperture directory.

        Args:
            aperture_dir: Target directory for this aperture value
        """
        # Output files are in sim/ directory (defined in Resources.java)
        sim_dir = self.project_root / "sim"

        # Expected output file
        exited_file = sim_dir / "exited.txt"

        if exited_file.exists():
            dest = aperture_dir / "exited.txt"
            shutil.copy2(exited_file, dest)
            print(f"Copied {exited_file} -> {dest}")
        else:
            print(f"WARNING: Expected output file {exited_file} not found!")
            print(f"  (Particles may not have exited yet - try longer simulation time)")

        # Optionally copy other output directories if needed
        for dir_name in ["particles", "walls"]:
            src_dir = sim_dir / dir_name
            if src_dir.exists() and src_dir.is_dir():
                dest_dir = aperture_dir / dir_name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(src_dir, dest_dir)
                print(f"Copied {src_dir}/ -> {dest_dir}/")

        # Copy setup file if exists
        setup_file = sim_dir / "setup.txt"
        if setup_file.exists():
            shutil.copy2(setup_file, aperture_dir / "setup.txt")

    def analyze_aperture_run(self, aperture: float, aperture_dir: Path) -> Dict:
        """
        Analyze results for a single aperture value.

        Args:
            aperture: Aperture size (meters)
            aperture_dir: Directory containing simulation output

        Returns:
            Analysis results dictionary
        """
        exited_file = aperture_dir / "exited.txt"

        if not exited_file.exists():
            print(f"ERROR: No exited.txt found in {aperture_dir}")
            return None

        print(f"\nAnalyzing discharge data for aperture = {aperture} m...")

        try:
            summary = analyze_discharge(
                exited_file=exited_file,
                output_dir=aperture_dir,
                omega=self.omega,
                transient_cutoff=self.transient_cutoff,
                method="binned",
                bin_size=self.bin_size
            )
            return summary
        except Exception as e:
            print(f"ERROR during analysis: {str(e)}")
            return None

    def run_sweep(self) -> None:
        """Execute complete aperture parameter sweep."""
        self.setup_directories()

        print(f"\n{'#'*70}")
        print(f"# APERTURE SWEEP: {len(self.aperture_values)} simulations")
        print(f"# Values: {self.aperture_values}")
        print(f"# Fixed omega: {self.omega} s⁻¹")
        print(f"{'#'*70}\n")

        # Run all simulations
        for i, aperture in enumerate(self.aperture_values, 1):
            print(f"\n{'*'*70}")
            print(f"* SIMULATION {i}/{len(self.aperture_values)}")
            print(f"{'*'*70}")

            try:
                # Run simulation
                aperture_dir = self.run_simulation(aperture)

                # Analyze results
                summary = self.analyze_aperture_run(aperture, aperture_dir)

                if summary:
                    self.results[aperture] = summary

            except Exception as e:
                print(f"\nFailed to complete aperture = {aperture}: {str(e)}")
                self.results[aperture] = None
                print("Continuing with next simulation...\n")

        # Save summary
        self.save_summary()

        # Print final summary
        self.print_final_summary()

    def save_summary(self) -> None:
        """Save sweep summary to JSON file."""
        summary_file = self.run_dir / "summary.json"

        summary_data = {
            'timestamp': self.timestamp,
            'parameters': {
                'aperture_values': self.aperture_values,
                'omega': self.omega,
                'simulation_time': self.simulation_time,
                'dt': self.dt,
                'steps': self.steps,
                'transient_cutoff': self.transient_cutoff,
                'bin_size': self.bin_size,
                'analysis_method': 'binned'
            },
            'results': {}
        }

        for aperture, result in self.results.items():
            if result:
                summary_data['results'][str(aperture)] = result
            else:
                summary_data['results'][str(aperture)] = {'error': 'Analysis failed'}

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*70}\n")

    def print_final_summary(self) -> None:
        """Print final summary of all runs."""
        print(f"\n{'#'*70}")
        print(f"# APERTURE SWEEP COMPLETE")
        print(f"{'#'*70}\n")

        print(f"{'Aperture (m)':<15} {'Flow Rate (p/s)':<25} {'Status':<10}")
        print(f"{'-'*70}")

        for aperture in self.aperture_values:
            result = self.results.get(aperture)
            if result:
                # Handle both binned and linear results
                if 'flow_rate_mean' in result:
                    q = result['flow_rate_mean']
                    q_err = result['flow_rate_std']
                else:
                    q = result['flow_rate']
                    q_err = result.get('flow_rate_std_err', 0.0)
                status = "✓ SUCCESS"
                print(f"{aperture:<15.3f} {q:.4f} ± {q_err:.4f}           {status}")
            else:
                print(f"{aperture:<15.3f} {'N/A':<25} {'✗ FAILED'}")

        print(f"{'-'*70}\n")
        print(f"Results directory: {self.run_dir}")
        print(f"Run timestamp: {self.timestamp}")
        print(f"Fixed omega: {self.omega} s⁻¹")
        print(f"Analysis method: Binned (bin size = {self.bin_size}s)")
        print(f"\nTo visualize results, run:")
        print(f"  python plot_aperture_results.py --timestamp {self.timestamp}")
        print(f"  or simply: python plot_aperture_results.py (uses latest)\n")


def main():
    """Main entry point for aperture sweep runner."""

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run granular media simulations across multiple aperture values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--apertures",
        type=float,
        nargs='+',
        required=True,
        help="List of aperture sizes to test (meters). Example: --apertures 0.02 0.025 0.03 0.035"
    )

    parser.add_argument(
        "--omega",
        type=float,
        default=400.0,
        help="Fixed vibration frequency (s⁻¹)"
    )

    parser.add_argument(
        "--sim-time",
        type=float,
        default=200.0,
        help="Simulation time in seconds"
    )

    parser.add_argument(
        "--bin-seconds",
        type=float,
        default=20.0,
        help="Size of time bins for flow rate calculation (seconds)"
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1e-4,
        help="Integration timestep (seconds)"
    )

    parser.add_argument(
        "--transient-cutoff",
        type=float,
        default=20.0,
        help="Time to skip before steady-state analysis (seconds)"
    )

    parser.add_argument(
        "--jar-path",
        type=str,
        default="target/tp5.jar",
        help="Path to compiled JAR file (relative to project root)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/main/python/results",
        help="Base directory for results (relative to project root)"
    )

    args = parser.parse_args()

    # Create runner with parsed parameters
    runner = ApertureSweepRunner(
        aperture_values=args.apertures,
        omega=args.omega,
        simulation_time=args.sim_time,
        dt=args.dt,
        jar_path=args.jar_path,
        output_base_dir=args.output_dir,
        transient_cutoff=args.transient_cutoff,
        bin_size=args.bin_seconds
    )

    # Run the sweep
    try:
        runner.run_sweep()
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user. Partial results saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSweep failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
