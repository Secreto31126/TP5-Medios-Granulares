"""
Omega Sweep Runner

Automates running granular media simulations across multiple omega (frequency) values.
Organizes results in timestamped directories for persistence across runs.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import shutil
from typing import List, Dict
from discharge_analysis import analyze_discharge


class OmegaSweepRunner:
    """Automates omega parameter sweep simulations."""

    def __init__(self,
                 omega_values: List[float],
                 simulation_time: float = 1000.0,
                 dt: float = 1e-4,
                 aperture: float = 0.03,
                 jar_path: str = "target/tp5.jar",
                 output_base_dir: str = "src/main/python/results",
                 transient_cutoff: float = 40.0,
                 project_root: Path = None):
        """
        Initialize omega sweep runner.

        Args:
            omega_values: List of omega frequencies to test (s⁻¹)
            simulation_time: Total simulation time (seconds)
            dt: Integration timestep (seconds)
            aperture: Box aperture size (meters)
            jar_path: Path to compiled JAR file (relative to project root)
            output_base_dir: Base directory for all results (relative to project root)
            transient_cutoff: Time to skip before steady-state analysis
            project_root: Project root directory (auto-detected if None)
        """
        self.omega_values = omega_values
        self.simulation_time = simulation_time
        self.dt = dt
        self.aperture = aperture
        self.transient_cutoff = transient_cutoff

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
        print(f"{'='*70}\n")

    def run_simulation(self, omega: float) -> Path:
        """
        Run a single simulation for given omega value.

        Args:
            omega: Frequency value (s⁻¹)

        Returns:
            Path to output directory for this omega
        """
        omega_dir = self.run_dir / f"omega_{int(omega)}"
        omega_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Running simulation: ω = {omega} s⁻¹")
        print(f"Parameters:")
        print(f"  - Steps: {self.steps:,}")
        print(f"  - Time: {self.simulation_time} s")
        print(f"  - dt: {self.dt} s")
        print(f"  - Aperture: {self.aperture} m")
        print(f"  - Output: {omega_dir}")
        print(f"{'='*70}")

        # Construct Java command
        cmd = [
            "java",
            "-jar",
            str(self.jar_path),
            str(self.steps),
            str(self.aperture),
            str(omega)
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
                print(f"\nERROR: Simulation failed for ω = {omega}")
                raise RuntimeError(f"Simulation failed with code {result.returncode}")

            print(f"\nSimulation completed successfully!")

            # Move output files to omega directory
            self._collect_output_files(omega_dir)

            return omega_dir

        except subprocess.TimeoutExpired:
            print(f"ERROR: Simulation timeout for ω = {omega}")
            raise
        except Exception as e:
            print(f"ERROR: {str(e)}")
            raise

    def _collect_output_files(self, omega_dir: Path) -> None:
        """
        Collect simulation output files and move to omega directory.

        Args:
            omega_dir: Target directory for this omega value
        """
        # Output files are in sim/ directory (defined in Resources.java)
        sim_dir = self.project_root / "sim"

        # Expected output file
        exited_file = sim_dir / "exited.txt"

        if exited_file.exists():
            dest = omega_dir / "exited.txt"
            shutil.copy2(exited_file, dest)
            print(f"Copied {exited_file} -> {dest}")
        else:
            print(f"WARNING: Expected output file {exited_file} not found!")
            print(f"  (Particles may not have exited yet - try longer simulation time)")

        # Optionally copy other output directories if needed
        for dir_name in ["particles", "walls"]:
            src_dir = sim_dir / dir_name
            if src_dir.exists() and src_dir.is_dir():
                dest_dir = omega_dir / dir_name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(src_dir, dest_dir)
                print(f"Copied {src_dir}/ -> {dest_dir}/")

        # Copy setup file if exists
        setup_file = sim_dir / "setup.txt"
        if setup_file.exists():
            shutil.copy2(setup_file, omega_dir / "setup.txt")

    def analyze_omega_run(self, omega: float, omega_dir: Path) -> Dict:
        """
        Analyze results for a single omega value.

        Args:
            omega: Frequency value
            omega_dir: Directory containing simulation output

        Returns:
            Analysis results dictionary
        """
        exited_file = omega_dir / "exited.txt"

        if not exited_file.exists():
            print(f"ERROR: No exited.txt found in {omega_dir}")
            return None

        print(f"\nAnalyzing discharge data for ω = {omega} s⁻¹...")

        try:
            summary = analyze_discharge(
                exited_file=exited_file,
                output_dir=omega_dir,
                omega=omega,
                transient_cutoff=self.transient_cutoff
            )
            return summary
        except Exception as e:
            print(f"ERROR during analysis: {str(e)}")
            return None

    def run_sweep(self) -> None:
        """Execute complete omega parameter sweep."""
        self.setup_directories()

        print(f"\n{'#'*70}")
        print(f"# OMEGA SWEEP: {len(self.omega_values)} simulations")
        print(f"# Values: {self.omega_values}")
        print(f"{'#'*70}\n")

        # Run all simulations
        for i, omega in enumerate(self.omega_values, 1):
            print(f"\n{'*'*70}")
            print(f"* SIMULATION {i}/{len(self.omega_values)}")
            print(f"{'*'*70}")

            try:
                # Run simulation
                omega_dir = self.run_simulation(omega)

                # Analyze results
                summary = self.analyze_omega_run(omega, omega_dir)

                if summary:
                    self.results[omega] = summary

            except Exception as e:
                print(f"\nFailed to complete ω = {omega}: {str(e)}")
                self.results[omega] = None
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
                'omega_values': self.omega_values,
                'simulation_time': self.simulation_time,
                'dt': self.dt,
                'steps': self.steps,
                'aperture': self.aperture,
                'transient_cutoff': self.transient_cutoff
            },
            'results': {}
        }

        for omega, result in self.results.items():
            if result:
                summary_data['results'][str(omega)] = result
            else:
                summary_data['results'][str(omega)] = {'error': 'Analysis failed'}

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*70}\n")

    def print_final_summary(self) -> None:
        """Print final summary of all runs."""
        print(f"\n{'#'*70}")
        print(f"# OMEGA SWEEP COMPLETE")
        print(f"{'#'*70}\n")

        print(f"{'Omega (s⁻¹)':<15} {'Flow Rate (p/s)':<20} {'R²':<10} {'Status':<10}")
        print(f"{'-'*70}")

        for omega in self.omega_values:
            result = self.results.get(omega)
            if result:
                q = result['flow_rate']
                q_err = result['flow_rate_std_err']
                r2 = result['r_squared']
                status = "✓ SUCCESS"
                print(f"{omega:<15.1f} {q:.4f} ± {q_err:.4f}      {r2:<10.4f} {status}")
            else:
                print(f"{omega:<15.1f} {'N/A':<20} {'N/A':<10} {'✗ FAILED'}")

        print(f"{'-'*70}\n")
        print(f"Results directory: {self.run_dir}")
        print(f"Run timestamp: {self.timestamp}")
        print(f"\nTo visualize results, run:")
        print(f"  python analysis/plot_omega_results.py --timestamp {self.timestamp}")
        print(f"  or simply: python analysis/plot_omega_results.py (uses latest)\n")


def main():
    """Main entry point for omega sweep runner."""

    # Default omega values for Point A
    omega_values = [400, 450, 500, 550, 600]

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            omega_values = [float(x) for x in sys.argv[1:]]
            print(f"Using custom omega values: {omega_values}")
        except ValueError:
            print("ERROR: Invalid omega values provided")
            print("Usage: python omega_sweep_runner.py [omega1 omega2 ...]")
            print("Example: python omega_sweep_runner.py 400 450 500 550 600")
            sys.exit(1)

    # Create runner with Point A parameters
    runner = OmegaSweepRunner(
        omega_values=omega_values,
        simulation_time=1000.0,      # 1000 seconds
        dt=1e-4,                      # 0.0001 s timestep
        aperture=0.03,                # 3 cm aperture
        jar_path="target/tp5.jar",
        output_base_dir="src/main/python/results",
        transient_cutoff=40.0         # Skip first 40s for steady-state analysis
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
