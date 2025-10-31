from pathlib import Path
import subprocess, shutil, json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# ---------- helpers de localización ----------
def find_repo_root(start: Path) -> Path:
    """
    Sube directorios hasta encontrar algo que parezca la raíz del proyecto:
    - un pom.xml (Maven) o
    - settings.gradle / build.gradle (Gradle)
    Si no encuentra, devuelve 'start'.
    """
    for p in [start] + list(start.parents):
        if (p / "pom.xml").exists() or (p / "build.gradle").exists() or (p / "settings.gradle").exists():
            return p
    return start

def find_jar(repo_root: Path, preferred_name="tp5") -> Path | None:
    """
    Busca un JAR en target/. Si hay varios, prioriza el que contenga 'preferred_name'.
    """
    target = repo_root / "target"
    if not target.exists():
        return None
    jars = sorted(target.glob("*.jar"))
    if not jars:
        return None
    for j in jars:
        if preferred_name.lower() in j.name.lower():
            return j
    return jars[0]  # al menos devuelve alguno

# ---------- tu runner, pero robusto ----------
def run_simulation(steps, aperture, omega, jar_path: Path, repo_root: Path):
    """
    Ejecuta el JAR con cwd en la raíz del repo para que 'sim/' quede en repo_root/sim.
    """
    cmd = ["java", "-jar", str(jar_path), str(steps), str(aperture), str(omega)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        print(f"Error running simulation with aperture={aperture}: {result.stderr.strip()}")
        return None
    return result.stdout.strip()

def collect_output_files(project_root: Path, omega_dir: Path) -> None:
    """
    Copia salidas generadas en project_root/sim/* hacia omega_dir.
    """
    sim_dir = project_root / "sim"
    exited_file = sim_dir / "exited.txt"

    if exited_file.exists():
        dest = omega_dir / "exited.txt"
        shutil.copy2(exited_file, dest)
        print(f"Copied {exited_file} -> {dest}")
    else:
        print(f"WARNING: Expected output file {exited_file} not found!")
        print("  (Particles may not have exited yet - try longer simulation time)")

    for dir_name in ["particles", "walls"]:
        src_dir = sim_dir / dir_name
        if src_dir.exists() and src_dir.is_dir():
            dest_dir = omega_dir / dir_name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(src_dir, dest_dir)
            print(f"Copied {src_dir}/ -> {dest_dir}/")

    setup_file = sim_dir / "setup.txt"
    if setup_file.exists():
        shutil.copy2(setup_file, omega_dir / "setup.txt")

# ---------- parsing/plot (tus funciones) ----------
def parse_exited_file(filename):
    """
    Lee exited.txt y devuelve un np.array de tiempos de salida (float).
    Soporta 1 columna, varias columnas (toma la 1ra), 1 solo número o vacío.
    """
    try:
        data = np.loadtxt(filename)

        # Normalizar formas: escalar -> (1,), vector -> (N,), matriz -> (N, M)
        data = np.squeeze(np.array(data, dtype=float))

        if data.ndim == 0:
            # Un solo número
            exit_times = np.array([float(data)])
        elif data.ndim == 1:
            # Una sola columna -> ya es (N,)
            exit_times = data
        else:
            # Varias columnas -> tomar la primera como "tiempo"
            exit_times = data[:, 0]

        return exit_times
    except Exception as e:
        print(f"Error leyendo {filename}: {e}")
        return np.array([])

def calculate_flow_rate(escaped_particles, steps, dt=1e-4):
    """
    Calculate flow rate in particles/second.

    Args:
        escaped_particles: number of particles that escaped
        steps: number of simulation steps
        dt: time step size (default: 1e-4 as used in Beeman integrator)

    Returns:
        flow rate in particles/second
    """
    total_time = steps * dt  # total simulation time in seconds
    return escaped_particles / total_time

def plot_flow_rate(flow_rates, apertures, out_path: Path | None = None):
    plt.figure(figsize=(8, 5))
    # scatter simple (un punto por apertura)
    plt.scatter(apertures, flow_rates)
    plt.plot(apertures, flow_rates)
    plt.xlabel('Apertura (m)')
    plt.ylabel('Caudal (partículas/s)')
    plt.title('Caudal vs Apertura')
    plt.grid(True)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Plot guardado en: {out_path}")
    else:
        plt.show()

# ---------- MAIN ----------
def main(results_dir: str | None = None):
    # parámetros
    steps = 2000000
    omega = 500
    apertures = [0.03, 0.04, 0.05, 0.06]

    # descubrir rutas
    script_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(script_dir)

    # === If results_dir is provided, SKIP running sims and just parse ===
    if results_dir:
        base_results_dir = Path(results_dir).resolve()
        print(f"Reusing existing results at: {base_results_dir}")

        # If apertures not known, infer from subfolders aperture_*.***
        aperture_dirs = sorted((p for p in base_results_dir.iterdir()
                                if p.is_dir() and p.name.startswith("aperture_")))
        if not aperture_dirs:
            print("No aperture_* folders found in results_dir.")
            return

        # If you want to infer apertures from folder names:
        apertures = []
        for ad in aperture_dirs:
            try:
                apertures.append(float(ad.name.split("_")[1]))
            except Exception:
                pass
        apertures = sorted(apertures)

        results = []
        for aperture in apertures:
            aperture_dir = base_results_dir / f"aperture_{aperture:.3f}"
            exited_path = aperture_dir / "exited.txt"
            exit_timestamps = parse_exited_file(exited_path)
            num_escaped = len(exit_timestamps)
            flow_rate = calculate_flow_rate(num_escaped, steps)
            results.append((aperture, flow_rate))
            print(f"[REUSE] Aperture: {aperture:.3f}, Escaped: {num_escaped}, Flow: {flow_rate:.6e}")

        # guardar/plotear
        apertures_list = [a for a, _ in results]
        values_list = [v for _, v in results]

        summary_path = base_results_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({"omega": omega, "steps": steps, "results": results}, f, indent=2)
        print(f"Resumen guardado en: {summary_path}")

        plot_path = base_results_dir / "flow_vs_aperture.png"
        plot_flow_rate(values_list, apertures_list, out_path=plot_path)
        return

    # === Otherwise, run simulations as before ===
    jar_path = find_jar(repo_root, preferred_name="tp5")
    if jar_path is None or not jar_path.exists():
        print("❌ No se encontró el JAR en target/. Construí el proyecto primero, por ejemplo:")
        print("   mvn -q -DskipTests package")
        print("   # o con Gradle:")
        print("   ./gradlew jar")
        return

    print(f"Usando repo_root = {repo_root}")
    print(f"Usando jar_path  = {jar_path}")

    # resultados (new timestamped run)
    from datetime import datetime
    base_results_dir = repo_root / "results" / f"omega_{omega}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for aperture in apertures:
        print(f"\n=== Running simulation with aperture={aperture} ===")
        _ = run_simulation(steps, aperture, omega, jar_path=jar_path, repo_root=repo_root)

        aperture_dir = base_results_dir / f"aperture_{aperture:.3f}"
        aperture_dir.mkdir(parents=True, exist_ok=True)
        collect_output_files(repo_root, aperture_dir)

        exited_path = aperture_dir / "exited.txt"
        exit_timestamps = parse_exited_file(exited_path)
        num_escaped = len(exit_timestamps)
        flow_rate = calculate_flow_rate(num_escaped, steps)
        results.append((aperture, flow_rate))
        print(f"Aperture: {aperture:.3f}, Escaped: {num_escaped}, Flow Rate: {flow_rate:.6e}")

    apertures_list = [a for a, _ in results]
    values_list = [v for _, v in results]

    summary_path = base_results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"omega": omega, "steps": steps, "results": results}, f, indent=2)
    print(f"Resumen guardado en: {summary_path}")

    plot_path = base_results_dir / "flow_vs_aperture.png"
    plot_flow_rate(values_list, apertures_list, out_path=plot_path)

if __name__ == "__main__":
    # Example: reuse existing
    main(results_dir="../../../results/omega_500_1761871530.747757")
    # main()