package org.sims.sand;

import java.io.*;
import java.util.*;

import org.sims.integrals.*;
import org.sims.interfaces.*;
import org.sims.models.*;

public record SandSimulation(long steps, long n, double omega, double aperture,
        List<Particle> entities, List<Wall> box, List<Portal> portals,
        Integrator<Particle, SandForce.Data> integrator, SandRespawn teleport)
        implements Simulation<Particle, SandForce.Data, SandStep> {
    private static final double DT = 1e-4;

    /**
     * Build a simulation
     *
     * @param steps    the number of steps to simulate
     * @param n        the number of particles
     * @param aperture the box aperture
     * @param omega    the vibration frequency
     * @return the built simulation
     */
    public static SandSimulation build(final long steps, final int n, final double aperture, final double omega) {
        final var force = new SandForce();
        final var integrator = new Beeman<SandForce.Data>(DT, force);

        final var simulation = new SandSimulation(steps, n, omega, aperture,
                new ArrayList<Particle>(n), new ArrayList<Wall>(), new ArrayList<Portal>(),
                integrator, SandInitialization::respawn);

        SandInitialization.RANDOM.initialize(simulation);

        return simulation;
    }

    /**
     * Get the time step of the simulation
     *
     * @return the time step
     */
    public double dt() {
        return DT;
    }

    @Override
    public void saveTo(final Writer writer) throws IOException {
        writer.write(String.format(Locale.ROOT, "%d %+.14f %+.14f\n", steps, aperture, omega));
    }
}
