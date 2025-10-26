package org.sims.sand;

import java.io.*;
import java.util.*;

import org.sims.integrals.*;
import org.sims.interfaces.*;
import org.sims.models.*;

public record SandSimulation(long steps, long n, double aperture, double omega,
        List<Particle> entities, List<Wall> box, List<Portal> portals,
        SandRespawn teleport)
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
    public SandSimulation(final long steps, final int n, final double aperture, final double omega) {
        this(steps, n, aperture, omega,
                new ArrayList<Particle>(n), new ArrayList<Wall>(), new ArrayList<Portal>(),
                SandInitialization::respawn);
        SandInitialization.RANDOM.initialize(this);
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
    public Integrator<Particle, SandForce.Data> integrator(final SandForce.Data data) {
        return new Beeman<>(entities, DT, new SandForce(), data);
    }

    @Override
    public void saveTo(final Writer writer) throws IOException {
        writer.write(String.format(Locale.ROOT, "%d %+.14f %+.14f\n", steps, aperture, omega));
    }
}
