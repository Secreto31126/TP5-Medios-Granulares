package org.sims.sand;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import java.util.random.*;

import org.sims.integrals.Beeman;
import org.sims.interfaces.*;
import org.sims.models.*;
import org.sims.models.Wall.Orientation;
import org.sims.neighbours.CIM;

public record SandSimulation(long steps, List<Particle> entities, List<Wall> box, Integrator<Particle> integrator)
        implements Simulation<Particle, SandStep> {
    private static final RandomGenerator RNG = ThreadLocalRandom.current();
        
    static final double WIDTH = 20;
    static final double HEIGHT = 70;
    static final double APERTURE = 3;
    static final double AMPLITUD = 0.15;
    static final double DT = 1e-4;

    /**
     * Build a simulation
     *
     * @param steps the number of steps to simulate
     * @param n     the number of particles
     * @param omega the vibration frequency
     * @return the built simulation
     */
    public static SandSimulation build(final long steps, final int n, final double omega) {
        //
        final var entities = Initialization.RANDOM.initialize(n);
        final var force = new SandForce();
        final var integrator = new Beeman(entities, DT, force);

        return new SandSimulation(steps, entities, box(omega), integrator);
    }

    @Override
    public void saveTo(final Writer writer) throws IOException {
        writer.write(String.format(Locale.ROOT, "%d\n", steps));
    }

    /**
     * Generate the contour of the system
     *
     * @param omega the vibration frequency
     * @return collision time
     */
    private static List<Wall> box(double omega) {
        final var w = new ArrayList<Wall>(5);
        final var short_wall = (WIDTH - APERTURE) / 2;

        final Function<Double, Double> vibration = (t) -> AMPLITUD * Math.sin(omega * t);

        w.add(new Wall(Orientation.HOR, Vector2.ZERO, new Vector2(short_wall, 0), vibration));
        w.add(new Wall(Orientation.HOR, new Vector2(short_wall + APERTURE, 0), new Vector2(WIDTH, 0), vibration));
        w.add(new Wall(Orientation.VER, new Vector2(WIDTH, 0), new Vector2(WIDTH, HEIGHT)));
        w.add(new Wall(Orientation.HOR, new Vector2(WIDTH, HEIGHT), new Vector2(0, HEIGHT)));
        w.add(new Wall(Orientation.VER, new Vector2(0, HEIGHT), Vector2.ZERO));

        return w;
    }

    public enum Initialization {
        RANDOM(Initialization::random);

        private final Function<Integer, List<Particle>> initializer;

        Initialization(final Function<Integer, List<Particle>> initializer) {
            this.initializer = initializer;
        }

        public static Initialization pick(final String name) {
            return Initialization.valueOf(name.toUpperCase());
        }

        public List<Particle> initialize(final int n) {
            return this.initializer.apply(n);
        }

        private static List<Particle> random(final int n) {
            final var particles = new ArrayList<Particle>(n);

            try (final var cim = new CIM(WIDTH, HEIGHT, 0.01)) {
                for (long i = 0; i < n; i++) {
                    double radius;
                    Vector2 position;

                    do {
                        radius = RNG.nextDouble(0.9, 1.1);
                        position = uniformVector(radius, WIDTH - radius, radius, HEIGHT - radius);
                    } while (i != 0 && superposition(cim, position, radius));
                    final var p = new Particle(position, Vector2.ZERO, Vector2.ZERO, radius);

                    particles.add(p);
                    cim.mapping().add(p);
                }
            }

            return particles;
        }

        /**
         * Generate a uniform vector in the given ranges
         *
         * @param mx Min x
         * @param Mx Max x
         * @param my Min y
         * @param My Max y
         * @return the generated vector
         */
        private static Vector2 uniformVector(final double mx, final double Mx, final double my, final double My) {
            return new Vector2(
                    RNG.nextDouble(mx, Mx),
                    RNG.nextDouble(my, My));
        }

        /**
         * Check if a position is in supperposition with existing particles
         *
         * @param cim      the CIM instance containing populated mappings
         * @param position the position to check
         * @param radius   the radius of the new particle
         * @return true if in supperposition, false otherwise
         */
        private static boolean superposition(final CIM cim, final Vector2 position, final double radius) {
            final var ghost = new Particle(position, radius);
            return cim.evaluate(ghost) != null;
        }
    }
}
