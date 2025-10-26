package org.sims.sand;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import java.util.random.*;

import org.sims.models.*;
import org.sims.neighbours.*;

public enum SandInitialization {
    RANDOM(SandInitialization::random);

    private static final double MASS = 0.001;
    private static final double MIN_RADIUS = 0.009;
    private static final double MAX_RADIUS = 0.011;

    private static final double WIDTH = 0.2;
    private static final double HEIGHT = 0.7;
    private static final double AMPLITUD = 0.0015;

    private static final double PORTAL = -(HEIGHT / 10f);
    private static final double RESPAWN_HEIGHT = 0.3;

    private static final RandomGenerator RNG = ThreadLocalRandom.current();

    private final BiConsumer<SandSimulation, CIM> initializer;

    SandInitialization(final BiConsumer<SandSimulation, CIM> initializer) {
        this.initializer = initializer;
    }

    public static CIM cim() {
        return new CIM(WIDTH, HEIGHT, MAX_RADIUS);
    }

    /**
     * Pick an initialization by name
     *
     * @param name The name of the initialization
     * @return the picked initialization
     */
    public static SandInitialization pick(final String name) {
        return SandInitialization.valueOf(name.toUpperCase());
    }

    /**
     * Initialize the simulation data
     *
     * @param simulation The simulation to initialize
     * @return the initialized data
     */
    public void initialize(final SandSimulation simulation) {
        try (final var cim = cim()) {
            this.initializer.accept(simulation, cim);
            simulation.integrator().initialize(simulation.entities(), new SandForce.Data(cim, simulation.box()));
        }
    }

    /**
     * Respawn particles that have exited through portals
     *
     * @implNote The CIM instance is expected to be already populated with the
     *           current particles
     *
     * @param particles The current particles
     * @param portals   The portals in the simulation
     * @param cim       The CIM for superposition checks
     * @return The updated list of particles after respawning
     */
    public static List<Particle> respawn(final Collection<Particle> particles, final Collection<Portal> portals,
            final CIM cim) {
        final var w = WIDTH;
        final var h = HEIGHT;
        final var res = RESPAWN_HEIGHT;

        final var updated = new ArrayList<Particle>(particles.size());

        for (final var p : particles) {
            if (portals.stream().noneMatch(portal -> portal.overlap(p) > 0)) {
                updated.add(p);
                continue;
            }

            Vector2 position;
            do {
                position = uniformVector(p.radius(), w - p.radius(), res, h - p.radius());
            } while (superposition(cim, position, p.radius()));

            final var new_p = new Particle(p, position, Vector2.ZERO, Vector2.ZERO);

            cim.add(new_p);
            updated.add(new_p);
        }

        return updated;
    }

    /**
     * Randomly populate the simulation with particles,
     * avoiding superpositions, and sets the walls and portals
     *
     * @param simulation The simulation to populate
     * @param cim        The CIM instance to use for superposition checks
     */
    private static void random(final SandSimulation simulation, final CIM cim) {
        particles(simulation, cim);
        box(simulation);
        portal(simulation);
    }

    /**
     * Generate the particles of the system
     *
     * @param simulation the simulation to populate
     */
    private static void particles(final SandSimulation simulation, final CIM cim) {
        final var w = WIDTH;
        final var h = HEIGHT;
        final var min = MIN_RADIUS;
        final var max = MAX_RADIUS;
        final var mass = MASS;

        for (long i = 0; i < simulation.n(); i++) {
            double radius;
            Vector2 position;

            do {
                radius = RNG.nextDouble(min, max);
                position = uniformVector(radius, w - radius, radius, h - radius);
            } while (i != 0 && superposition(cim, position, radius));

            final var p = new Particle(position, Vector2.ZERO, Vector2.ZERO, radius, mass);

            cim.add(p);
            simulation.entities().add(p);
        }
    }

    /**
     * Generate the box of the system
     *
     * @param simulation the simulation to populate
     */
    private static void box(final SandSimulation simulation) {
        final var w = WIDTH;
        final var h = HEIGHT;
        final var a = simulation.aperture();
        final var osc = AMPLITUD;
        // Bottom walls length
        final var bl = (w - a) / 2;

        final Function<Double, Double[]> vibration = (t) -> new Double[] {
                /* X(t) */
                osc * Math.sin(simulation.omega() * t),
                /* X'(t) = V(t) */
                osc * simulation.omega() * Math.cos(simulation.omega() * t),
        };

        simulation.box().addAll(List.of(
                new Wall(Orientation.HOR, Vector2.ZERO, new Vector2(bl, 0), vibration),
                new Wall(Orientation.HOR, new Vector2(bl + a, 0), new Vector2(w, 0), vibration),
                new Wall(Orientation.VER, new Vector2(w, -osc), new Vector2(w, h)),
                new Wall(Orientation.HOR, new Vector2(w, h), new Vector2(0, h)),
                new Wall(Orientation.VER, new Vector2(0, h), new Vector2(0, -osc))));
    }

    /**
     * Generate the portals of the system
     *
     * @param simulation the simulation to populate
     */
    private static void portal(final SandSimulation simulation) {
        simulation.portals().add(new Portal(Orientation.HOR, PORTAL));
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
