package org.sims.interfaces;

import java.util.*;

import org.sims.models.Particle;

/**
 * Integrators are the mathematical implementations of the
 * time evolution algorithms, such as Verlet, or Beeman.
 *
 * As the responsible of moving the entities, it must know
 * the entities type and the memory they need to save,
 * hence the existance of a nested Constructor interface.
 *
 * Constructor requires the user to provide a static class
 * that creates the integrator instance and initializes
 * the entities' memories as deemed necesary by the
 * algorithm.
 *
 * @param <E> the type of entities the integrator works with.
 */
public interface Integrator<E> extends Named {
    /**
     * Advance the simulation by one time step
     *
     * @implNote The integrator MUST NOT alter the collection
     *           NOR the entities in it directly.
     *
     * @param entities the entities to move
     * @return the moved entities
     */
    List<E> step(final Collection<E> entities);

    /**
     * Optional method to gear up the integrator
     * if it needs to initialize some memory
     * based on the simulation data.
     *
     * @param k Spring constant
     * @param gamma TODO: Idk
     * @param mass Mass of the particle
     * @param particles The particles being simulated
     */
    default void gearUp(final double k, final double gamma, final double mass, final Collection<Particle> particles) {}

    /**
     * A constructor for integrators
     *
     * @param <E> the type of entities the integrator works with
     */
    public interface Constructor<E> {
        /**
         * Create an integrator instance
         *
         * @param entities the entities to integrate
         * @param dt    the time step
         * @param force the force calculator
         * @return the integrator instance
         */
        Integrator<E> get(final Collection<E> entities, final double dt, final Force<E> force);
    }
}
