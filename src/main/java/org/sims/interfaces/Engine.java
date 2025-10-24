package org.sims.interfaces;

/**
 * Engines are used to run simulations step by step.
 *
 * Most of the algorithm logic should NOT be here.
 * Engine is responsible for controlling the iteration
 * speed, generating the Steps, and invoking the
 * integrator as needed.
 *
 * They may know the entity type, but it isn't required.
 */
public interface Engine<S extends Step> extends Iterable<S>, AutoCloseable {
    /**
     * Get the initial step of the simulation
     *
     * @return the initial step
     */
    S initial();
}
