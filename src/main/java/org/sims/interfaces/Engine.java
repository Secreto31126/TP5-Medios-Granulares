package org.sims.interfaces;

/**
 * Engines are used to run simulations step by step.
 *
 * Most of the algorithm logic should NOT be here.
 * Engine is responsible for controlling the iteration
 * speed, generating the Steps, and invoking the
 * simulation methods as needed.
 *
 * They may know the entity type, but it isn't required.
 *
 * @apiNote
 * An Engine can only be used once. For multiple runs,
 * initiate multiple Engine instances, but be aware
 * that each engine might allocate N threads,
 * leaving none for others, so do this cautiously
 * if running them parallely.
 *
 * @param <S> the type of steps the engine produces.
 */
public interface Engine<S extends Step> extends Iterable<S>, AutoCloseable {
    /**
     * Get the initial step of the simulation
     *
     * @return the initial step
     */
    S initial();
}
