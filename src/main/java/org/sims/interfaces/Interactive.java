package org.sims.interfaces;

import org.sims.models.Vector2;

/**
 * An interface for interactive objects in the simulation
 */
public interface Interactive<E extends Interactive<E>> extends Identified, Named {
    /**
     * Calculates the overlap distance with the entity
     *
     * @apiNote Overlap with self is zero.
     *
     * @param i the other Interactive
     * @return the overlap distance
     */
    double overlap(final E i);
    /**
     * Get the relative position to the entity
     *
     * @param i the other Interactive
     * @return relative position vector
     */
    Vector2 position(final E i);
    /**
     * Get the relative velocity to the entity
     *
     * @param i the other Interactive
     * @return relative velocity vector
     */
    Vector2 velocity(final E i);
    /**
     * Get the normal vector between this and the entity
     *
     * @param i the other Interactive
     * @return normal vector
     */
    Vector2 normal(final E i);
}
