package org.sims.interfaces;

import org.sims.models.*;

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
     * Checks if this entity is overlaping with the other entity
     *
     * @param i the other Interactive
     * @return whether they are overlaping
     */
    default boolean overlaping(final E i) {
        return this.overlap(i) > 0;
    }

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
