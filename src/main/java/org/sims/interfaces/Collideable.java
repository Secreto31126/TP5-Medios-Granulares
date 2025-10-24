package org.sims.interfaces;

import java.util.*;

/**
 * An interface for collideable objects
 */
public interface Collideable<E extends Collideable<E>> extends Identified, Named {
    /**
     * Calculates the time until collision with another Collideable
     *
     * @param c the other Collideable
     * @return the time until collision
     */
    double collisionTime(final E c);
    /**
     * Collides this Collideable with another Collideable
     *
     * @param c the other Collideable
     * @return the resulting Collideable after collision
     */
    List<E> collide(final E c);
}
