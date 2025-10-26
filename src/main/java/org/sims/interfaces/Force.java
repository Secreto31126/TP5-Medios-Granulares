package org.sims.interfaces;

import java.util.*;
import java.util.function.*;

import org.sims.models.*;

/**
 * A Force calculates the pull over every entity
 * in a collection. For each entity, it returns
 * a Vector3 representing the force acting on it.
 *
 * A force implementation must of course know the
 * type of entities it works with, which will
 * generally include information about position,
 * mass, velocity, charge, etc.
 *
 * @implNote The force MUST NOT alter the collection
 *           NOR the entities in it directly.
 *
 * @param <E> The type of entities the force works with.
 * @param <D> Additional data the force might need.
 */
public interface Force<E, D> extends BiFunction<Collection<E>, D, Map<E, Vector2>> {
}
