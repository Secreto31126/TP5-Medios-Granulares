package org.sims.models;

public enum Orientation {
    VER(Vector2.ONE_ZERO, Vector2.ZERO_ONE),
    HOR(Vector2.ZERO_ONE, Vector2.ONE_ZERO);

    private final Vector2 normal;
    private final Vector2 tangencial;

    Orientation(final Vector2 normal, final Vector2 tangencial) {
        this.normal = normal;
        this.tangencial = tangencial;
    }

    /**
     * Get the constant coordinate of a vector according to the orientation type
     *
     * @param v vector to get coordinate from
     * @return constant value
     */
    public double constant(final Vector2 v) {
        return v.dot(normal);
    }

    /**
     * Get the perpendicular coordinate of a vector according to the orientation
     * type
     *
     * @param v vector to get coordinate from
     * @return perpendicular value
     */
    public double perpendicular(final Vector2 v) {
        return constant(v);
    }

    /**
     * Get the variable coordinate of a vector according to the orientation type
     *
     * @param v vector to get coordinate from
     * @return variable value
     */
    public double variable(final Vector2 v) {
        return v.dot(tangencial);
    }

    /**
     * Get the parallel coordinate of a vector according to the orientation type
     *
     * @param v vector to get coordinate from
     * @return parallel value
     */
    public double parallel(final Vector2 v) {
        return variable(v);
    }

    /**
     * Get the normal vector according to the orientation type
     *
     * @return normal vector
     */
    public Vector2 normal() {
        return normal;
    }

    /**
     * Get the tangencial vector according to the orientation type
     *
     * @return tangencial vector
     */
    public Vector2 tangencial() {
        return tangencial;
    }

    /**
     * Collide a velocity vector according to the orientation
     *
     * @param v vector to collide
     * @return collided vector
     */
    public Vector2 collide(final Vector2 v) {
        return v.hadamard(normal).neg().add(v.hadamard(tangencial));
    }

    /**
     * Move a vector according to the orientation
     *
     * @param v        vector to move
     * @param constant new constant displacement
     * @return moved vector
     */
    public Vector2 move_constant(final Vector2 v, double constant) {
        return this.normal.mult(constant).add(v.hadamard(tangencial));
    }
}
