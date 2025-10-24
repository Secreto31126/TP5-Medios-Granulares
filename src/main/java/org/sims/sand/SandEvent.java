package org.sims.sand;

import java.util.stream.Stream;

import org.sims.interfaces.Collideable;
import org.sims.models.Particle;

public record SandEvent(Particle p, Collideable<Particle> c, double time, long etag)
        implements Comparable<SandEvent> {
    public SandEvent(Particle p, Collideable<Particle> c, double time) {
        this(p, c, time, SandEvent.etag(p, c));
    }

    /**
     * Generates an event tag based on the number of events each particle has
     *
     * @param p A particle
     * @param c A collideable
     * @return The event tag
     */
    private static long etag(final Particle p, final Collideable<Particle> c) {
        var etag = p.events().get();
        if (c instanceof final Particle p2) {
            etag += p2.events().get();
        }
        return etag;
    }

    public boolean valid(double currentTime) {
        return this.time >= currentTime && valid();
    }

    private boolean valid() {
        return this.etag == SandEvent.etag(p, c);
    }

    public Stream<Particle> involved() {
        if (c instanceof final Particle p2) {
            return Stream.of(p, p2);
        }

        return Stream.of(p);
    }

    public void execute() {
        if (valid())
            c.collide(p);
    }

    @Override
    public final String toString() {
        return "%.14f %s %d %d".formatted(this.time, this.c.name(), this.p.id(), this.c.id());
    }

    @Override
    public int compareTo(SandEvent o) {
        return Double.compare(this.time, o.time);
    }
}
