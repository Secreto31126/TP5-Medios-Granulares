package org.sims;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;

import org.sims.interfaces.*;

/**
 * Orchestrates the simulation and engine
 *
 * @param simulation the simulation to run
 * @param engine     the engine to use
 * @param outputs    the output directories
 */
public record Orchestrator(Simulation<?, ?, ?> simulation, Engine<?> engine, List<String> outputs, List<String> logs) {
    /**
     * Start the simulation.
     *
     * OnStep is called on each step and can be used
     * to filter which steps to save.
     *
     * @apiNote Saves the setup and steps in the "steps" directory.
     * @apiNote The step 0 is always saved.
     *
     * @param onStep The OnStep event handler.
     */
    public void start(final OnStep onStep, final OnWrite onWrite) throws Exception {
        Resources.init();
        outputs.forEach(Resources::prepareDir);

        try (final var writer = Resources.writer("setup.txt")) {
            simulation.saveTo(writer);
        }

        try (
                final var animator = Executors.newFixedThreadPool(7);
                final var logger = Executors.newSingleThreadExecutor()) {
            Orchestrator.save(animator, engine.initial(), 0L, outputs, onWrite);
            engine.forEach(
                    step -> {
                        if (step.hasLogs())
                            Orchestrator.log(logger, step, logs);
                        onStep.apply(step).ifPresent(idx -> Orchestrator.save(animator, step, idx, outputs, onWrite));
                    });
        }
    }

    /**
     * A function called on each step.
     *
     * Returning empty optionals allow the caller to skip saving steps.
     * Similar to a filter, but it requires the idx to save to.
     */
    public interface OnStep extends Function<Step, Optional<Long>> {
    }

    /**
     * A function called on each step write.
     */
    public interface OnWrite extends Runnable {
    }

    /**
     * A simple OnStep implementation that skips saving every n steps
     * and notifies a callback on each execution.
     */
    public record SkipSteps(long n, Runnable callback) implements OnStep {
        @Override
        public Optional<Long> apply(final Step step) {
            callback.run();
            return step.i() % n == 0 ? Optional.of(step.i() / n) : Optional.empty();
        }
    }

    /**
     * Save a step with idx using an executor service
     */
    private static void save(final ExecutorService ex, final Step step, final Long idx, List<String> outputs,
            final OnWrite onWrite) {
        ex.submit(new Animator(step, idx, outputs, onWrite));
    }

    /**
     * Log a step using an executor service
     */
    private static void log(final ExecutorService ex, final Step step, List<String> outputs) {
        ex.submit(new Logger(step, outputs));
    }

    /**
     * A task to save an animation step
     */
    private static record Animator(Step step, Long idx, List<String> outputs, OnWrite onWrite) implements Runnable {
        @Override
        public void run() {
            final var filename = "%d.txt".formatted(idx);

            final var writers = new ArrayList<Writer>(outputs.size());
            try {
                for (final var dirname : outputs) {
                    writers.add(Resources.writer(dirname, filename));
                }

                step.saveTo(writers);

                onWrite.run();
            } catch (IOException e) {
            } finally {
                for (final var writer : writers) {
                    try {
                        writer.close();
                    } catch (final IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    /**
     * A task to save a log step
     */
    private static record Logger(Step step, List<String> outputs) implements Runnable {
        @Override
        public void run() {
            final var appenders = new ArrayList<Writer>(outputs.size());
            try {
                for (final var filename : outputs) {
                    appenders.add(Resources.appender(filename));
                }

                step.log(appenders);
            } catch (IOException e) {
            } finally {
                for (final var writer : appenders) {
                    try {
                        writer.close();
                    } catch (final IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }
}
