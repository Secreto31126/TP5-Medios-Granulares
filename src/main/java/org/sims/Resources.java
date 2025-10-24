package org.sims;

import java.io.*;
import java.nio.file.Path;
import java.util.List;

public abstract class Resources {
    public static final String OUTPUT_PATH = Path.of("sim").toAbsolutePath().toString();

    /**
     * Initialize the resources system
     */
    public static void init() {
        prepareDir(true);
    }

    /**
     * Prepare a directory path for output
     *
     * @param preserve whether to preserve existing files
     * @param path     the path components
     */
    public static void prepareDir(boolean preserve, String... path) {
        final var directory = pathed(path).toFile();

        if (!directory.exists()) {
            directory.mkdirs();
        } else if (!preserve) {
            List.of(directory.listFiles()).parallelStream()
                    .filter(File::isFile)
                    .forEach(File::delete);
        }
    }

    /**
     * Prepare a directory path for output, deleting existing files
     *
     * @param path the path components
     */
    public static void prepareDir(String... path) {
        prepareDir(false, path);
    }

    /**
     * Get a buffered reader for a path
     *
     * @param path the path components
     * @return the buffered reader
     * @throws IOException if the file cannot be opened
     */
    public static BufferedReader reader(String... path) throws IOException {
        return new BufferedReader(new FileReader(pathed(path).toFile()));
    }

    /**
     * Get a buffered reader for a path
     *
     * @param path the path components
     * @return the buffered reader
     * @throws IOException if the file cannot be opened
     */
    public static BufferedWriter writer(String... path) throws IOException {
        return new BufferedWriter(new FileWriter(pathed(path).toFile()));
    }

    /**
     * Get a path from components, using the base path
     *
     * @param path the path components
     * @return the built path
     */
    private static Path pathed(String... path) {
        return Path.of(OUTPUT_PATH, path);
    }
}
