package org.sims.models;

import java.util.Iterator;
import java.util.function.Supplier;

public record Matrix<T>(int rows, int cols, T[][]data) implements Iterable<T> {
    @SuppressWarnings("unchecked")
    public Matrix(int rows, int cols) {
        this(rows, cols, (T[][]) new Object[rows][cols]);
    }

    public Matrix(int rows, int cols, T initialValue) {
        this(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = initialValue;
            }
        }
    }

    public Matrix(int rows, int cols, Supplier<T> supplier) {
        this(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = supplier.get();
            }
        }
    }

    public Matrix(int side) {
        this(side, side);
    }

    public Matrix(int side, T initialValue) {
        this(side, side, initialValue);
    }

    public Matrix(int side, Supplier<T> supplier) {
        this(side, side, supplier);
    }

    public T get(int row, int col) {
        return data[row][col];
    }

    public void set(int row, int col, T value) {
        data[row][col] = value;
    }

    @Override
    public Iterator<T> iterator() {
        return new Iterator<T>() {
            private int currentRow = 0;
            private int currentCol = 0;

            @Override
            public boolean hasNext() {
                return currentRow < rows && currentCol < cols;
            }

            @Override
            public T next() {
                T value = data[currentRow][currentCol];

                if (++currentCol >= cols) {
                    currentCol = 0;
                    currentRow++;
                }

                return value;
            }
        };
    }
}
