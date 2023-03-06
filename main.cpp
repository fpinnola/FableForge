#include <iostream>
#include <stdio.h>

// CPU Matrix Operations
// Implement operations as CUDA kernels to optimize

class Matrix {
private:
    int rows, cols;
    double* data;

public:
    // Constructors
    Matrix (int rows, int cols) : rows(rows), cols(cols) {
        data = (double*)malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = 1.0 * (i * cols + j);
            }
        }
        for (int i = 0; i < rows * cols; i++) {
            data[i] = 1.0 * (i);
        }
    }

    Matrix(int rows, int cols, double* data) : rows(rows), cols(cols), data(data) {}

    // Destructor
    ~Matrix() {
        delete[] data;
    }

    // Getter and Setter
    double get(int row, int col) const {
        if (!(row < rows && col < cols)) {
            throw std::out_of_range("Attempted to get value outside of matrix bounds");
        }
        return data[row * cols + col];
    }

    void set(int row, int col, double value) {
        if (row < rows && col < cols) {
            data[row * cols + col] = value;
        }
    }

    // Display
    void printMatrix() {
        for (int i = 0; i < rows; i++) {
            printf("\n");
            for (int j = 0; j < cols; j++) {
                printf("%f ", data[i * cols + j]);
            }
        }
    }

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimsension must match for addition");
        }

        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i,j, get(i,j) + other.get(i,j));
            }
        }

        return result;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimsension must match for subtraction");
        }

        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i,j, get(i,j) - other.get(i,j));
            }
        }

        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions must match (m,n) (n,p)");
        }

        Matrix result(rows, other.cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0.0;

                for (int k = 0; k < other.rows; k++) {
                    sum += get(i,k) * other.get(k,j);
                }

                result.set(i, j, sum);
            }
        }
        return result;
    }

    Matrix transpose() const {
        Matrix result(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(j, i, get(i,j));
            }
        }
        return result;
    }

};





int main(int argc, char const *argv[])
{
    Matrix a = Matrix(3, 2);
    Matrix b = Matrix(2,2);
    Matrix c = a*b;
    Matrix d = b+b;
    Matrix e = d.transpose();
    a.printMatrix();
    printf("\n\n");
    b.printMatrix();
    printf("\n\n");
    c.printMatrix();
    printf("\n\n");
    d.printMatrix();
    printf("\n\n");
    e.printMatrix();
    printf("\n\n");

    return 0;
}
