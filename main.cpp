#include <iostream>
#include <stdio.h>
#include <vector>

// CPU Matrix Operations
// Implement operations as CUDA kernels to optimize

class Matrix {
private:
    int rows, cols;
    double* data;

public:
    // Constructors
    static Matrix zeros(int rows, int cols) {
        double * data = (double*) malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = 0.0;
            }
        }
        Matrix a = Matrix(rows,cols,data);
        return a;
    } 

    static Matrix ones(int rows, int cols) {
        double * data = (double*) malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = 1.0;
            }
        }
        Matrix a = Matrix(rows,cols,data);
        return a;
    } 

    static Matrix randN(int rows, int cols) {
        double * data = (double*) malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = std::rand();
            }
        }
        Matrix a = Matrix(rows,cols,data);
        return a;
    }

    Matrix (int rows, int cols) : rows(rows), cols(cols) {
        data = (double*)malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = 0.0;
            }
        }
    }

    Matrix(int rows, int cols, double* data) : rows(rows), cols(cols), data(data) {}

    Matrix (int rows, int cols, std::vector<std::vector<double>> v) : rows(rows), cols(cols) {
        data = (double*) malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = v[i][j];
            }
        }
    } 

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
    Matrix a = Matrix::ones(24, 15);
    Matrix b = Matrix::zeros(15,18);
    Matrix x = Matrix(5,10);
    Matrix c = a*b;
    Matrix d = b+b;
    Matrix e = d.transpose();

    std::vector<std::vector<double>> v =    {{1.1, 2.4, 3.5},
                                            {3.7, -0.9, 0.01}};

    Matrix y = Matrix(v.size(), v[0].size(), v);
    y.printMatrix();

    // double * inputV = &std::vector<double>()[0];

    // a.printMatrix();
    // printf("\n\n");
    // b.printMatrix();
    // printf("\n\n");
    // c.printMatrix();
    // printf("\n\n");
    // d.printMatrix();
    // printf("\n\n");
    // e.printMatrix();
    // printf("\n\n");

    return 0;
}
