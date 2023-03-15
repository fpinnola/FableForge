#include <iostream>
#include <stdio.h>
#include <vector>
#include <time.h>
#include <algorithm> 

// CPU Matrix Operations
// Implement operations as CUDA kernels to optimize

double RandomNumber(double Min, double Max)
{
    return ((double(std::rand()) / double(RAND_MAX)) * (Max - Min)) + Min;
}

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
        srand(time(0));
        double * data = (double*) malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = RandomNumber(-10.0, 10.0);
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

    // In place operations

    void prependVec (double value) {
        if (cols != 1) {
            throw std::invalid_argument("Matrix must be of dimension (n, 1)");
        }
        
        double * newData = (double*) malloc((rows + 1) * cols * sizeof(double));

        newData[0] = value;

        for (int i = 0; i < rows; i++) {
            newData[i+1] = get(i, 0);
        }

        rows = rows + 1;
        data = newData;

    }

    void applyFunction(double func (double a)) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                set(i, j, func(get(i,j)));
            }
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


double leakyRelu(double a) {
    return std::max(0.01 * a, a);
}


int main(int argc, char const *argv[])
{
    // Forward Pass test

    Matrix input  = Matrix(3, 1, { {5.4}, {-0.9}, {4.3}});

    int inputFeatures = 3;

    int h1Nodes = 1024;

    int h2Nodes = 4096;

    int h3Nodes = 360;


    // FORWARD PASS

    // TODO: Cache each layer's output
    Matrix theta1 = Matrix::randN(h1Nodes, inputFeatures + 1);
    input.prependVec(1.0); // Add bias
    // input.applyFunction(leakyRelu);
    // input.printMatrix();
    Matrix z1 = theta1*input;
    z1.applyFunction(leakyRelu);
    // z1.printMatrix();

    Matrix theta2 = Matrix::randN(h2Nodes, h1Nodes+1);
    z1.prependVec(1.0);
    Matrix z2 =  theta2 * z1;
    z2.applyFunction(leakyRelu);

    // z2.printMatrix();

    Matrix theta3 = Matrix::randN(h3Nodes, h2Nodes+1);
    z2.prependVec(1.0);
    Matrix z3 = theta3 * z2;
    z3.applyFunction(leakyRelu);

    // z3.printMatrix();

    Matrix theta4 = Matrix::randN(3, h3Nodes+1);
    z3.prependVec(1.0);
    Matrix z4 = theta4 * z3;
    z4.printMatrix();
    z4.applyFunction(leakyRelu);
    z4.printMatrix();


    return 0;
}
