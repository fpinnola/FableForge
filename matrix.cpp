#include "Matrix.h"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include <math.h>

double randomNum(double Min, double Max) {
    return ((double(std::rand()) / double(RAND_MAX)) * (Max - Min)) + Min;
};

Matrix::Matrix (int rows, int cols) : rows(rows), cols(cols) {
    data = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i * cols + j] = 0.0;
        }
    }
}

Matrix::Matrix(int rows, int cols, double* data) : rows(rows), cols(cols), data(data) {}

Matrix::Matrix (int rows, int cols, std::vector<std::vector<double>> v) : rows(rows), cols(cols) {
    data = (double*) malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i * cols + j] = v[i][j];
        }
    }
} 

Matrix::Matrix (const Matrix& other) {
    if (other.data) {
        data = (double*) malloc(other.rows * other.cols * sizeof(double));
        for (int i = 0; i < other.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                data[i * other.cols + j] = other.data[i * other.cols + j];
            }
        }
        // data = other.data;
        rows = other.rows;
        cols = other.cols;
    }
}


Matrix Matrix::zeros(int rows, int cols) {
    double * data = (double*) malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i * cols + j] = 0.0;
        }
    }
    Matrix a = Matrix(rows,cols,data);
    return a;
} 

Matrix Matrix::ones(int rows, int cols) {
    double * data = (double*) malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i * cols + j] = 1.0;
        }
    }
    return Matrix(rows,cols,data);
}

Matrix Matrix::randN(int rows, int cols) {
    srand(time(0));
    double * data = (double*) malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i * cols + j] = randomNum(-0.15, 0.15);
        }
    }
    Matrix a = Matrix(rows,cols,data);
    return a;
}

Matrix::~Matrix() {
    if (rows > 0 && cols > 0) {
        free(data);
        rows = 0;
        cols = 0;
    }
}

void Matrix::printMatrix() {
    for (int i = 0; i < rows; i++) {
        printf("\n");
        for (int j = 0; j < cols; j++) {
            printf("%f ", data[i * cols + j]);
        }
    }
    printf("\n");
}

int Matrix::getRows() const {
    return rows;
}

int Matrix::getCols() const {
    return cols;
}

double Matrix::get(int row, int col) const {
    if (!(row < rows && col < cols)) {
        throw std::out_of_range("Attempted to get value outside of matrix bounds");
    }
    return data[row * cols + col];
}

double Matrix::getMax() const {
    double max = 0.0;
    for (int i  = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (get(i, j) > max) {
                max = get(i,j);
            }
        }
    }
    return max;
}

void Matrix::set(int row, int col, double value) {
    if (row < rows && col < cols) {
        data[row * cols + col] = value;
    }
}

void Matrix::setColToVal(int col, double value) {
    if (col < cols ) {
        for (int i = 0; i < rows; i++) {
            data[i * cols + col] = value;
        }
    }
}

void Matrix::setCol(int col, Matrix values) {
    if (col < cols && rows == values.getRows()) {
        for (int i = 0; i < rows; i++) {
            data[i * cols + col] = values.get(i, col);
        }
    }
}

void Matrix::prependVec (double value) {
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

double Matrix::expSumVec() {
    if (cols != 1) {
        throw std::invalid_argument("Matrix must be of dimension (n, 1)");
    }

    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        sum += std::exp(get(i,0));
    }

    return sum;
}

double Matrix::sumVec() {
    if (cols != 1) {
        throw std::invalid_argument("Matrix must be of dimension (n, 1)");
    }

    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        sum += get(i,0);
    }

    return sum;
}

void Matrix::applyFunction(double func (double a)) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(i, j, func(get(i,j)));
        }
    }
}

void Matrix::applyFunction(double func (double a, double b), double c) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(i, j, func(get(i,j), c));
        }
    }
}

void Matrix::applyFunction(double func (double a, double b), Matrix c) {
    if (rows != c.getRows() || cols != c.getCols()) {
        throw std::invalid_argument("Matrix dimsensions must match!");
    };
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(i, j, func(get(i,j), c.get(i,j)));
        }
    }
}

void Matrix::printSize() {
    printf("(%i,%i)\n", rows, cols);
}

void Matrix::printDataAddress() {
    printf("address, %p\n", (void*)&data);
}

void Matrix::printDataOne() {
    printf("1(%i,%i)data %f\n", rows, cols, data[0]);
}

Matrix Matrix::elemMult(Matrix a, Matrix b) {
    if (b.getRows() != a.getRows() || b.getCols() != a.getCols()) {
        throw std::invalid_argument("Matrix dimsension must match for element wise multiplication");
    }

    Matrix result(a.rows, a.cols);


    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result.set(i,j, a.get(i,j) * b.get(i,j));
        }
    }


    return result;

}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) {
        return *this;
    }

    free(data);
    rows = other.rows;
    cols = other.cols;
    data = (double*) malloc(rows * cols * sizeof(double));
    memcpy(data, other.data, rows * cols * sizeof(double));

    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
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

Matrix Matrix::operator-(const Matrix& other) const {
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

Matrix Matrix::operator*(const Matrix& other) const {
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

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.set(j, i, get(i,j));
        }
    }
    return result;
}
