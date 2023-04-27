#include "Matrix.h"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include <math.h>
#include <cstring>

float randomNum(float Min, float Max) {
    return ((float(std::rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
};

Matrix::Matrix (int rows, int cols) : rows(rows), cols(cols) {
    data = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i * cols + j] = 0.0;
        }
    }
}

Matrix::Matrix(int rows, int cols, float* data) : rows(rows), cols(cols), data(data) {}

Matrix::Matrix (int rows, int cols, std::vector<std::vector<float>> v) : rows(rows), cols(cols) {
    data = (float*) malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i * cols + j] = v[i][j];
        }
    }
} 

Matrix::Matrix (const Matrix& other) {
    if (other.data) {
        data = (float*) malloc(other.rows * other.cols * sizeof(float));
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
    float * data = (float*) malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i * cols + j] = 0.0;
        }
    }
    Matrix a = Matrix(rows,cols,data);
    return a;
} 

Matrix Matrix::ones(int rows, int cols) {
    float * data = (float*) malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i * cols + j] = 1.0;
        }
    }
    return Matrix(rows,cols,data);
}

Matrix Matrix::randN(int rows, int cols) {
    // srand(time(0));
    float * data = (float*) malloc(rows * cols * sizeof(float));
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

float Matrix::get(int row, int col) const {
    if (!(row < rows && col < cols)) {
        throw std::out_of_range("Attempted to get value outside of matrix bounds");
    }
    return data[row * cols + col];
}

float Matrix::getMax() const {
    float max = 0.0;
    for (int i  = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (get(i, j) > max) {
                max = get(i,j);
            }
        }
    }
    return max;
}

float* Matrix::getVals() const {
    return data;
}

void Matrix::set(int row, int col, float value) {
    if (row < rows && col < cols) {
        data[row * cols + col] = value;
    }
}

void Matrix::setColToVal(int col, float value) {
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

void Matrix::prependVec (float value) {
    if (cols != 1) {
        throw std::invalid_argument("Matrix must be of dimension (n, 1)");
    }
    
    float * newData = (float*) malloc((rows + 1) * cols * sizeof(float));

    newData[0] = value;

    for (int i = 0; i < rows; i++) {
        newData[i+1] = get(i, 0);
    }

    rows = rows + 1;
    data = newData;

}

float Matrix::expSumVec() {
    if (cols != 1) {
        throw std::invalid_argument("Matrix must be of dimension (n, 1)");
    }

    float sum = 0.0;
    for (int i = 0; i < rows; i++) {
        sum += std::exp(get(i,0));
    }

    return sum;
}

float Matrix::sumVec() {
    if (cols != 1) {
        throw std::invalid_argument("Matrix must be of dimension (n, 1)");
    }

    float sum = 0.0;
    for (int i = 0; i < rows; i++) {
        sum += get(i,0);
    }

    return sum;
}

void Matrix::applyFunction(float func (float a)) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(i, j, func(get(i,j)));
        }
    }
}

void Matrix::applyFunction(float func (float a, float b), float c) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(i, j, func(get(i,j), c));
        }
    }
}

void Matrix::applyFunction(float func (float a, float b), Matrix c) {
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

Matrix Matrix::scalarMult(float b) {
    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.set(i,j, get(i,j) * b);
        }
    }

    return result;
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
    data = (float*) malloc(rows * cols * sizeof(float));
    memcpy(data, other.data, rows * cols * sizeof(float));

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
            float sum = 0.0;

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
