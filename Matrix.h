#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <vector>

class Matrix {
public:
    // Constructor
    Matrix (int rows, int cols);
    Matrix(int rows, int cols, double* data);
    Matrix (int rows, int cols, std::vector<std::vector<double>> v);
    Matrix (const Matrix& other);

    ~Matrix();

    static Matrix zeros(int rows, int cols);
    static Matrix ones(int rows, int cols);
    static Matrix randN(int rows, int cols);

    // // Getter methods

    int getRows() const;
    int getCols() const;
    double get(int row, int col) const;

    // // Setter methods
    void set(int row, int col, double value);
    void setColToVal(int col, double value);
    void setCol(int col, Matrix values);

    // // Other methods
    void prependVec (double value);
    double expSumVec();
    double sumVec();
    void applyFunction(double func (double a));
    void applyFunction(double func (double a, double b), double c);
    void printMatrix();
    void printSize();
    void printDataAddress();
    void printDataOne();

    static Matrix elemMult(Matrix a, Matrix b);
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix transpose() const;



private:
    int rows, cols;
    double* data;
};

#endif





