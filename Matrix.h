#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <vector>
#include <random>

class Matrix {
public:
    // Constructor
    Matrix (int rows, int cols);
    Matrix(int rows, int cols, float* data);
    Matrix (int rows, int cols, std::vector<std::vector<float>> v);
    Matrix (const Matrix& other);

    ~Matrix();

    static Matrix zeros(int rows, int cols);
    static Matrix ones(int rows, int cols);
    static Matrix randN(int rows, int cols);
    static Matrix He(int rows, int cols, std::normal_distribution<float> distribution);

    // // Getter methods

    int getRows() const;
    int getCols() const;
    float get(int row, int col) const;
    float getMax() const;
    float getMin() const;
    float* getVals() const;

    // // Setter methods
    void set(int row, int col, float value);
    void setColToVal(int col, float value);
    void setCol(int col, Matrix values);
    void setMinMax(float mn, float mx); // Maximum values of elements in matrix

    // // Other methods
    void prependVec (float value);
    float expSumVec();
    float sumVec();
    void normalizeVec();
    void applyFunction(float func (float a));
    void applyFunction(float func (float a, float b), float c);
    void applyFunction(float func (float a, float b), Matrix c);
    void printMatrix();
    void printSize();
    void printDataAddress();
    void printDataOne();

    static Matrix elemMult(Matrix a, Matrix b);
    Matrix scalarMult(float b);
    Matrix& operator=(const Matrix& other) ;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix transpose() const;



private:
    int rows, cols;
    float* data;
    float min = -6.0;
    float max = 6.0;
};

#endif





