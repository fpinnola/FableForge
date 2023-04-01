
#include "Matrix.h"

int main(int argc, char const *argv[])
{
    /* code */
    Matrix a = Matrix(4,3);

    Matrix b = Matrix::randN(4,3);

    b.printMatrix();

    return 0;
}
