#include <vector>
#include "Matrix.h"

namespace MatrixGPU
{
    float* matrixAdd(float* a, float* b, int size);
    float* matrixSubtract(float* a, float* b, int size);
    float* matrixMul(float* m, float* n, int m_height, int m_width, int n_height, int n_width) ;
    float* matrixTranspose(float* a, int height, int width);
    float* storeOnDevice(float* a, int size);
    float* removeFromDevice(float* a_d, int size);
    Matrix forwardPass(std::vector<Matrix> W_d, std::vector<Matrix> b_d, std::vector<Matrix>z_d, std::vector<Matrix>a_d, float* X, float* Y, int inputSize, int outputSize);

} // namespace MatrixGPU
