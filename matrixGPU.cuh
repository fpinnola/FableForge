

namespace MatrixGPU
{
    float* matrixAdd(float* a, float* b, int size);
    float* matrixSubtract(float* a, float* b, int size);
    float* matrixMul(float* m, float* n, int m_height, int m_width, int n_height, int n_width) ;
    float* matrixTranspose(float* a, int height, int width);
} // namespace MatrixGPU
