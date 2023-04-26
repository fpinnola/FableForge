#include <stdio.h>
#include "matrixGPU.cuh"

__global__ void add(float* a, float* b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

__global__ void subtract(float* a, float* b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a[idx] -= b[idx];
    }
}

#define TILE_WIDTH 32

__global__ void matMul(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    for (int k = 0; k < (TILE_WIDTH + ACols - 1)/TILE_WIDTH; k++) {

         if (k*TILE_WIDTH + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_WIDTH + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_WIDTH + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_WIDTH + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_WIDTH; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

__global__ void transpose(float* a, float* b, int height, int width) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx < width && idy < height) {
        b[idy * width + idx] = a[idx * width + idy];
    }
}


namespace MatrixGPU {
    float* matrixAdd(float* a, float* b, int size) {
        float *a_d;
        float *b_d;

        // Allocate Device memory
        cudaMalloc(&a_d, size * sizeof(float));
        cudaMalloc(&b_d, size * sizeof(float));

        // Copy to Device
        cudaMemcpy(a_d, a, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, size * sizeof(float), cudaMemcpyHostToDevice);

        // 1 d in nature
        // Blocksize = 1024
        // Num blocks = size / 1024
        int blockSize = 1024;
        int gridSize = size / 1024 + (size % 1024 != 0 );

        // execute Kernel
        add<<<gridSize, blockSize>>>(a_d, b_d, size);

        cudaMemcpy(a, a_d, size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(a_d);
        cudaFree(b_d);

        return a;
    }

    float* matrixSubtract(float* a, float* b, int size) {
        float *a_d;
        float *b_d;

        // Allocate Device memory
        cudaMalloc(&a_d, size * sizeof(float));
        cudaMalloc(&b_d, size * sizeof(float));

        // Copy to Device
        cudaMemcpy(a_d, a, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, size * sizeof(float), cudaMemcpyHostToDevice);

        // 1 d in nature
        // Blocksize = 1024
        // Num blocks = size / 1024
        int blockSize = 1024;
        int gridSize = size / 1024 + (size % 1024 != 0 );

        // execute Kernel
        subtract<<<gridSize, blockSize>>>(a_d, b_d, size);

        cudaMemcpy(a, a_d, size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(a_d);
        cudaFree(b_d);

        return a;
    }

    float* matrixMul(float* m, float* n, int m_height, int m_width, int n_height, int n_width) {

        int p_size = m_height * n_width;
        float *p_h = (float*)malloc(p_size * sizeof(float));
        float *m_d;
        float *n_d;
        float *p_d;


        printf("p_size: %i\n", p_size);

        // Allocate Device memory
        cudaMalloc(&m_d, m_height * m_width * sizeof(float));
        cudaMalloc(&n_d, n_height * n_width * sizeof(float));
        cudaMalloc(&p_d, m_height * n_width * sizeof(float));

        // Copy to Device
        cudaMemcpy(m_d, m, m_height * m_width * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(n_d, n, n_height * n_width * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_d, p_h, p_size * sizeof(float), cudaMemcpyHostToDevice);

        int gridWidth = ceil(m_height / 32.0);
        int gridHeight = ceil(n_width / 32.0);
        dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridSize(gridHeight, gridWidth, 1);

        printf("gridWidth: %i, gridHeight: %i\n", gridWidth, gridHeight);

        matMul<<<gridSize, blockSize>>>(m_d, n_d, p_d, m_height, m_width, n_height, n_width, m_height, n_width);

        cudaMemcpy(p_h, p_d, p_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(m_d);
        cudaFree(n_d);
        cudaFree(p_d);

        return p_h;
    }

    float* matrixTranspose(float* a, int height, int width) {
        float* b = (float*)malloc(height * width * sizeof(float));

        float* a_d;
        float* b_d;

        cudaMalloc(&a_d, a);
        cudaMalloc(&b_d, b);

        cudaMemcpy(a_d, a, height * width * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, height * width * sizeof(float), cudaMemcpyHostToDevice);


        dim3 blockSize(32,32,1);
        dim3 gridSize(ceil(width / 32.0), ceil(height / 32.0), 1);

        matrixTranspose<<<gridSize, blockSize>>>(a_d, b_d, height, width);

        cudaMemcpy(b, b_d, height * width * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(b_d);
        cudaFree(a_d);

        return b;
    }
}

