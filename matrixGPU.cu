#include <stdio.h>
#include "matrixGPU.cuh"
#include <cmath>
// #include <vector>
#include <iostream>

float softmax2(float z, float sum) {
    float sftmax = std::exp(z) / (sum);
    if (std::isnan(sftmax)){
        // printf("nan\n");
        // printf("z: %f\n", z);
        // printf("sum: %f\n", sum);
    }
    return std::exp(z) / (sum);
};

float leakyRelu2(float a) {
    return std::max(0.01f * a, a);
};

#define CUDA_CALL(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Adds inplace to float* a
__global__ void add(float* a, float* b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

// Subtracts inplace to float* a
__global__ void subtract(float* a, float* b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a[idx] -= b[idx];
    }
}

#define TILE_WIDTH 32


#define TILE_SIZE 32

__global__ void matrixMultiplicationKernel(float *a, float *b, float *c, int a_rows, int a_cols, int b_cols) {
    // Calculate row and column indices for the current thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0;

    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    // Iterate through the tiles of the input matrices.
    for (int tileIdx = 0; tileIdx < (a_cols + TILE_SIZE - 1) / TILE_SIZE; tileIdx++) {
        // Declare shared memory tiles.

        // Load data into shared memory.
        if (row < a_rows && tileIdx * TILE_SIZE + threadIdx.x < a_cols) {
            tile_a[threadIdx.y][threadIdx.x] = a[row * a_cols + tileIdx * TILE_SIZE + threadIdx.x];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < b_cols && tileIdx * TILE_SIZE + threadIdx.y < a_cols) {
            tile_b[threadIdx.y][threadIdx.x] = b[(tileIdx * TILE_SIZE + threadIdx.y) * b_cols + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }

        // Synchronize to make sure the matrices are loaded.
        __syncthreads();

        // Multiply the tiles.
        int k_end = min(TILE_SIZE, a_cols - tileIdx * TILE_SIZE);
        for (int k = 0; k < k_end; k++) {
            value += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        // Synchronize to make sure the computation is done before loading the next tile.
        __syncthreads();
    }

    // Write the result to the output matrix.
    if (row < a_rows && col < b_cols) {
        c[row * b_cols + col] = value;
    }
}
// Multiplies to C
__global__ void matMul(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols)
{
    float CValue = 0.0;

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

    if (Row < CRows && Col < CCols) {
        // printf("Row: %i, Col: %i, Val: %f\n", Row, Col, CValue);
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
    }

}

#define TRANSPOSE_BLOCK_DIM 16
__global__ void transpose(float *odata, float *idata, int width, int height) 
{
	__shared__ float block[TRANSPOSE_BLOCK_DIM][TRANSPOSE_BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * TRANSPOSE_BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * TRANSPOSE_BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

        // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * TRANSPOSE_BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * TRANSPOSE_BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

__global__ void leakyReluGPU(float *in, float* out, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        printf("idx: %i, val: %f\n", idx, in[idx]);
        if (in[idx] < 0.0) {
            out[idx] = 0.01 * in[idx];
        } else {
            out[idx] = in[idx];
        }
    }
}

__global__ void softmax_kernel(const float * input, float* res, float vecSum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("idx: %i, N: %i\n", idx, N);
    if (idx < N) {
        // printf("i: %i, elem: %f, exp(i): %f\n", idx, input[idx], expf(input[idx]));
        res[idx] = expf(input[idx]) / vecSum;
    }
}

__global__ void elementWiseMultKernel(float * a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void scalarMultKernel(float *a, float b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        c[idx] = a[idx] * b;
    }
}

__global__ void sumKernel(float* a, float* out, int size) {
    float total = 0.0;
    for (int i  = 0; i < size; i++) {
        total += a[i];
    }
    out[0] = total;
}

__global__ void assignScalarKernel(float* a, float b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        a[idx] = b;
    }
}

// __global__ void exp_sum_kernel(const float *input, float *result, int N) {
//     // Calculate the global thread index
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     // Shared memory to store the partial sums of each block
//     __shared__ float shared_sum[1024];

//     // Initialize shared memory to 0
//     shared_sum[threadIdx.x] = 0.0f;

//     // Check if the index is within the input range
//     if (idx < N) {
//         // Apply std::exp to the input value and store in shared memory
//         shared_sum[threadIdx.x] = std::exp(input[idx]);
//     }
//     __syncthreads();

//     // Perform a parallel reduction within the block
//     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//         if (threadIdx.x < stride) {
//             shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
//         }
//         __syncthreads();
//     }

//     // Store the sum of each block in the result array
//     if (threadIdx.x == 0) {
//         atomicAdd(result, shared_sum[0]);
//     }
// }

// __global__ void expSumGPU(....., int* iter_result, int iter_num) {

//     // Your calculations first so that each thread holds its result

//     // Block wise reduction so that one thread in each block holds sum of thread results

//     // The one thread holding the adds the block result to the global iteration result
//     if (threadIdx.x == 0)
//         atomicAdd(iter_result + std::exp(iter_num), block_ressult);
// }

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


        // Allocate Device memory
        cudaMalloc(&m_d, m_height * m_width * sizeof(float));
        cudaMalloc(&n_d, n_height * n_width * sizeof(float));
        cudaMalloc(&p_d, m_height * n_width * sizeof(float));

        // Copy to Device
        cudaMemcpy(m_d, m, m_height * m_width * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(n_d, n, n_height * n_width * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_d, p_h, p_size * sizeof(float), cudaMemcpyHostToDevice);

        int gridHeight = ceil(n_width / 32.0);
        int gridWidth = ceil(m_height / 32.0);
        dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridSize(gridHeight, gridWidth, 1);

        matMul<<<gridSize, blockSize>>>(m_d, n_d, p_d, m_height, m_width, n_height, n_width, m_height, n_width);

        cudaMemcpy(p_h, p_d, p_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(m_d);
        cudaFree(n_d);
        cudaFree(p_d);

        return p_h;
    }

    float* storeOnDevice(float* a, int size) {
        float *a_d;

        CUDA_CALL(cudaMalloc(&a_d, size*sizeof(float)));

        CUDA_CALL(cudaMemcpy(a_d, a, size * sizeof(float), cudaMemcpyHostToDevice));

        return a_d;
    }

    float* removeFromDevice(float* a_d, int size) {
        float* a_h = (float*)malloc(size * sizeof(float));

        cudaMemcpy(a_h, a_d, size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(a_d);
        return a_h;
    }



    Matrix forwardPass(std::vector<Matrix> W_d, std::vector<Matrix> b_d, std::vector<Matrix>z_d, std::vector<Matrix>a_d, float* X, float* Y, int inputSize, int outputSize) {
        // printf("Starting forward pass!\n");

        for (int i = 0; i < W_d.size(); i++) {
            // float* a_prev_d;
            if (i == 0) {
                // cudaMalloc(&a_prev_d, inputSize * sizeof(float));
                cudaMemcpy(a_d[i].getDeviceData(), X, inputSize * sizeof(float), cudaMemcpyHostToDevice);
            } 
            // else {
            //     a_prev_d = a_d[i].getDeviceData();
            // }

            // float* a_prev_d_temp = (float*)malloc(a_d[i].getRows() * sizeof(float));
            // cudaMemcpy(a_prev_d_temp, a_d[i].getDeviceData(), a_d[i].getRows() * sizeof(float), cudaMemcpyDeviceToHost);
            // Matrix a_prev_d_temp_mat = Matrix(a_d[i].getRows(), 1, a_prev_d_temp);
            // printf("a[%i]\n", i);
            // a_prev_d_temp_mat.printMatrix();
            // free(a_prev_d_temp);



            // float* w_temp = (float*)malloc(W_d[i].getRows() * W_d[i].getCols() * sizeof(float));
            // CUDA_CALL(cudaMemcpy(w_temp, W_d[i].getDeviceData(), W_d[i].getRows() * W_d[i].getCols() * sizeof(float), cudaMemcpyDeviceToHost));
            // Matrix w_temp_mat = Matrix(W_d[i].getRows(), W_d[i].getCols(), w_temp);
            // printf("W_d[%i]\n", i);
            // w_temp_mat.printMatrix();
            // free(w_temp);

            
            // Matrix z_temp = (W[i].transpose())*a_prev + b[i];
            // transpose matrix W[i] in float* W_i_t
            float* W_i_t_d;
            cudaMalloc(&W_i_t_d, W_d[i].getRows() * W_d[i].getCols() * sizeof(float));
            dim3 grid(ceil(W_d[i].getCols() / TRANSPOSE_BLOCK_DIM), ceil(W_d[i].getRows() / TRANSPOSE_BLOCK_DIM), 1);
            dim3 threads(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM, 1);
            transpose<<<grid, threads>>>(W_i_t_d, W_d[i].getDeviceData(), W_d[i].getCols(), W_d[i].getRows());
            CUDA_CALL(cudaDeviceSynchronize());

            // float* w_d_temp = (float*)malloc(W_d[i].getRows() * W_d[i].getCols() * sizeof(float));
            // std::cout << "Loading Device data W[" << i << "]: " << W_d[i].getDeviceData() << std::endl;
            // CUDA_CALL(cudaMemcpy(w_d_temp, W_i_t_d, W_d[i].getRows() * W_d[i].getCols() * sizeof(float), cudaMemcpyDeviceToHost));
            // Matrix w_d_temp_mat = Matrix(W_d[i].getCols(), W_d[i].getRows(), w_d_temp);
            // printf("W_d[%i].T\n", i);
            // w_d_temp_mat.printMatrix();
            // free(w_d_temp);


            // float* z_d_temp1 = (float*)malloc(z_d[i].getRows() * sizeof(float));
            // cudaMemcpy(z_d_temp1, z_d[i].getDeviceData(), z_d[i].getRows() * sizeof(float), cudaMemcpyDeviceToHost);
            // Matrix z_d_temp_mat1 = Matrix(z_d[i].getRows(), 1, z_d_temp1);
            // printf("z_d[%i]\n", i);
            // z_d_temp_mat1.printMatrix();
            // free(z_d_temp1);


            // EXECUTE kernel, copy result to W_i_t_d

            // z_inter = W_i_t_d * a_prev
            // EXECUTE kernel, copy result to z_d[i].getDeviceData()
            dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
            dim3 gridSize(ceil(a_d[i].getCols() / 32.0), ceil(W_d[i].getCols() / 32.0));
            // printf("gridSize: (%i, %i, %i)\n", gridSize.x, gridSize.y, gridSize.z);
            matMul<<<gridSize, blockSize>>>(W_i_t_d, a_d[i].getDeviceData(), z_d[i].getDeviceData(), W_d[i].getCols(), W_d[i].getRows(), a_d[i].getRows(), a_d[i].getCols(), W_d[i].getCols(), a_d[i].getCols());
            // printf("Output shape: (%i, %i)\n", W_d[i].getCols(), a_d[i].getCols());
            CUDA_CALL(cudaDeviceSynchronize());

            // float* z_d_temp = (float*)malloc(z_d[i].getRows() * sizeof(float));
            // cudaMemcpy(z_d_temp, z_d[i].getDeviceData(), z_d[i].getRows() * sizeof(float), cudaMemcpyDeviceToHost);
            // Matrix z_d_temp_mat = Matrix(z_d[i].getRows(), 1, z_d_temp);
            // printf("mul z_d[%i]\n", i);
            // z_d_temp_mat.printMatrix();
            // free(z_d_temp);

            cudaFree(W_i_t_d);


            // reset z_inter + b[i]
            // EXECUTE kernel, copy result to z_d[i].getDeviceData()
            add<<<ceil(z_d[i].getRows() / 1024.0) , 1024>>>(z_d[i].getDeviceData(), b_d[i].getDeviceData(), z_d[i].getRows());
            CUDA_CALL(cudaDeviceSynchronize());


            // // // PRINT z_d[i]
            // float* z_d_temp = (float*)malloc(z_d[i].getRows() * sizeof(float));
            // cudaMemcpy(z_d_temp, z_d[i].getDeviceData(), z_d[i].getRows() * sizeof(float), cudaMemcpyDeviceToHost);
            // Matrix z_d_temp_mat = Matrix(z_d[i].getRows(), 1, z_d_temp);
            // printf("z_d[%i]\n", i);
            // z_d_temp_mat.printMatrix();
            // free(z_d_temp);

            // COMPUTE ACTIVATION FUNCTION

            if (i < W_d.size() - 1) {
                // LEAKY RELU
                // copy res into a_d[i].getDeviceData()
                // float* a_prev_d_temp = (float*)malloc(a_d[i].getRows() * sizeof(float));
                // cudaMemcpy(a_prev_d_temp, a_prev_d, a_d[i].getRows() * sizeof(float), cudaMemcpyDeviceToHost);
                // Matrix a_prev_d_temp_mat = Matrix(a_d[i].getRows(), 1, a_prev_d_temp);
                // printf("a[%i]\n", i);
                // a_prev_d_temp_mat.printMatrix();
                // free(a_prev_d_temp);

                float* z_h_temp = (float*)malloc(z_d[i].getRows() * sizeof(float));
                CUDA_CALL(cudaMemcpy(z_h_temp, z_d[i].getDeviceData(), z_d[i].getRows() * sizeof(float), cudaMemcpyDeviceToHost));
                Matrix z_h_temp_mat = Matrix(z_d[i].getRows(), 1, z_h_temp);

                z_h_temp_mat.applyFunction(leakyRelu2);

                // z_h_temp_mat.printMatrix();

                CUDA_CALL(cudaMemcpy(a_d[i+1].getDeviceData(), z_h_temp_mat.getVals(), a_d[i+1].getRows() * sizeof(float), cudaMemcpyHostToDevice));

                // leakyReluGPU<<<ceil(a_d[i+1].getRows() / 1024), 1024>>>(z_d[i].getDeviceData(), a_d[i+1].getDeviceData(), a_d[i+1].getRows());
                // CUDA_CALL(cudaDeviceSynchronize());

                // float* a_prev_d_temp = (float*)malloc(a_d[i+1].getRows() * sizeof(float));
                // cudaMemcpy(a_prev_d_temp, a_d[i+1].getDeviceData(), a_d[i+1].getRows() * sizeof(float), cudaMemcpyDeviceToHost);
                // Matrix a_prev_d_temp_mat = Matrix(a_d[i+1].getRows(), 1, a_prev_d_temp);
                // printf("a[%i]\n", i+1);
                // a_prev_d_temp_mat.printMatrix();
                // free(a_prev_d_temp);

            } else {
                // SOFTMAX OUTPUT

                float* z_h_temp = (float*)malloc(z_d[i].getRows() * sizeof(float));
                CUDA_CALL(cudaMemcpy(z_h_temp, z_d[i].getDeviceData(), z_d[i].getRows() * sizeof(float), cudaMemcpyDeviceToHost));
                Matrix z_h_temp_mat = Matrix(z_d[i].getRows(), 1, z_h_temp);

                // Normalize vector
                z_h_temp_mat.normalizeVec();
                // printf("z_h_temp_mat normalized: \n");
                // z_h_temp_mat.printMatrix();
                // Get sum
                float z_sum = z_h_temp_mat.expSumVec();  

                // printf("z_sum: %f\n", z_sum);

                z_h_temp_mat.applyFunction(softmax2, z_sum);
                // z_h_temp_mat.printMatrix();

                // std::cout << "Copying to a[" << i+1 << "[: " << a_d[i+1].getDeviceData() << std::endl;

                CUDA_CALL(cudaMemcpy(a_d[i+1].getDeviceData(), z_h_temp_mat.getVals(), a_d[i+1].getRows() * sizeof(float), cudaMemcpyHostToDevice));

                
                // CUDA_CALL(cudaMemcpy(a_d[i+1].getDeviceData(), z_h_temp_mat.getVals(), a_d[i+1].getRows() * sizeof(float), cudaMemcpyHostToDevice));

                // CUDA_CALL(cudaMemcpy(z_d[i].getDeviceData(), z_h_temp_mat.getVals(), z_d[i].getRows() * sizeof(float), cudaMemcpyHostToDevice));

                // // EXECUTE Softmax kernel, copy res into a_d[i].getDeviceData()
                // printf("z_d[%i] rows: %i\n", i, z_d[i].getRows());
                // printf("a_d[%i] rows: %i\n", i+1, a_d[i+1].getRows());

                // softmax_kernel<<<ceil(z_d[i].getRows() / 256.0), 256>>>(z_d[i].getDeviceData(), a_d[i+1].getDeviceData(), z_sum, a_d[i+1].getRows());
                // CUDA_CALL(cudaDeviceSynchronize());
            }

            // cudaFree(a_prev_d);
        }

        float* y_hat_val = (float*)malloc(outputSize * sizeof(float));
        cudaMemcpy(y_hat_val, a_d[a_d.size()-1].getDeviceData(), a_d[a_d.size()-1].getRows() * sizeof(float), cudaMemcpyDeviceToHost);
        Matrix y_hat = Matrix(outputSize, 1, y_hat_val);

        // printf("Prediction!\n");
        // y_hat.printMatrix();

        return y_hat;


    }

    Matrix backProp(Matrix W_layer, Matrix b_layer, Matrix z_layer, Matrix a_layer, Matrix dA, Matrix dB, Matrix dW, bool outputLayer) {
        float* dZ_d;
        cudaMalloc(&dZ_d, z_layer.getRows() * z_layer.getCols() * sizeof(float));

        if (outputLayer) {
            cudaMemcpy(&dZ_d, z_layer.getDeviceData(), z_layer.getRows() * z_layer.getCols() * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            int z_size = z_layer.getRows() * z_layer.getCols();
            elementWiseMultKernel<<< ceil(z_size / 1024.0), 1024 >>>(z_layer.getDeviceData(), dA.getDeviceData(), dZ_d, z_size);
        }

        // a[l-1].T
        float* a_layer_1t_d;
        cudaMalloc(&a_layer_1t_d, a_layer.getCols() * a_layer.getRows() * sizeof(float));
        dim3 grid(ceil(a_layer.getCols() / TRANSPOSE_BLOCK_DIM), ceil(a_layer.getRows() / TRANSPOSE_BLOCK_DIM), 1);
        dim3 threads(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM, 1);
        transpose<<<grid, threads>>>(a_layer_1t_d, a_layer.getDeviceData(), a_layer.getCols(), a_layer.getRows());

        // dW[l]
        dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridSize(ceil(a_layer.getRows() / 32.0), ceil(z_layer.getRows() / 32.0)); // CHECK
        matMul<<<gridSize, blockSize>>>(dZ_d, a_layer_1t_d, dW.getDeviceData(), z_layer.getRows(), z_layer.getCols(),  a_layer.getCols(), a_layer.getRows(), dW.getRows(), dW.getCols());

        // db[l] = sum(dZ[l])
        float* b_d;
        cudaMalloc(&b_d, sizeof(float));
        sumKernel<<<1, 1>>>(dZ_d, b_d, z_layer.getRows() * z_layer.getCols());
        assignScalarKernel<<<ceil(b_layer.getRows() / 1024.0), 1024>>>(dB.getDeviceData(), b_d[0], b_layer.getRows());

        // dA[l-1] = w[l].T * dZ[l]
        float* da_d;
        cudaMalloc(&da_d, a_layer.getRows() * sizeof(float));
        dim3 gridSize2(ceil(W_layer.getRows() / 32.0), ceil(z_layer.getRows() / 32.0));
        matMul<<<gridSize2, blockSize>>>(W_layer.getDeviceData(), dZ_d, da_d, W_layer.getRows(), W_layer.getCols(), z_layer.getRows(), z_layer.getCols(), a_layer.getRows(), a_layer.getCols());

        float* da_h = (float*)malloc(a_layer.getRows() * a_layer.getCols() * sizeof(float));
        cudaMemcpy(da_h, da_d, a_layer.getRows() * a_layer.getCols() * sizeof(float), cudaMemcpyDeviceToHost);
        Matrix dA_1 = Matrix(a_layer.getRows(), a_layer.getCols(), da_h);

        // Device memory
        cudaFree(w_temp_t_d);
        cudaFree(a_layer_1t_d);
        cudaFree(b_d);
        cudaFree(dZ_d);

        return dA_1;
    }

    void updateWeights(std::vector<Matrix> W_d, std::vector<Matrix> b_d, std::vector<Matrix> dW_d, std::vector<Matrix> db_d, float alpha) {
        for (int i = 0; i < W_d.size(); i++) {
            // dW * alpha
            scalarMultKernel<<<ceil(W_d[i].getRows() * W_d[i].getCols() / 1024.0), 1024>>>(dW_d[i].getDeviceData(), alpha, W_d[i].getRows() * W_d[i].getCols());
            // W[i] = W[i] - dW
            subtract<<< ceil(W_d[i].getRows() * W_d[i].getCols() / 1024.0), 1024 >>>(W_d[i].getDeviceData, dW_d[i].getDeviceData(),  W_d[i].getRows() * W_d[i].getCols())

            // db * alpha
            scalarMultKernel<<<ceil(b_d[i].getRows() * b_d[i].getCols() / 1024.0), 1024>>>(db_d[i].getDeviceData(), alpha, b_d[i].getRows() * b_d[i].getCols());
            // W[i] = W[i] - dW
            subtract<<< ceil(b_d[i].getRows() * b_d[i].getCols() / 1024.0), 1024 >>>(b_d[i].getDeviceData, db_d[i].getDeviceData(),  b_d[i].getRows() * b_d[i].getCols())
        }
    }


    void trainingStep(Matrix X, Matrix Y, std::vector<Matrix> W_d, std::vector<Matrix> b_d, std::vector<Matrix>z_d, std::vector<Matrix>a_d, std::vector<Matrix> dW_d, std::vector<Matrix> db_d) {
        // Forward Pass
        Matrix y_hat = forwardPass(W_d, b_d, z_d, a_d, X.getVals(), Y.getVals(), X.getRows());

        // Backprop loop
        

        // Update weights
        updateWeights(W_d, b_d, dW_d, db_d, 0.01);
    }

    // float* w_temp_t_d;
    // cudaMalloc(&w_temp_t_d, W_layer.getRows() * W_layer.getCols() * sizeof(float));
    // dim3 grid2(ceil(W_layer.getCols() / TRANSPOSE_BLOCK_DIM), ceil(W_layer.getRows() / TRANSPOSE_BLOCK_DIM), 1)
    // transpose<<<grid2, threads>>>(w_temp_t_d, W_layer.getDeviceData(), W_layer.getCols() * W_layer.getRows());

    // Matrix forwardPass(std::vector<Matrix> W_d, std::vector<Matrix> b_d, std::vector<Matrix>z_d, std::vector<Matrix>a_d, float* X, float* Y, int inputSize, int outputSize) {
    //     // printf("Starting forward pass!\n");

    //     for (int i = 0; i < W_d.size(); i++) {
    //         // float* a_prev_d;
    //         if (i == 0) {
    //             // cudaMalloc(&a_prev_d, inputSize * sizeof(float));
    //             cudaMemcpy(a_d[i].getDeviceData(), X, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    //         } 
            
    //         // Matrix z_temp = (W[i].transpose())*a_prev + b[i];
    //         // transpose matrix W[i] in float* W_i_t
    //         float* W_i_t_d;
    //         cudaMalloc(&W_i_t_d, W_d[i].getRows() * W_d[i].getCols() * sizeof(float));
    //         dim3 grid(ceil(W_d[i].getCols() / TRANSPOSE_BLOCK_DIM), ceil(W_d[i].getRows() / TRANSPOSE_BLOCK_DIM), 1);
    //         dim3 threads(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM, 1);
    //         transpose<<<grid, threads>>>(W_i_t_d, W_d[i].getDeviceData(), W_d[i].getCols(), W_d[i].getRows());
    //         CUDA_CALL(cudaDeviceSynchronize());

    //         // z_inter = W_i_t_d * a_prev
    //         // EXECUTE kernel, copy result to z_d[i].getDeviceData()
    //         dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    //         dim3 gridSize(ceil(a_d[i].getCols() / 32.0), ceil(W_d[i].getCols() / 32.0));
    //         matMul<<<gridSize, blockSize>>>(W_i_t_d, a_d[i].getDeviceData(), z_d[i].getDeviceData(), W_d[i].getCols(), W_d[i].getRows(), a_d[i].getRows(), a_d[i].getCols(), W_d[i].getCols(), a_d[i].getCols());
    //         CUDA_CALL(cudaDeviceSynchronize());


    //         cudaFree(W_i_t_d);


    //         // reset z_inter + b[i]
    //         // EXECUTE kernel, copy result to z_d[i].getDeviceData()
    //         add<<<ceil(z_d[i].getRows() / 1024.0) , 1024>>>(z_d[i].getDeviceData(), b_d[i].getDeviceData(), z_d[i].getRows());
    //         CUDA_CALL(cudaDeviceSynchronize());

    //         // Compute Activation
    //         if (i < W_d.size() - 1) {
    //             // LEAKY RELU
    //             float* z_h_temp = (float*)malloc(z_d[i].getRows() * sizeof(float));
    //             CUDA_CALL(cudaMemcpy(z_h_temp, z_d[i].getDeviceData(), z_d[i].getRows() * sizeof(float), cudaMemcpyDeviceToHost));
    //             Matrix z_h_temp_mat = Matrix(z_d[i].getRows(), 1, z_h_temp);

    //             z_h_temp_mat.applyFunction(leakyRelu2);

    //             CUDA_CALL(cudaMemcpy(a_d[i+1].getDeviceData(), z_h_temp_mat.getVals(), a_d[i+1].getRows() * sizeof(float), cudaMemcpyHostToDevice));

    //         } else {
    //             // SOFTMAX OUTPUT

    //             float* z_h_temp = (float*)malloc(z_d[i].getRows() * sizeof(float));
    //             CUDA_CALL(cudaMemcpy(z_h_temp, z_d[i].getDeviceData(), z_d[i].getRows() * sizeof(float), cudaMemcpyDeviceToHost));
    //             Matrix z_h_temp_mat = Matrix(z_d[i].getRows(), 1, z_h_temp);

    //             // Normalize vector
    //             z_h_temp_mat.normalizeVec();

    //             float z_sum = z_h_temp_mat.expSumVec();  


    //             z_h_temp_mat.applyFunction(softmax2, z_sum);

    //             CUDA_CALL(cudaMemcpy(a_d[i+1].getDeviceData(), z_h_temp_mat.getVals(), a_d[i+1].getRows() * sizeof(float), cudaMemcpyHostToDevice));

    //         }

    //     }

    //     float* y_hat_val = (float*)malloc(outputSize * sizeof(float));
    //     cudaMemcpy(y_hat_val, a_d[a_d.size()-1].getDeviceData(), a_d[a_d.size()-1].getRows() * sizeof(float), cudaMemcpyDeviceToHost);
    //     Matrix y_hat = Matrix(outputSize, 1, y_hat_val);


    //     return y_hat;


    // }

    // float* matrixTranspose(float* a, int height, int width) {
    //     float* b = (float*)malloc(height * width * sizeof(float));

    //     float* a_d;
    //     float* b_d;

    //     cudaMalloc(&a_d, a);
    //     cudaMalloc(&b_d, b);

    //     cudaMemcpy(a_d, a, height * width * sizeof(float), cudaMemcpyHostToDevice);
    //     cudaMemcpy(b_d, b, height * width * sizeof(float), cudaMemcpyHostToDevice);


    //     dim3 blockSize(32,32,1);
    //     dim3 gridSize(ceil(width / 32.0), ceil(height / 32.0), 1);

    //     matrixTranspose<<<gridSize, blockSize>>>(a_d, b_d, height, width);

    //     cudaMemcpy(b, b_d, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    //     cudaFree(b_d);
    //     cudaFree(a_d);

    //     return b;
    // }
}

