#include <vector>
#include <cmath>
#include <string>
#include <random>
#include <map>
#include <iostream>

#include "Matrix.h"
#include "NN.h"
#include "matrixGPU.cuh"

float leakyRelu(float a) {
    return std::max(0.01f * a, a);
};

float leakyReluDeriv(float x) {
    if (x >= 0) return 1.0;
    return 0.01;
};

float softmax(float z, float sum) {
    float sftmax = std::exp(z) / (sum);
    if (std::isnan(sftmax)){
        // printf("nan\n");
        // printf("z: %f\n", z);
        // printf("sum: %f\n", sum);
    }
    return std::exp(z) / (sum);
};

float log2loss(float p, float y) {
    if (p == 0.0) {
        p = 0.00001;
    }

    return y * std::log10(p);
};

float cost(Matrix &h, Matrix &y) {
    // Compute log loss of predictions
    Matrix h_log = Matrix(h);
    h_log.applyFunction(log2loss, y);

    double finalCost = h_log.sumVec() * -1.0;
    if (std::isnan(finalCost)) {
        printf("Cost evaluated nan\n");
    }

    return finalCost;
}

Matrix oneHot(char a, std::map<char, int> alphabet) {
    Matrix vector = Matrix::zeros(alphabet.size(), 1);
    vector.set(alphabet[a],0,1.0);
    return vector;
}

Matrix clipGradients(Matrix gradients, float threshold) {
    float grad_norm = 0.0;

    for (int i = 0; i < gradients.getRows(); i++) {
        for (int j = 0; j < gradients.getCols(); j++) {
            grad_norm += gradients.get(i,j) * gradients.get(i,j);
        }
    }

    grad_norm = sqrt(grad_norm);
    // printf("grad_norm: %f\n", grad_norm);

    if (grad_norm > threshold) {
        float scale = threshold / grad_norm;

        for (int i = 0; i < gradients.getRows(); i++) {
            for (int j = 0; j < gradients.getCols(); j++) {
                gradients.set(i,j, gradients.get(i,j) * scale);
            }
        }
    }

    return gradients;

}

void NeuralNetwork::addLayer(int numNodes, Activation activation) {
    int layer = W.size();
    int prevSize = inputSize;
    if (layer > 0) {
        prevSize = W[layer - 1].getCols();
    };

    // He initialization
    float stddev = sqrt(2.0 / prevSize);
    std::normal_distribution<float> distribution(0.0, stddev);
    W.push_back(Matrix::He(prevSize, numNodes, distribution));

    // float bVal  = ((float(std::rand()) / float(RAND_MAX)) * (3.0 - -3.0)) + -3.0;
    // Matrix bias = Matrix::ones(numNodes, 1).scalarMult(bVal);
    b.push_back(Matrix::zeros(numNodes, 1));

    g.push_back(activation);
    a.push_back(Matrix::ones(numNodes, 1));
    z.push_back(Matrix::ones(numNodes, 1));
}

void NeuralNetwork::addLayer(int numNodes, Activation activation, int input) {
    W.push_back(Matrix::randN(input, numNodes));
    b.push_back(Matrix(numNodes, 1)); 
    g.push_back(activation);
    a.push_back(Matrix::ones(numNodes, 1));
    z.push_back(Matrix::ones(numNodes, 1));

}


void NeuralNetwork::updateBiasLayer(int layer, float val) {
        b[layer].setColToVal(0, val);
}

void NeuralNetwork::updateWeightsLayer(int layer, Matrix vals) {
        W[layer].setCol(0, vals);
}

void NeuralNetwork::updateActivationLayer(int layer, Matrix vals) {
        a[layer].setCol(0, vals);
}

void NeuralNetwork::updateLayerOutput(int layer, Matrix vals) {
        z[layer].setCol(0, vals);
}

void NeuralNetwork::printNN() {
        for (int i = 0; i < W.size(); i++) {
            std::string ac = "";
            if (g[i] == Activation::LeakyRelu) {
                ac = "LeakyRelu";
            } else if (g[i] == Activation::Softmax) {
                ac = "Softmax";
            }
            printf("Layer %i: %i nodes, %s activation\n", i+1, W[i].getCols(), ac.c_str());
        }
}

float NeuralNetwork::getAvgCost() {
    float sum = 0.0;
    for (int i = 0; i < costHistory.size(); i++) {
        sum += costHistory[i];
    }
    float avg = (sum / costHistory.size());
    costHistory.clear();
    return avg;
}

Matrix NeuralNetwork::predict(Matrix X) {

    for (int i = 0; i < W.size(); i++) {
        // printf("Forward pass layer %i\n", i+1);

        Matrix a_temp = Matrix::zeros(1,1);
        if (i == 0) {
            a_temp = X;
        } else {
            a_temp = a[i];
        }

        Matrix z_temp = (W[i].transpose())*a_temp + b[i];
        updateLayerOutput(i, z_temp);

        if (g[i] == 1) {
            // Leaky Relu
            z_temp.applyFunction(leakyRelu);

            updateActivationLayer(i+1, z_temp); // CHECK

        } else if (g[i] == 2) {
            // Softmax


            // Subtract max to normalize, prevent NaN during exp of sum
            z_temp.normalizeVec();
            float z_sum = z_temp.expSumVec();
            z_temp.applyFunction(softmax, z_sum);

            updateActivationLayer(i+1, z_temp); // CHECK
        }
    }


    Matrix y_hat = a[a.size() - 1];

    return y_hat;
}

Matrix NeuralNetwork::forwardPassGPU(Matrix X, Matrix Y) {
    // User Kernel Functions

    Matrix y_hat = Matrix::zeros(1,1);

    for (int i = 0; i < W.size(); i++) {
        // printf("Forward pass layer %i\n", i+1);

        Matrix a_prev = Matrix::zeros(1,1);
        if (i == 0) {
            a_prev = X;
            updateActivationLayer(i, X);
        } else {
            a_prev = a[i];
        }

        //GPU 
        // Matrix z_temp = (W[i].transpose())*a_prev + b[i];
        Matrix W_t = W[i].transpose(); // GPU transpose (cols, rows) z.dim, a.dim = (w.cols, 1)
        float* W_t2_vals = MatrixGPU::matrixMul(W_t.getVals(), a_prev.getVals(), W_t.getRows(), W_t.getCols(), a_prev.getRows(), a_prev.getCols()); // GPU Multiply
        float* z_temp_vals = MatrixGPU::matrixAdd(W_t2_vals, b[i].getVals(), b[i].getRows() * b[i].getCols()); // GPU Add
        Matrix z_temp = Matrix(b[i].getRows(), 1, z_temp_vals);
        // END GPU

        // if (std::isnan(z_temp.sumVec()) || std::isinf(z_temp.sumVec())) {
        //     printf("NaN or inf detected in layer output: %i\n", i);
        //     W_t.printMatrix();

        //     exit(0);
        // }   

        updateLayerOutput(i, z_temp);

        if (g[i] == 1) {
            // Leaky Relu

            z_temp.applyFunction(leakyRelu);

            // if (std::isnan(z_temp.sumVec()) || std::isinf(z_temp.sumVec())) {
            //     printf("NaN or inf detected in activation: %i\n", i);
            //     exit(0);
            // }

            updateActivationLayer(i+1, z_temp);

        } else if (g[i] == 2) {
            // Softmax

            // Matrix pre_z = Matrix(z_temp);

            z_temp.normalizeVec(); // Normalize to prevent inf exp sum
            float z_sum = z_temp.expSumVec();

            z_temp.applyFunction(softmax, z_sum);
            // if (std::isnan(z_temp.sumVec())) {
            //     printf("NaN detected in activation: %i \n", i);
            //     exit(0);
            // }

            y_hat = z_temp;


            updateActivationLayer(i+1, z_temp);

        }
    }

    // if (std::isnan(y_hat.sumVec())) {
    //     printf("y_hat\n");
    //     y_hat.printMatrix();
    //     exit(0);
    // }

    float pass_cost = cost(y_hat, Y);
    costHistory.push_back(pass_cost);

    return y_hat;

    // for (int i = 0; i < W.size(); i++) {    
    //     // printf("Forward pass layer %i\n", i+1);

    //     Matrix a_temp = Matrix::zeros(1,1);
    //     if (i == 0) {
    //         a_temp = X;
    //     } else {
    //         a_temp = a[i-1];
    //     }

    //     // a_temp.printSize();
    //     // W[i].printSize();

    //     Matrix W_t = W[i].transpose(); // GPU transpose

    //     float* W_t2_vals = MatrixGPU::matrixMul(W_t.getVals(), a_temp.getVals(), W_t.getRows(), W_t.getCols(), a_temp.getRows(), a_temp.getCols()); // GPU Multiply

    //     float* z_temp_vals = MatrixGPU::matrixAdd(W_t2_vals, b[i].getVals(), b[i].getRows() * b[i].getCols()); // GPU Add

    //     Matrix z_temp = Matrix(b[i].getRows(), 1, z_temp_vals);

    //     updateLayerOutput(i, z_temp);

    //     if (g[i] == 1) {
    //         // Leaky Relu
    //         z_temp.applyFunction(leakyRelu); // GPU
    //         updateActivationLayer(i, z_temp);
    //         // z_temp.printSize();

    //     } else if (g[i] == 2) {
    //         // Softmax
    //         Matrix maxVal = Matrix(z_temp.getRows(), 1);
    //         maxVal.setColToVal(0, z_temp.getMax());

    //         // Subtract max to normalize, prevent NaN during exp of sum
    //         float* z_temp_vals_2 = MatrixGPU::matrixSubtract(z_temp.getVals(), maxVal.getVals(), z_temp.getRows()); // GPU subtraction
    //         z_temp = Matrix(z_temp.getRows(), 1, z_temp_vals_2);

    //         float z_sum = z_temp.expSumVec(); // GPU summation

    //         // z_temp.printMatrix();
    //         z_temp.applyFunction(softmax, z_sum); // GPU
    //         // z_temp.printSize();

    //         updateActivationLayer(i, z_temp);
    //     }
    // }

    // Matrix y_hat = a[a.size() - 1];
    // float pass_cost = cost(y_hat, Y);
    // printf("Forward pass complete!\nCost: %f\n", pass_cost);

}

void NeuralNetwork::trainingStepGPU(Matrix X, Matrix Y, float lr) {

    Matrix y_hat = forwardPassGPU(X, Y);

    Matrix dA = Matrix(Y.getRows(), Y.getCols(), MatrixGPU::matrixSubtract(y_hat.getVals(), Y.getVals(), Y.getRows())); // GPU Subtraction

    // if (std::isnan(dA.sumVec())) {
    //     printf("NaN detected in dA\n");
    //     exit(0);
    // }
    
    for (int i = W.size() - 1; i >= 0; i--) {
        
        std::tuple<Matrix , Matrix, float> res = backprop(i, dA);
        dA = std::get<0>(res);
        Matrix dW = std::get<1>(res);
        float db = std::get<2>(res);
        Matrix db2 = Matrix::ones(b[i].getRows(), 1);

        // if (std::isnan((W[i] - (dW.scalarMult(lr))).get(0,0))) {
        //     printf("NaN or inf detected in W[%i] update\n", i);
        //     exit(0);
        // }
        dW = clipGradients(dW, 3.0f); // GPU implementation?
        if (dW.get(0,0) > 1000) {
            exit(0);
        }

        // W[i] = W[i] - (dW.scalarMult(lr));
        float * W_i_vals =  MatrixGPU::matrixSubtract(W[i].getVals(), (dW.scalarMult(lr)).getVals(), W[i].getCols() * W[i].getRows());
        // Matrix new_Wi = Matrix(W[i].getRows(), W[i].getCols(), W_i_vals);
        W[i] = Matrix(W[i].getRows(), W[i].getCols(), W_i_vals);
        float * b_i_vals =  MatrixGPU::matrixSubtract(b[i].getVals(), (db2.scalarMult(lr)).getVals(), b[i].getCols() * b[i].getRows());
        b[i] = Matrix(b[i].getRows(), b[i].getCols(), b_i_vals);
        // b[i] = b[i] - (db2.scalarMult(lr)); // Update biases

        // W[i] = Matrix(W[i].getRows(), W[i].getCols(), MatrixGPU::matrixSubtract(W[i].getVals(), (dW.scalarMult(lr)).getVals(), W[i].getCols() * W[i].getRows())); // GPU subtract
        // b[i] = Matrix(b[i].getRows(), b[i].getCols(), MatrixGPU::matrixSubtract(b[i].getVals(), (db2.scalarMult(lr)).getVals(), b[i].getCols() * b[i].getRows())); // GPU subtract 
        
    }

    return;
}

void NeuralNetwork::trainingStep(Matrix X, Matrix Y, float lr) {

    Matrix y_hat = forwardPass(X, Y);

    Matrix dA = y_hat - Y;

    if (std::isnan(dA.sumVec())) {
        printf("X\n");
        X.printMatrix();
        printf("Y\n");
        Y.printMatrix();
        printf("y_hat\n");
        y_hat.printMatrix();
        printf("dA\n");
        dA.printMatrix();
        printf("dA output nan\n");
        exit(0);
    }
    
    for (int i = W.size() - 1; i >= 0; i--) {
        
        std::tuple<Matrix , Matrix, float> res = backprop(i, dA);
        dA = std::get<0>(res);
        Matrix dW = std::get<1>(res);
        float db = std::get<2>(res);
        Matrix db2 = Matrix::ones(b[i].getRows(), 1);

        if (std::isnan((W[i] - (dW.scalarMult(lr))).get(0,0))) {
            printf("W[%i]\n", i);
            W[i].printMatrix();

            printf("dW\n");
            dW.printMatrix();

            printf("dW.scalarMult(0.01)\n");
            dW.scalarMult(lr).printMatrix();

            exit(0);
        }
        dW = clipGradients(dW, 3.0f);
        if (dW.get(0,0) > 1000) {
            exit(0);
        }

        W[i] = W[i] - (dW.scalarMult(lr));
        b[i] = b[i] - (db2.scalarMult(lr)); // Update biases
        
    }

    return;
}

Matrix NeuralNetwork::forwardPass(Matrix X, Matrix Y) {

    Matrix y_hat = Matrix::zeros(1,1);

    for (int i = 0; i < W.size(); i++) {
        // printf("Forward pass layer %i\n", i+1);

        Matrix a_prev = Matrix::zeros(1,1);
        if (i == 0) {
            a_prev = X;
            updateActivationLayer(i, X);
        } else {
            a_prev = a[i];
        }

        Matrix z_temp = (W[i].transpose())*a_prev + b[i];

        if (std::isnan(z_temp.sumVec()) || std::isinf(z_temp.sumVec())) {
                printf("W[i].T\n");
                W[i].transpose().printMatrix();

                printf("b[i]\n");
                b[i].printMatrix();

                printf("a[%i]\n", i);
                a_prev.printMatrix();
                printf("a size %lu\n", a.size());

                z[i].printMatrix();
                exit(0);
        }

        updateLayerOutput(i, z_temp);

        if (g[i] == 1) {
            // Leaky Relu
            Matrix pre_z = Matrix(z_temp);

            z_temp.applyFunction(leakyRelu);

            if (std::isnan(z_temp.sumVec()) || std::isinf(z_temp.sumVec())) {
                printf("z_temp: \n");
                pre_z.printMatrix();
                z_temp.printMatrix();

                printf("z_temp nan\n");
                exit(0);
            }

            updateActivationLayer(i+1, z_temp);

        } else if (g[i] == 2) {
            // Softmax

            Matrix pre_z = Matrix(z_temp);

            z_temp.normalizeVec(); // Normalize to prevent inf exp sum
            float z_sum = z_temp.expSumVec();
            
            if (std::isnan(z_sum)) {
                printf("z_temp\n");
                z_temp.printMatrix();
                z[i].printMatrix();
            }
            z_temp.applyFunction(softmax, z_sum);
            if (std::isnan(z_temp.sumVec())) {
                printf("z_temp: \n");
                pre_z.printMatrix();

                printf("z_temp nan, z_sum: %f\n", z_sum);
                exit(0);
            }

            y_hat = z_temp;


            updateActivationLayer(i+1, z_temp);

        }
    }

    if (std::isnan(y_hat.sumVec())) {
        printf("y_hat\n");
        y_hat.printMatrix();
        exit(0);
    }

    float pass_cost = cost(y_hat, Y);
    costHistory.push_back(pass_cost);

    return y_hat;
}

std::tuple<Matrix, Matrix, float> NeuralNetwork::backprop(int layer, Matrix dA) {

    Matrix z_layer = z[layer];

    if (g[layer] == Activation::LeakyRelu) {
        z_layer.applyFunction(leakyReluDeriv);
    }

    Matrix dZ = Matrix::elemMult(dA, z[layer]);

    if (std::isnan(dA.get(0,0))) {
        dA.printMatrix();
        printf("nan dA\n");
        printf("layer %i\n", layer);
        exit(0);
    }
    if (layer == z.size() - 1) {
        dZ = dA;
    }

    // a[l-1].T
    Matrix a_layer_1t = a[layer].transpose();



    // dW[l]
    Matrix dW_l = dZ*a_layer_1t;

    if (std::isnan(dW_l.get(0,0))) {
        dW_l.printMatrix();
        printf("nan\n");
        exit(0);
    }


    // db[l] = sum(dZ[l])
    float db_l = dZ.sumVec();

    // dA[l-1] = w[l].T * dZ[l]
    Matrix dA_l1 = W[layer]*dZ;

    if (std::isnan(dA_l1.get(0,0))) {

        dZ.printMatrix();
        printf("dZ layer %i nan \n", layer);
        // W[layer].printMatrix();
        // printf("W[%i] nan\n", layer);

        dW_l.printMatrix();


        exit(0);
    }


    //  Output dA[l-1], dW[l], db[l]
    return std::tuple<Matrix, Matrix, float>(dA_l1, dW_l.transpose(), db_l);
}

std::tuple<Matrix, Matrix, float> NeuralNetwork::backpropGPU(int layer, Matrix dA) {

    Matrix z_layer = z[layer];

    if (g[layer] == Activation::LeakyRelu) {
        z_layer.applyFunction(leakyReluDeriv);
    }

    Matrix dZ = Matrix::elemMult(dA, z[layer]); // GPU Element wise multiplication

    if (std::isnan(dA.get(0,0))) {
        dA.printMatrix();
        printf("nan dA\n");
        printf("layer %i\n", layer);
        exit(0);
    }
    if (layer == z.size() - 1) {
        dZ = dA;
    }

    // a[l-1].T
    Matrix a_layer_1t = a[layer].transpose(); // GPU transpose


    // dW[l]
    //  dZ*a_layer_1t
    Matrix dW_l = Matrix(dZ.getRows(), a_layer_1t.getCols(), MatrixGPU::matrixMul(dZ.getVals(), a_layer_1t.getVals(), dZ.getRows(), dZ.getCols(), a_layer_1t.getRows(), a_layer_1t.getCols()));

    if (std::isnan(dW_l.get(0,0))) {
        dW_l.printMatrix();
        printf("nan\n");
        exit(0);
    }


    // db[l] = sum(dZ[l])
    float db_l = dZ.sumVec(); // GPU summation

    // dA[l-1] = w[l].T * dZ[l]
    // Matrix dA_l1 = W[layer]*dZ;
    Matrix dA_l1 = Matrix(W[layer].getRows(), dZ.getCols(), MatrixGPU::matrixMul(W[layer].getVals(), dZ.getVals(), W[layer].getRows(), W[layer].getCols(), dZ.getRows(), dZ.getCols()));

    if (std::isnan(dA_l1.get(0,0))) {

        dZ.printMatrix();
        printf("dZ layer %i nan \n", layer);
        // W[layer].printMatrix();
        // printf("W[%i] nan\n", layer);

        dW_l.printMatrix();

        exit(0);
    }


    //  Output dA[l-1], dW[l], db[l]
    return std::tuple<Matrix, Matrix, float>(dA_l1, dW_l.transpose(), db_l);
}

void NeuralNetwork::trainingLoopGPU(std::vector<std::vector<char>> trainingSet, int numEpochs, std::map<char, int> alphabet) {
    for (int e = 0; e < numEpochs; e++) {
        time_t start = time(0);
        // Store W, b, a, z on GPU

        Matrix X_placeholder = Matrix::zeros(alphabet.size(), 1);
        float* X_d = MatrixGPU::storeOnDevice(X_placeholder.getVals(), X_placeholder.getRows());
        a[0].setDeviceData(X_d);

        for (int w = 0; w < W.size(); w++) {
            // Store Weights for layer on device
            float* w_d = MatrixGPU::storeOnDevice(W[w].getVals(), W[w].getCols() * W[w].getRows());
            W[w].setDeviceData(w_d);

            // Store bias term for layer on device
            float* b_d = MatrixGPU::storeOnDevice(b[w].getVals(), b[w].getCols() * b[w].getRows());
            b[w].setDeviceData(b_d);

            // Store placeholder for z, a
            Matrix z_d_mat = Matrix::zeros(W[w].getCols(), 1);
            float* z_d = MatrixGPU::storeOnDevice(z_d_mat.getVals(), W[w].getCols());
            z[w].setDeviceData(z_d);
            float* a_d = MatrixGPU::storeOnDevice(Matrix::zeros(W[w].getCols(), 1).getVals(), W[w].getCols());
            a[w+1].setDeviceData(a_d);
        }


        // How do we handle a, a??

        float costSum = 0.0;

        for (int i = 0; i < trainingSet.size(); i++) {
            // printf("Training item %i\n", i);
            Matrix X = oneHot(trainingSet[i][0], alphabet);
            Matrix Y = oneHot(trainingSet[i][1], alphabet);

            Matrix y_hat = MatrixGPU::forwardPass(W, b, z, a, X.getVals(), Y.getVals(), X.getRows(), Y.getRows());

            costSum = cost(y_hat, Y) + costSum;
            // y_hat.printMatrix();
            // Get cost
            Matrix dA = y_hat - Y;



            // trainingStepGPU(X, expected, 0.001);
        }   
        // Clear W, b, a, z on GPU
        // printf("Epoch %i, avg cost: %f | time elapsed: %fs\n", e, cost, seconds_since_start);

        // printf("Cost avg: %f\n", costSum / trainingSet.size());

        float * a_h_final = MatrixGPU::removeFromDevice(a[a.size() - 1].getDeviceData(), W[W.size() - 1].getCols());

        for (int w = 0; w < W.size(); w++) {
            // Store Weights for layer on device

            int outputSize = W[w].getCols();
            float* w_h = MatrixGPU::removeFromDevice(W[w].getDeviceData(), W[w].getCols() * W[w].getRows());
            W[w] = Matrix(W[w].getRows(), W[w].getCols(), w_h);

            // Store bias term for layer on device
            float* b_h = MatrixGPU::removeFromDevice(b[w].getDeviceData(), b[w].getCols() * b[w].getRows());
            b[w] = Matrix(b[w].getRows(), b[w].getCols(), b_h);


            // Clear z from device
            float* z_h = MatrixGPU::removeFromDevice(z[w].getDeviceData(), outputSize);
            float* a_h = MatrixGPU::removeFromDevice(a[w].getDeviceData(), outputSize);
        }

        double seconds_since_start = difftime( time(0), start);
        printf("Epoch %i, avg cost: %f | time elapsed: %fs\n", e, costSum / trainingSet.size(), seconds_since_start);
        // if (std::isnan(cost)){
        //     exit(1);
        // }
        // if (cost < 0.005){
        //     break;
        // }
    }
    return;
}



NeuralNetwork::NeuralNetwork(int size)
{
    inputSize = size;
    W = std::vector<Matrix>();
    b = std::vector<Matrix>();
    g = std::vector<Activation>();
    a = std::vector<Matrix>();
    a.push_back(Matrix::zeros(size, 1));

    z = std::vector<Matrix>();
    costHistory = std::vector<float>();
}

NeuralNetwork::~NeuralNetwork()
{
}
