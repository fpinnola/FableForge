#include <vector>
#include <cmath>
#include <string>
#include <random>

#include "Matrix.h"
#include "NN.h"
// #include "matrixGPU.cuh"

float leakyRelu(float a) {
    return std::max(0.01f * a, a);
};

float leakyReluDeriv(float x) {
    if (x >= 0) return 1.0;
    return 0.01;
};

float softmax(float z, float sum) {
    float sftmax = std::exp(z) / (sum);
    if (isnan(sftmax)){
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
    if (isnan(finalCost)) {
        printf("Cost evaluated nan\n");
    }

    return finalCost;
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

    float bVal  = ((float(std::rand()) / float(RAND_MAX)) * (3.0 - -3.0)) + -3.0;
    Matrix bias = Matrix::ones(numNodes, 1).scalarMult(bVal);
    b.push_back(bias);

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
    Matrix y_hat = Matrix::ones(4,4);

    return y_hat;

}

void NeuralNetwork::trainingStepGPU(Matrix X, Matrix Y) {

    Matrix y_hat = forwardPassGPU(X, Y);
    Matrix dA = y_hat - Y; // GPU 
}



void NeuralNetwork::trainingStep(Matrix X, Matrix Y) {

    Matrix y_hat = forwardPass(X, Y);

    Matrix dA = y_hat - Y;

    if (isnan(dA.sumVec())) {
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
    
    float lr = 0.001;

    for (int i = W.size() - 1; i >= 0; i--) {
        
        std::tuple<Matrix , Matrix, float> res = backprop(i, dA);
        dA = std::get<0>(res);
        Matrix dW = std::get<1>(res);
        float db = std::get<2>(res);
        Matrix db2 = Matrix::ones(b[i].getRows(), 1);
        db2 = db2.scalarMult(db);

        if (isnan((W[i] - (dW.scalarMult(lr))).get(0,0))) {
            printf("W[%i]\n", i);
            W[i].printMatrix();

            printf("dW\n");
            dW.printMatrix();

            printf("dW.scalarMult(0.01)\n");
            dW.scalarMult(lr).printMatrix();

            exit(0);
        }
        W[i] = W[i] - (dW.scalarMult(lr)); // Update Weights
        b[i] = b[i] - (db2.scalarMult(lr)); // Update biases
        
    }

    return;
}

Matrix NeuralNetwork::forwardPass(Matrix X, Matrix Y) {

    Matrix y_hat = Matrix::zeros(1,1);

    for (int i = 0; i < W.size(); i++) {
        // printf("Forward pass layer %i\n", i+1);

        Matrix a_temp = Matrix::zeros(1,1);
        if (i == 0) {
            a_temp = X;
            updateActivationLayer(i, X);
        } else {
            a_temp = a[i];
        }

        Matrix z_temp = (W[i].transpose())*a_temp + b[i];

        if (isnan(z_temp.sumVec())) {
                printf("W[i].T\n");
                W[i].transpose().printMatrix();

                printf("a[%i]\n", i);
                a_temp.printMatrix();

                z[i].printMatrix();
                exit(0);
        }

        updateLayerOutput(i, z_temp);

        if (g[i] == 1) {
            // Leaky Relu
            z_temp.applyFunction(leakyRelu);

            updateActivationLayer(i+1, z_temp);

        } else if (g[i] == 2) {
            // Softmax

            Matrix pre_z = Matrix(z_temp);

            z_temp.normalizeVec(); // Normalize to prevent inf exp sum
            float z_sum = z_temp.expSumVec();
            
            if (isnan(z_sum)) {
                printf("z_temp\n");
                z_temp.printMatrix();
                z[i].printMatrix();
            }
            z_temp.applyFunction(softmax, z_sum);
            if (isnan(z_temp.sumVec())) {
                printf("z_temp: \n");
                pre_z.printMatrix();

                printf("z_temp nan, z_sum: %f\n", z_sum);
                exit(0);
            }

            y_hat = z_temp;


            updateActivationLayer(i+1, z_temp);

        }
    }

    if (isnan(y_hat.sumVec())) {
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

    if (isnan(dA.get(0,0))) {
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

    if (isnan(dW_l.get(0,0))) {
        dW_l.printMatrix();
        printf("nan\n");
        exit(0);
    }


    // db[l] = sum(dZ[l])
    float db_l = dZ.sumVec();

    // dA[l-1] = w[l].T * dZ[l]
    Matrix dA_l1 = W[layer]*dZ;

    if (isnan(dA_l1.get(0,0))) {

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
