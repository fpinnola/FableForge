#include "vector"
#include <cmath>
#include <string>

#include "Matrix.h"
#include "NN.h"
#include "matrixGPU.cuh"

float leakyRelu(float a) {
    return std::max(0.01f * a, a);
};

float leakyReluDeriv(float x) {
    if (x >= 0) return 1;
    return 0.01;
};

float softmax(float z, float sum) {
    return std::exp(z) / (sum);
};

float log2loss(float p, float y) {
    if (p == 0.0) {
        p = 0.000001;
    }

    return y * std::log(p);
};

float cost(Matrix &h, Matrix &y) {
    // Compute log ofo predictions
    Matrix h_log = Matrix(h);
    h_log.applyFunction(log2loss, y);

    Matrix mulRes = Matrix::elemMult(y, h_log);


    return mulRes.sumVec() * -1.0;
}

void NeuralNetwork::addLayer(int numNodes, Activation activation) {
    int layer = W.size();
    int prevSize = inputSize;
    if (layer > 0) {
        prevSize = W[layer - 1].getCols();
    };
    W.push_back(Matrix::randN(prevSize, numNodes));
    b.push_back(Matrix(numNodes, 1)); 
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
        // a_update.setCol(0, vals);
}

void NeuralNetwork::updateLayerOutput(int layer, Matrix vals) {
        z[layer].setCol(0, vals);
        // a_update.setCol(0, vals);
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

Matrix NeuralNetwork::forwardPass(Matrix X, Matrix Y) {

    for (int i = 0; i < W.size(); i++) {
        // printf("Forward pass layer %i\n", i+1);

        Matrix a_temp = Matrix::zeros(1,1);
        if (i == 0) {
            a_temp = X;
        } else {
            a_temp = a[i-1];
        }

        // a_temp.printSize();
        // W[i].printSize();

        Matrix z_temp = (W[i].transpose())*a_temp + b[i];
        updateLayerOutput(i, z_temp);

        // z_temp.printSize();

        if (g[i] == 1) {
            // Leaky Relu
            z_temp.applyFunction(leakyRelu);
            updateActivationLayer(i, z_temp);
            // z_temp.printSize();

        } else if (g[i] == 2) {
            // Softmax
            Matrix maxVal = Matrix(z_temp.getRows(), 1);
            maxVal.setColToVal(0, z_temp.getMax());

            // Subtract max to normalize, prevent NaN during exp of sum
            z_temp = z_temp - maxVal;
            float z_sum = z_temp.expSumVec();
            // z_temp.printMatrix();
            z_temp.applyFunction(softmax, z_sum);
            // z_temp.printSize();

            updateActivationLayer(i, z_temp);
        }
    }


    Matrix y_hat = a[a.size() - 1];
    float pass_cost = cost(y_hat, Y);
    printf("Forward pass complete!\nCost: %f\n", pass_cost);

    return y_hat;
}

Matrix NeuralNetwork::forwardPassGPU(Matrix X, Matrix Y) {
    // User Kernel Functions

    for (int i = 0; i < W.size(); i++) {    
        // printf("Forward pass layer %i\n", i+1);

        Matrix a_temp = Matrix::zeros(1,1);
        if (i == 0) {
            a_temp = X;
        } else {
            a_temp = a[i-1];
        }

        // a_temp.printSize();
        // W[i].printSize();

        Matrix W_t = W[i].transpose(); // GPU transpose
        
        float* W_t2_vals = MatrixGPU::matrixMul(W_t.getVals(), a_temp.getVals(), W_t.getRows(), W_t.getCols(), a_temp.getRows(), a_temp.getCols()); // GPU Multiply

        float* z_temp_vals = MatrixGPU::matrixAdd(W_t2_vals, b[i].getVals(), b[i].getRows() * b[i].getCols()); // GPU Add

        Matrix z_temp = Matrix(b[i].getRows(), 1, z_temp_vals);

        updateLayerOutput(i, z_temp);

        if (g[i] == 1) {
            // Leaky Relu
            z_temp.applyFunction(leakyRelu); // GPU
            updateActivationLayer(i, z_temp);
            // z_temp.printSize();

        } else if (g[i] == 2) {
            // Softmax
            Matrix maxVal = Matrix(z_temp.getRows(), 1);
            maxVal.setColToVal(0, z_temp.getMax());

            // Subtract max to normalize, prevent NaN during exp of sum
            z_temp = z_temp - maxVal; // GPU subtraction
            float z_sum = z_temp.expSumVec(); // GPU summation
            // z_temp.printMatrix();
            z_temp.applyFunction(softmax, z_sum); // GPU
            // z_temp.printSize();

            updateActivationLayer(i, z_temp);
        }
    }

    Matrix y_hat = a[a.size() - 1];
    float pass_cost = cost(y_hat, Y);
    printf("Forward pass complete!\nCost: %f\n", pass_cost);

    return y_hat;


}

void NeuralNetwork::trainingStepGPU(Matrix X, Matrix Y) {

    Matrix y_hat = forwardPassGPU(X, Y);
    Matrix dA = y_hat - Y; // GPU 
}

void NeuralNetwork::trainingStep(Matrix X, Matrix Y) {
    
    Matrix y_hat = forwardPass(X, Y);
    Matrix dA = y_hat - Y;

    

}

// TODO: implement backpropagation
std::tuple<Matrix, Matrix, Matrix> NeuralNetwork::backprop(int layer, Matrix dA) {

    Matrix z_layer = z[layer];
    if (g[layer] == Activation::LeakyRelu) {
        z_layer.applyFunction(leakyReluDeriv);
    }
    // z_layer.applyFunction()
    Matrix dZ = dA * (z[layer]);

    return std::tuple<Matrix, Matrix, Matrix>(dA, z_layer, dZ);
}



NeuralNetwork::NeuralNetwork(int size)
{
    inputSize = size;
    W = std::vector<Matrix>();
    b = std::vector<Matrix>();
    g = std::vector<Activation>();
    a = std::vector<Matrix>();
    z = std::vector<Matrix>();

}

NeuralNetwork::~NeuralNetwork()
{
}
