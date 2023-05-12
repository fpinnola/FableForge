#include <tuple>
#include "Matrix.h"

enum Activation {
    LeakyRelu = 1,
    Softmax = 2
};

class NeuralNetwork
{
private:
    /* data */
    std::vector<Matrix> W;
    std::vector<Matrix> b;
    std::vector<Activation> g;
    std::vector<Matrix> a;
    std::vector<Matrix> z;
    std::vector<float> costHistory;
    std::vector<Matrix> dW_h;
    std::vector<Matrix> db_h;
    int inputSize;

public:
    NeuralNetwork(int size);
    ~NeuralNetwork();


    void addLayer(int numNodes, Activation activation);
    void addLayer(int numNodes, Activation activation, int input);

    void updateBiasLayer(int layer, float val);

    void updateWeightsLayer(int layer, Matrix vals);

    void updateActivationLayer(int layer, Matrix vals);
    void updateLayerOutput(int layer, Matrix vals);

    void printNN();
    float getAvgCost();

    Matrix forwardPass(Matrix X, Matrix Y);
    Matrix predict(Matrix X);
    Matrix forwardPassGPU(Matrix X, Matrix Y);
    void trainingStep(Matrix X, Matrix Y, float lr);
    void trainingStepGPU(Matrix X, Matrix Y, float lr);
    void trainingLoopGPU(std::vector<std::vector<char>> trainingSet, int numEpochs, std::map<char, int> alphabet);


    std::tuple<Matrix, Matrix, float> backprop(int layer, Matrix dA);
    std::tuple<Matrix, Matrix, float> backpropGPU(int layer, Matrix dA);

};
