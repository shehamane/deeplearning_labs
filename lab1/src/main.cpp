#include "lib/Tensor.h"
#include "lib/Layer.h"
#include "lib/NeuralNetwork.h"
#include <iostream>

double getAccuracy(std::vector<unsigned int> labels, std::vector<unsigned int> preds) {
    double hits = 0;
    for (int i = 0; i < labels.size(); ++i)
        if (labels[i] == preds[i])
            ++hits;
    return (hits / labels.size()) * 100;
}

int main() {
    Tensor2D<float> data = Tensor2D<float>::random(300, 2, -3, 0);
    std::vector<unsigned int> labels(300);
    for (int i = 0; i < 300; ++i)
        if (data[i][1] < 2*data[i][0]+3)
            labels[i] = 1;
        else
            labels[i] = 0;


    NeuralNetwork<float> nn(new CrossEntropyLoss<float>(), new SoftMaxLayer<float>());
    nn.addLayer(new FCLayer<float>(2, 2), true);

    for (unsigned int epoch = 0; epoch < 500; ++epoch) {
        Tensor2D<float> probas = nn.forward(data);

        std::vector<unsigned int> predictions = nn.makeChoices(probas);
        std::cout << "Epoch №" << epoch << " CrossEntropyLoss: " << nn.loss->loss(probas, labels)
        << " Accuracy = " << getAccuracy(labels, predictions) << "\n";
        nn.backward(labels);
        nn.optimize();
    }
//    nn.addLayer(new FCLayer<float>(24, 16));
//    nn.addLayer(new TanhLayer<float>());
//    nn.addLayer(new FCLayer<float>(16, 10));


    return 0;
}