#include "lib/Tensor.h"
#include "lib/Layer.h"
#include "lib/NeuralNetwork.h"
#include "lib/DataSet.h"
#include <iostream>

double getAccuracy(std::vector<unsigned int> labels, std::vector<unsigned int> preds) {
    double hits = 0;
    for (int i = 0; i < labels.size(); ++i)
        if (labels[i] == preds[i])
            ++hits;
    return (hits / labels.size()) * 100;
}

int main() {
    DataSet dataset(100);
    auto train_test_split = dataset.trainTestSplit();
    auto train = train_test_split.first;
    auto data = Tensor2D<float>(train.first);
    auto labels = train.second;

    NeuralNetwork<float> nn(new CrossEntropyLoss<float>(), new SoftMaxLayer<float>());
    nn.addLayer(new FCLayer<float>(24, 20), true);
    nn.addLayer(new TanhLayer<float>());
    nn.addLayer(new FCLayer<float>(20, 16), true);
    nn.addLayer(new TanhLayer<float>());
    nn.addLayer(new FCLayer<float>(16, 13), true);
    nn.addLayer(new TanhLayer<float>());
    nn.addLayer(new FCLayer<float>(13, 10), true);


    for (unsigned int epoch = 0; epoch < 700; ++epoch) {
        Tensor2D<float> probas = nn.forward(data);

        std::vector<unsigned int> predictions = nn.makeChoices(probas);
        std::cout << "Epoch №" << epoch << " CrossEntropyLoss: " << nn.loss->loss(probas, labels)
                  << " Accuracy = " << getAccuracy(labels, predictions) << "\n";
        nn.backward(labels);
        nn.optimize();
    }

    std::cout << "===========================\n";

    auto test = train_test_split.second;
    auto test_data = Tensor2D<float>(test.first);
    auto test_labels = test.second;
    auto probas = nn.forward(data);
    auto predictions = nn.makeChoices(probas);
    std::cout << "Accuracy on test: " << getAccuracy(test_labels, predictions) << "\n";

    return 0;
}