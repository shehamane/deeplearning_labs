#ifndef LAB1_NEURALNETWORK_H
#define LAB1_NEURALNETWORK_H

#include "Layer.h"
#include <list>
#include <map>
#include <memory>

template<typename T>
class NeuralNetwork {
private:
    std::list<Layer<T> *> layers;
    std::list<Layer<T> *> layersOptim;
    std::map<Layer<T> *, Tensor2D<T>> interimResults;
    Tensor2D<T> lastRes;

public:
    SoftMaxLayer<T> *softMax;
    CrossEntropyLoss<T> *loss;

    NeuralNetwork(CrossEntropyLoss<T> *loss, SoftMaxLayer<T> *softMaxLayer) {
        this->loss = loss;
        this->softMax = softMaxLayer;
        this->layers = std::list<Layer<T> *>(0);
    }

    void addLayer(Layer<T> *layer, bool isOptim = false) {
        this->layers.push_back(layer);
        if (isOptim)
            this->layersOptim.push_back(layer);
    }

    Tensor2D<T> forward(Tensor2D<T> data) {
        Tensor2D<T> res(data);
        for (Layer<T> *layer: layers) {
            interimResults.insert({layer, res});
            res = layer->forward(res);
        }
        interimResults.insert({this->softMax, res});
        this->lastRes = softMax->forward(res);
        return this->lastRes;
    }

    Tensor2D<T> backward(std::vector<unsigned int> labels) {
        Tensor2D<T> grads = this->loss->backward(this->lastRes, labels);
        std::list<Layer<T> *> layersReversed(this->layers);
        layersReversed.reverse();
        for (Layer<T> *layer: layersReversed) {
            grads = layer->backward(this->interimResults[layer], grads);
        }
        return grads;
    }

    void optimize() {
        for (auto layer: this->layersOptim)
            layer->makeStep(0.005);
    }

    unsigned int makeChoice(std::vector<T> probas) {
        T max = 0;
        unsigned int argmax;

        for (int i = 0; i<probas.size(); ++i){
            if (probas[i]>max) {
                max = probas[i];
                argmax = i;
            }
        }
        return argmax;
    }

    std::vector<unsigned int> makeChoices(Tensor2D<T> probas){
        std::vector<unsigned int> predictions(probas.getShape().first);
        for (int i = 0; i < predictions.size(); ++i)
            predictions[i] = makeChoice(probas[i]);
        return predictions;
    }
};


#endif