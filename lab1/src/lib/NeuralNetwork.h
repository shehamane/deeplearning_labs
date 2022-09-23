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

    void addLayer(Layer<T> *layer) {
        this->layers.push_back(layer);
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
        std::list<Layer<T>*> layersReversed(this->layers);
        layersReversed.reverse();
        for (Layer<T>* layer : layersReversed){
            grads = layer->backward(this->interimResults[layer], grads);
        }
    }
};


#endif