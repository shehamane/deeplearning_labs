#ifndef LAB1_LAYER_H
#define LAB1_LAYER_H

#include "Tensor.h"

template<typename T>
class Layer {

public:
    virtual Tensor2D<T> forward(Tensor2D<T> data) = 0;

    virtual Tensor2D<T> backward(Tensor2D<T> data, Tensor2D<T> prevGrad) = 0;

    virtual void makeStep(double step) = 0;
};

template<typename T>
class FCLayer : public Layer<T> {
private:
    unsigned int in, out;
    Tensor2D<T> weights;
    Tensor2D<T> biases;
    Tensor2D<T> weightsGrads;
    Tensor2D<T> biasesGrads;

public:

    FCLayer(unsigned int in, unsigned int out) : in(in), out(out) {
        double bound = sqrt(6) / sqrt(this->in + this->out);
        this->weights = Tensor2D<T>::random(this->in, this->out, -bound, bound);
        this->biases = Tensor2D<T>::random(1, this->out, -bound, bound);
    }

    Tensor2D<T> forward(Tensor2D<T> data) override {
        return data.matmul(this->weights).add(this->biases);
    }

    Tensor2D<T> backward(Tensor2D<T> interimData, Tensor2D<T> prevGrad) override {
        this->weightsGrads = interimData.transpose().matmul(prevGrad);
        this->biasesGrads = prevGrad.sumByDimension(0);
        return prevGrad.matmul(this->weights.transpose());
    }

    void makeStep(double step) override {
        this->weights = this->weights - this->weightsGrads.scale(step);
    }
};

template<typename T>
T tanh_templated(T x) {
    return static_cast<T>(tanh(x));
}

template<typename T>
T tanh2_templated(T x){
    return tanh_templated(x)* tanh_templated(x);
}

template<typename T>
class TanhLayer : public Layer<T> {
public:
    TanhLayer() {};

    Tensor2D<T> forward(Tensor2D<T> data) override {
        return data.map(&tanh_templated);
    }

    Tensor2D<T> backward(Tensor2D<T> interimData, Tensor2D<T> prevGrad) override {
        return Tensor2D<T>::ones(interimData.getShape().first, interimData.getShape().second) - interimData.map(&tanh2_templated);
    }

    void makeStep(double step) override{return;};
};

template<typename T>
class SoftMaxLayer : public Layer<T> {
private:
    std::vector<T> softmax(std::vector<T> logits) {
        std::vector<T> res(logits.size());
        T expSum = 0;
        for (int i = 0; i < logits.size(); ++i)
            expSum += exp(logits[i]);

        for (int i = 0; i < res.size(); ++i)
            res[i] = exp(logits[i]) / expSum;
        return res;
    }

public:
    SoftMaxLayer() {};

    Tensor2D<T> forward(Tensor2D<T> data) override {
        Tensor2D<T> res(data);
        for (int objIdx = 0; objIdx < data.getShape().first; ++objIdx) {
            res[objIdx] = this->softmax(data[objIdx]);
        }
        return res;
    }

    Tensor2D<T> backward(Tensor2D<T> interimData, Tensor2D<T> prevGrad) override {
        return prevGrad;
    }

    void makeStep(double step) override{return;};
};

template<typename T>
class CrossEntropyLoss{
public:
    CrossEntropyLoss() {}

    double loss(Tensor2D<T> predictedProbas, std::vector<unsigned int> trueLabels) {
        double loss = 0;
        for (int i = 0; i < predictedProbas.getShape().first; ++i)
            loss -= log(predictedProbas[i][trueLabels[i]]);
        loss /= predictedProbas.getShape().first;

        return loss;
    }

    Tensor2D<T> backward(Tensor2D<T> softmaxedData, std::vector<unsigned int> trueLabels) {
        Tensor2D<T> res(softmaxedData);
        for (int i = 0; i < softmaxedData.getShape().first; ++i) {
            res[i][trueLabels[i]] -= 1;
        }

        for (int i = 0; i < softmaxedData.getShape().first; ++i)
            for (int j = 0; j<softmaxedData.getShape().second; ++j)
                res[i][j] /= trueLabels.size();
        return res;
    }
};


#endif //LAB1_LAYER_H
