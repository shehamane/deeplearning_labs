#include "lib/Tensor.h"
#include "lib/Layer.h"
#include "lib/NeuralNetwork.h"

int main() {
    Tensor2D<float> data = Tensor2D<float>::random(5, 24, 0, 10);
    std::vector<unsigned int> labels{
            0, 1, 1, 0, 2
    };


    NeuralNetwork<float> nn(new CrossEntropyLoss<float>(), new SoftMaxLayer<float>());
    nn.addLayer(new FCLayer<float>(24, 16));
    nn.addLayer(new TanhLayer<float>());
    nn.addLayer(new FCLayer<float>(16, 10));


    return 0;
}