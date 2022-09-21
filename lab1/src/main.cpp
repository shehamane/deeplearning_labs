#include "lib/Tensor.h"

int main() {
    Tensor2D<int> tensor1 = Tensor2D<int>::zeros(2, 3);
    Tensor2D<int> tensor2 = tensor1.add(Tensor2D<int>::ones(2, 3)).scale(4);
    int a = 1;

    return 0;
}