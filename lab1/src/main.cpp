#include "lib/Tensor.h"

int main() {
    Tensor2D<int> tensor1 = Tensor2D<int>::zeros(2, 3);
    Tensor2D<int> tensor2 = tensor1.add(Tensor2D<int>::ones(2, 3)).scale(4);

    std::vector<std::vector<int>> uVec {
            {1, 2, 3},
            {4, 5, 6}
    };
    Tensor2D<int> u = Tensor2D<int>(uVec);

    std::vector<std::vector<int>> vVec {
            {0, 9, 8, 7},
            {7, 6, 5, 4},
            {5, 4, 3, 2},
            {3, 2, 1, 0}
    };
    Tensor2D<int> v = Tensor2D<int>(vVec);

    Tensor2D<int> mul = u.matmul(v);

    return 0;
}