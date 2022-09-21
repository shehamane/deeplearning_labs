#ifndef LAB1_TENSOR_H
#define LAB1_TENSOR_H

#include <vector>
#include <stdexcept>

template<typename T>
class Tensor2D {
private:
    std::vector<std::vector<T>> data;
    std::pair<unsigned int, unsigned int> shape;

    static bool isMatrix(std::vector<std::vector<T>> data) {
        unsigned int rows = data.size();
        if (rows == 0)
            return false;

        unsigned int cols = data[0].size();
        if (cols == 0)
            return false;

        for (int i = 0; i < data.size(); ++i)
            if (data[i].size() != cols)
                return false;
        return true;
    }

public:
    explicit Tensor2D(std::vector<std::vector<T>> data) {
        if (!isMatrix(data))
            throw std::invalid_argument("received data is not a matrix");

        this->data = data;
        this->shape = std::pair<unsigned int, unsigned int>(data.size(), data[0].size());
    }

    Tensor2D(const Tensor2D<T> &other) {
        this->shape = other.shape;
        this->data = other.data;
    }

    static Tensor2D zeros(unsigned int rows, unsigned int cols) {
        std::vector<std::vector<T>> data(rows, std::vector<T>(cols, static_cast<T>(0)));
        return Tensor2D<T>(data);
    }

    static Tensor2D ones(unsigned int rows, unsigned int cols) {
        std::vector<std::vector<T>> data(rows, std::vector<T>(cols, static_cast<T>(1)));
        return Tensor2D<T>(data);
    }

    static Tensor2D empty(unsigned int rows, unsigned int cols) {
        std::vector<std::vector<T>> data(rows, std::vector<T>(cols));
        return Tensor2D<T>(data);
    }

    std::vector<T> getRow(unsigned int index) {
        return new std::vector<T>(this->data[index]);
    }

    std::vector<T> getCol(unsigned int index) {
        std::vector<T> res = new std::vector<T>(this->shape.first);
        for (int i = 0; i < this->shape.first; ++i)
            res[i] = this->data[i][index];
        return res;
    }


    Tensor2D<T> map(T (*func)(T x)) {
        Tensor2D<T> res = new Tensor2D<T>(this);
        int rows = res.shape.first;
        int cols = res.shape.second;

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res.data[i][j] = func(res.data[i][j]);

        return res;
    }

    Tensor2D<T> scale(T multiplier) {
        Tensor2D<T> res = new Tensor2D<T>(this);
        int rows = res.shape.first;
        int cols = res.shape.second;

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res.data[i][j] *= multiplier;

        return res;
    }

    Tensor2D<T> add(Tensor2D<T> other) {
        if (this->shape != other.shape)
            throw std::invalid_argument("shapes of the matrices are incompatible");

        Tensor2D<T> res = new Tensor2D(this);
        int rows = res.shape.first;
        int cols = res.shape.second;

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res.data[i][j] += other.data[i][j];

        return res;
    }

    Tensor2D<T> matmul(Tensor2D<T> other) {
        if (this->shape.second != other.shape.first)
            throw std::invalid_argument("shapes of the matrices are incompatible");

        unsigned int rows = this->shape.first;
        unsigned int cols = other.shape.second;
        Tensor2D<T> res = Tensor2D<T>::empty(rows, cols);

        for (int i = 0; i < rows; ++i) {
            std::vector<T> row = this->getRow(i);
            for (int j = 0; j < cols; ++j) {
                std::vector<T> col = other.getCol(j);
                res.data[i][j] = scalarProduct(row, col);
            }
        }
        return res;
    }
};


#endif //LAB1_TENSOR_H
