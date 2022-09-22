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

    static T scalarProduct(std::vector<T> left, std::vector<T> right) {
        if (left.size() != right.size())
            throw std::invalid_argument("");
        T sum = 0;
        for (int i = 0; i < left.size(); ++i)
            sum += left[i] * right[i];
        return sum;
    }

public:
    explicit Tensor2D(std::vector<std::vector<T>> data) {
        if (!isMatrix(data))
            throw std::invalid_argument("received data is not a matrix");

        this->data = data;
        this->shape = std::pair<unsigned int, unsigned int>(data.size(), data[0].size());
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
        return std::vector<T>(this->data[index]);
    }

    std::vector<T> getCol(unsigned int index) {
        std::vector<T> res(this->shape.first);
        for (int i = 0; i < this->shape.first; ++i)
            res[i] = this->data[i][index];
        return res;
    }


    Tensor2D<T> map(T (*func)(T x)) {
        Tensor2D<T> res(*this);
        int rows = res.shape.first;
        int cols = res.shape.second;

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res[i][j] = func(res[i][j]);

        return res;
    }

    Tensor2D<T> scale(T multiplier) {
        Tensor2D<T> res(*this);
        int rows = res.shape.first;
        int cols = res.shape.second;

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res[i][j] *= multiplier;

        return res;
    }

    Tensor2D<T> add(Tensor2D<T> other) {
        if (this->shape != other.shape)
            throw std::invalid_argument("shapes of the matrices are incompatible");

        Tensor2D<T> res(*this);
        int rows = res.shape.first;
        int cols = res.shape.second;

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res[i][j] += other[i][j];

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
                res[i][j] = scalarProduct(row, col);
            }
        }
        return res;
    }

    Tensor2D<T> transpose() {
        Tensor2D<T> res = Tensor2D<T>::empty(this->shape.second, this->shape.first);
        for (int i = 0; i < this->shape.first; ++i)
            for (int j = 0; j < this->shape.second; ++j)
                res[j][i] = this[i][j];
        return res;
    }

    std::vector<T> &operator[](int idx) {
        return this->data[idx];
    }

    std::vector<T> operator[](int idx) const {
        return this->data[idx];
    }
};

#endif //LAB1_TENSOR_H
