#ifndef LAB1_TENSOR_H
#define LAB1_TENSOR_H

#include <vector>
#include <stdexcept>
#include <random>

template<typename T = double>
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
    Tensor2D() {
        this->data = std::vector<std::vector<T>>(0);
        this->shape = std::pair<unsigned int, unsigned int>(0, 0);
    }

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

    static Tensor2D random(unsigned int rows, unsigned int cols, double min, double max) {
        std::random_device seeder;
        std::mt19937 engine(seeder());
        std::uniform_real_distribution<double> dist(min, max);

        Tensor2D<T> res = empty(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res[i][j] = dist(engine);
        return res;
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
        if (other.shape.first == 1 && other.shape.second == this->shape.second) {
            Tensor2D<T> res(*this);
            int rows = res.shape.first;
            int cols = res.shape.second;

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    res[i][j] += other[0][j];
            return res;
        } else if (this->shape != other.shape)
            throw std::invalid_argument("shapes of the matrices are incompatible");


        Tensor2D<T> res(*this);
        int rows = res.shape.first;
        int cols = res.shape.second;

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res[i][j] += other[i][j];

        return res;
    }

    Tensor2D<T> multiply(Tensor2D<T> other) {
        if (this->shape != other.shape)
            throw std::invalid_argument("shapes of the matrices are incompatible");
        Tensor2D<T> res = Tensor2D<T>::empty(this->shape.first, this->shape.second);

        int rows = res.shape.first;
        int cols = res.shape.second;

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                res[i][j] = this->data[i][j] * other[i][j];

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
                res[j][i] = this->data[i][j];
        return res;
    }

    Tensor2D<T> sumByDimension(unsigned int dim = 0) {
        if (dim == 0) {
            Tensor2D<T> res = Tensor2D<T>::zeros(1, this->shape.second);
            for (int i = 0; i < this->shape.first; ++i)
                for (int j = 0; j < this->shape.second; ++j)
                    res.data[0][j] += this->data[i][j];
            return res;
        } else if (dim == 1) {
            Tensor2D<T> res = Tensor2D<T>::zeros(this->shape.first, 1);
            for (int i = 0; i < this->shape.first; ++i)
                for (int j = 0; j < this->shape.second; ++j)
                    res.data[i][0] += this->data[i][j];
            return res;
        } else throw std::invalid_argument("invalid dimenstion");
    }

    Tensor2D<T> operator-(Tensor2D<T> other) {
        if (this->shape != other.shape)
            throw std::invalid_argument("sizes are incompatible");

        Tensor2D<T> res(*this);
        for (int i = 0; i < this->shape.first; ++i)
            for (int j = 0; j < this->shape.second; ++j)
                res[i][j] -= other[i][j];

        return res;
    }

    std::vector<T> &operator[](int idx) {
        return this->data[idx];
    }

    std::vector<T> operator[](int idx) const {
        return this->data[idx];
    }

    std::pair<unsigned int, unsigned int> getShape() {
        return this->shape;
    }
};

#endif //LAB1_TENSOR_H
