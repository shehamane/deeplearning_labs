#ifndef LAB1_DATALOADER_H
#define LAB1_DATALOADER_H

#include <vector>
#include <random>
#include <algorithm>
#include "Tensor.h"

class DataSet {
    const std::vector<std::vector<std::vector<float>>> origins{
            {
                    {1, 1, 1, 1,},//@@@@
                    {1, 0, 0, 1,},//@--@
                    {1, 0, 0, 1,},//@--@
                    {1, 0, 0, 1,},//@--@
                    {1, 0, 0, 1,},//@--@
                    {1, 1, 1, 1,},//@@@@
            },
            {
                    {0, 0, 1, 0,},//--@-
                    {0, 1, 1, 0,},//-@@-
                    {1, 0, 1, 0,},//@-@-
                    {0, 0, 1, 0,},//--@-
                    {0, 0, 1, 0,},//--@-
                    {0, 1, 1, 1,},//-@@@
            },
            {
                    {0, 1, 1, 0,},//-@@-
                    {1, 0, 0, 1,},//@--@
                    {0, 0, 1, 0,},//--@-
                    {0, 1, 0, 0,},//-@--
                    {1, 0, 0, 0,},//@---
                    {1, 1, 1, 1,},//@@@@
            },
            {
                    {0, 1, 1, 0,},//-@@-
                    {1, 0, 0, 1,},//@--@
                    {0, 0, 1, 0,},//--@-
                    {0, 0, 0, 1,},//---@
                    {1, 0, 0, 1,},//@--@
                    {0, 1, 1, 0,},//-@@-
            },
            {
                    {1, 0, 0, 1,},//@--@
                    {1, 0, 0, 1,},//@--@
                    {1, 0, 0, 1,},//@@@@
                    {1, 1, 1, 1,},//---@
                    {0, 0, 0, 1,},//---@
                    {0, 0, 0, 1,},//---@
            },
            {
                    {1, 1, 1, 1,},//@@@@
                    {1, 0, 0, 0,},//@---
                    {1, 1, 1, 1,},//@@@@
                    {0, 0, 0, 1,},//---@
                    {0, 0, 0, 1,},//---@
                    {1, 1, 1, 1,},//@@@@
            },
            {
                    {1, 1, 1, 1,},//@@@@
                    {1, 0, 0, 0,},//@---
                    {1, 1, 1, 1,},//@@@@
                    {1, 0, 0, 1,},//@--@
                    {1, 0, 0, 1,},//@--@
                    {1, 1, 1, 1,},//@@@@
            },
            {
                    {1, 1, 1, 1,},//@@@@
                    {0, 0, 0, 1,},//---@
                    {0, 0, 1, 0,},//--@-
                    {0, 0, 1, 0,},//--@-
                    {0, 1, 0, 0,},//-@--
                    {0, 1, 0, 0,},//-@--
            },

            {
                    {1, 1, 1, 1,},//@@@@
                    {1, 0, 0, 1,},//@--@
                    {1, 1, 1, 1,},//@@@@
                    {1, 0, 0, 1,},//@--@
                    {1, 0, 0, 1,},//@--@
                    {1, 1, 1, 1,},//@@@@
            },
            {
                    {1, 1, 1, 1,},//@@@@
                    {1, 0, 0, 1,},//@--@
                    {1, 0, 0, 1,},//@--@
                    {1, 1, 1, 1,},//@@@@
                    {0, 0, 0, 1,},//---@
                    {1, 1, 1, 1,},//@@@@
            },
    };
    const unsigned int NUM_CLASSES = 10;

    std::vector<float> flatten(std::vector<std::vector<float>> matrix) {
        std::vector<float> res(matrix.size() * matrix[0].size());
        for (int i = 0; i < matrix.size(); ++i)
            for (int j = 0; j < matrix[0].size(); ++j)
                res[i * matrix[0].size() + j] = matrix[i][j];
        return res;
    }

    std::vector<float> addGaussianNoise(std::vector<float> vec) {
        std::vector<float> res(vec.size());

        const double mean = 0.0;
        const double stddev = 0.1;
        std::default_random_engine generator;
        std::normal_distribution<double> dist(mean, stddev);
        for (int i = 0; i < res.size(); ++i)
            res[i] = vec[i] + dist(generator);
        return res;
    }

public:
    std::vector<std::vector<float>> X;
    std::vector<unsigned int> y;
    std::vector<unsigned int> indexes;

    DataSet(unsigned int n = 30) {
        X = std::vector<std::vector<float>>(NUM_CLASSES * n);
        y = std::vector<unsigned int>(NUM_CLASSES * n);
        indexes = std::vector<unsigned int>(NUM_CLASSES * n);

        unsigned int index = 0;
        for (int cls = 0; cls < 10; ++cls) {
            std::vector<float> originFlatten = flatten(origins[cls]);
            for (int i = 0; i < n; ++i) {
                X[index] = addGaussianNoise(originFlatten);
                y[index] = cls;
                indexes[index] = index;
                ++index;
            }
        }
    }

    std::pair<std::vector<std::vector<float>>, std::vector<unsigned int>> getShuffled() {
        std::random_shuffle(indexes.begin(), indexes.end());
        std::vector<std::vector<float>> X_shuffled(X.size());
        std::vector<unsigned int> y_shuffled(y.size());

        for (int i = 0; i < indexes.size(); ++i) {
            X_shuffled[i] = (X[indexes[i]]);
            y_shuffled[i] = (y[indexes[i]]);
        }
        return std::pair<std::vector<std::vector<float>>, std::vector<unsigned int>>(X_shuffled, y_shuffled);
    }

    std::pair<std::pair<std::vector<std::vector<float>>, std::vector<unsigned int>>, std::pair<std::vector<std::vector<float>>, std::vector<unsigned int>>>
    trainTestSplit() {
        auto dataset = this->getShuffled();
        auto X = dataset.first;
        auto y = dataset.second;

        std::size_t const train_size = (int) X.size() * 0.75;
        std::vector<std::vector<float>> X_train(X.begin(), X.begin() + train_size);
        std::vector<unsigned int> y_train(y.begin(), y.begin() + train_size);
        std::vector<std::vector<float>> X_test(X.begin() + train_size, X.end());
        std::vector<unsigned int> y_test(y.begin() + train_size, y.end());

        std::pair<std::vector<std::vector<float>>, std::vector<unsigned int>> train(X_train, y_train);
        std::pair<std::vector<std::vector<float>>, std::vector<unsigned int>> test(X_test, y_test);
        return std::pair<std::pair<std::vector<std::vector<float>>, std::vector<unsigned int>>, std::pair<std::vector<std::vector<float>>, std::vector<unsigned int>>>(train, test);
    }

};


#endif //LAB1_DATALOADER_H
