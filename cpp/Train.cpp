#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <chrono>
#include <cstdio>
#include "Config.h"
#include "Utils.h"
#include "NeuralNetwork.h"

void readCSV(const std::string& filename, std::vector<Candle>& history, std::vector<double>& closes) {
    std::ifstream file(filename);
    if (!file.is_open()) return;
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), ';', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::stringstream ss(line);
        std::string date, time;
        double open, high, low, close;
        ss >> date >> time >> open >> high >> low >> close;
        if (ss.fail()) continue;
        Candle c;
        Utils::parseTime(time, c.hour, c.minute);
        c.open = open; c.high = high; c.low = low; c.close = close;
        history.push_back(c);
        closes.push_back(c.close);
    }
}

bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

std::string getTargetModelFile() {
    std::string base = Config::MODEL_FILE_BASE;
    int i = 1;
    while (true) {
        std::string name = base + "_v" + std::to_string(i) + ".bin";
        if (!fileExists(name)) return name;
        i++;
    }
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    Config::ModelConfig config;
    std::string outputModelFile;

    if (argc > 1) {
        outputModelFile = Config::MODEL_FILE_BASE + "_v" + std::string(argv[1]) + ".bin";
    } else {
        outputModelFile = getTargetModelFile();
    }

    std::cout << "Target Model File: " << outputModelFile << std::endl;

    std::vector<Candle> history;
    std::vector<double> closes;
    
    std::cout << "Loading training data..." << std::endl;
    readCSV(Config::TRAIN_DATA_FILE, history, closes);
    
    if (history.size() < config.inputTimeSteps + 100) return 1;

    int valSize = static_cast<int>(history.size() * 0.10);

    std::cout << "Calculating indicators..." << std::endl;
    std::vector<double> rsi = Utils::calculateRSI(closes, 9);
    std::vector<double> ema20 = Utils::calculateEMA(closes, 20);
    std::vector<double> atr = Utils::calculateATR(history, 14);
    std::vector<double> adx = Utils::calculateADX(history, 14);
    std::vector<double> bbPct = Utils::calculateBB_PctB(closes, 20, 2.0);

    int trainSize = std::max(0, (int)history.size() - valSize);
    std::vector<int> trainIndices, valIndices;

    std::vector<double> allBuyLabels(history.size(), 0.0);
    std::vector<double> allSellLabels(history.size(), 0.0);

    int buyWins = 0, sellWins = 0, totalTrain = 0;

    for (int i = config.inputTimeSteps + 50; i < history.size() - Config::TARGET_CANDLES - 1; ++i) {
        double currentClose = closes[i];
        int targetIdx = i + Config::TARGET_CANDLES;
        double targetClose = closes[targetIdx];

        double movePct = ((targetClose - currentClose) / currentClose) * 100.0;

        double bLabel = (movePct > Config::MIN_MOVEMENT_PCT) ? 1.0 : 0.0;
        double sLabel = (movePct < -Config::MIN_MOVEMENT_PCT) ? 1.0 : 0.0;

        allBuyLabels[i] = bLabel;
        allSellLabels[i] = sLabel;

        if (i < trainSize) {
            trainIndices.push_back(i);
            if (bLabel == 1.0) buyWins++;
            if (sLabel == 1.0) sellWins++;
            totalTrain++;
        } else {
            valIndices.push_back(i);
        }
    }

    if (totalTrain == 0) return 1;

    double bW1 = (double)totalTrain / (2.0 * std::max(1, buyWins));
    double bW0 = (double)totalTrain / (2.0 * std::max(1, totalTrain - buyWins));
    double sW1 = (double)totalTrain / (2.0 * std::max(1, sellWins));
    double sW0 = (double)totalTrain / (2.0 * std::max(1, totalTrain - sellWins));

    std::cout << "Total Train Data: " << totalTrain << std::endl;
    std::cout << "Buy Weights  [0]: " << std::fixed << std::setprecision(4) << bW0 << "  [1]: " << bW1 << std::endl;
    std::cout << "Sell Weights [0]: " << std::fixed << std::setprecision(4) << sW0 << "  [1]: " << sW1 << std::endl;

    std::random_device rd;
    std::mt19937 g(rd());

    std::cout << "Generating tensors..." << std::endl;
    std::vector<Tensor> trainInputs(trainIndices.size());
    std::vector<double> trainBuyTargets(trainIndices.size()), trainSellTargets(trainIndices.size());

    for (size_t i = 0; i < trainIndices.size(); ++i) {
        int idx = trainIndices[i];
        trainInputs[i] = Utils::generateInputTensor(idx, history, closes, ema20, rsi, atr, adx, bbPct, config.inputTimeSteps, config.inputFeatures);
        trainBuyTargets[i] = allBuyLabels[idx];
        trainSellTargets[i] = allSellLabels[idx];
    }

    std::vector<Tensor> valInputs(valIndices.size());
    std::vector<double> valBuyTargets(valIndices.size()), valSellTargets(valIndices.size());

    for (size_t i = 0; i < valIndices.size(); ++i) {
        int idx = valIndices[i];
        valInputs[i] = Utils::generateInputTensor(idx, history, closes, ema20, rsi, atr, adx, bbPct, config.inputTimeSteps, config.inputFeatures);
        valBuyTargets[i] = allBuyLabels[idx];
        valSellTargets[i] = allSellLabels[idx];
    }

    CNN buyModel(config), sellModel(config);
    double currentLR = Config::LEARNING_RATE;
    double bestAcc = 0.0;
    int timeStep = 0;

    if (argc > 1) {
        std::ifstream f(outputModelFile, std::ios::binary);
        if (f.is_open() && buyModel.load(f) && sellModel.load(f)) {
            std::cout << "Successfully loaded, Fine-Tuning..." << std::endl;
            currentLR = Config::LEARNING_RATE / 10.0;
            timeStep = 1000;
        } else {
            std::cout << "ERROR: Failed to load model file!" << std::endl;
            return 1;
        }
        f.close();
    }

    std::vector<int> shuffleIdx(trainInputs.size());
    std::iota(shuffleIdx.begin(), shuffleIdx.end(), 0);

    std::cout << "Starting DUAL-MODEL training for Binary Options..." << std::endl;
    
    for (int epoch = 1; epoch <= Config::EPOCHS; ++epoch) {
        auto startClock = std::chrono::high_resolution_clock::now();
        if (epoch > 1 && epoch % 10 == 0) currentLR *= 0.9;
        std::shuffle(shuffleIdx.begin(), shuffleIdx.end(), g);
        
        int batchCount = 0;
        buyModel.clearGradients();
        sellModel.clearGradients();

        for (size_t i = 0; i < shuffleIdx.size(); ++i) {
            int idx = shuffleIdx[i];
            Tensor input = trainInputs[idx];
            
            if (input.depth == 0) continue;
            if (Config::INPUT_NOISE > 0.0) {
                std::normal_distribution<> d(0, Config::INPUT_NOISE);
                for (double& val : input.data) val += d(g);
            }

            double bWeight = (trainBuyTargets[idx] == 1.0) ? bW1 : bW0;
            double sWeight = (trainSellTargets[idx] == 1.0) ? sW1 : sW0;

            double bPred = buyModel.train(input, trainBuyTargets[idx], bWeight);
            double sPred = sellModel.train(input, trainSellTargets[idx], sWeight);
            
            if (std::isnan(bPred) || std::isnan(sPred)) {
                buyModel.clearGradients();
                sellModel.clearGradients();
                batchCount = 0;
                continue;
            }

            batchCount++;
            if (batchCount == Config::BATCH_SIZE || i == shuffleIdx.size() - 1) {
                buyModel.averageGradients(batchCount);
                sellModel.averageGradients(batchCount);
                timeStep++;
                buyModel.updateParams(currentLR, timeStep);
                sellModel.updateParams(currentLR, timeStep);
                buyModel.clearGradients();
                sellModel.clearGradients();
                batchCount = 0;
            }
        }

        int bGiven = 0, bCorrect = 0;
        int sGiven = 0, sCorrect = 0;
        
        for (size_t j = 0; j < valInputs.size(); ++j) {
            if (valInputs[j].depth == 0) continue;
            double bPred = buyModel.predict(valInputs[j]);
            double sPred = sellModel.predict(valInputs[j]);

            if (bPred > 0.50) {
                bGiven++;
                if (valBuyTargets[j] == 1.0) bCorrect++;
            }
            if (sPred > 0.50) {
                sGiven++;
                if (valSellTargets[j] == 1.0) sCorrect++;
            }
        }

        double bValAcc = (bGiven > 0) ? (double)bCorrect / bGiven * 100.0 : 0.0;
        double sValAcc = (sGiven > 0) ? (double)sCorrect / sGiven * 100.0 : 0.0;
        double avgValAcc = (bValAcc + sValAcc) / 2.0;
        
        auto endClock = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epochTime = endClock - startClock;

        std::cout << "[Epoch " << std::setw(3) << epoch << "] Time: " 
                  << std::fixed << std::setprecision(2) << epochTime.count() << "s | "
                  << "Buy_Val: " << std::setprecision(2) << bValAcc << "% | "
                  << "Sell_Val: " << std::setprecision(2) << sValAcc << "%";

        if (avgValAcc > bestAcc && epoch > 4) {
            bestAcc = avgValAcc;
            std::ofstream file(outputModelFile, std::ios::binary);
            if (file.is_open()) {
                buyModel.save(file);
                sellModel.save(file);
                file.close();
                std::cout << " [SAVED]";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Training complete. Saved to: " << outputModelFile << std::endl;
    return 0;
}