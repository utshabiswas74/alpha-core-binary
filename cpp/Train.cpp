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

void getTargetModelFiles(std::string& buyFile, std::string& sellFile) {
    std::string base = Config::MODEL_FILE_BASE;
    int i = 1;
    while (true) {
        buyFile = base + "_buy_v" + std::to_string(i) + ".bin";
        sellFile = base + "_sell_v" + std::to_string(i) + ".bin";
        if (!fileExists(buyFile) && !fileExists(sellFile)) return;
        i++;
    }
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    Config::ModelConfig config;
    std::string outputBuyFile;
    std::string outputSellFile;

    if (argc > 1) {
        outputBuyFile = Config::MODEL_FILE_BASE + "_buy_v" + std::string(argv[1]) + ".bin";
        outputSellFile = Config::MODEL_FILE_BASE + "_sell_v" + std::string(argv[1]) + ".bin";
    } else {
        getTargetModelFiles(outputBuyFile, outputSellFile);
    }

    std::cout << "Target Buy Model:  " << outputBuyFile << std::endl;
    std::cout << "Target Sell Model: " << outputSellFile << std::endl;

    std::vector<Candle> history;
    std::vector<double> closes;
    
    std::cout << "Loading training data..." << std::endl;
    readCSV(Config::TRAIN_DATA_FILE, history, closes);
    
    if (history.size() < config.inputTimeSteps + 100) return 1;

    std::vector<Candle> kalmanHistory;
    std::vector<double> kalmanCloses;
    std::cout << "Applying Kalman Filter..." << std::endl;
    Utils::applyKalmanFilter(history, kalmanHistory, kalmanCloses);

    int valSize = static_cast<int>(history.size() * 0.10);

    std::cout << "Calculating indicators..." << std::endl;
    std::vector<double> rsi = Utils::calculateRSI(kalmanCloses, 9);
    std::vector<double> ema20 = Utils::calculateEMA(kalmanCloses, 20);
    std::vector<double> atr = Utils::calculateATR(kalmanHistory, 14);
    std::vector<double> adx = Utils::calculateADX(kalmanHistory, 14);
    std::vector<double> bbPct = Utils::calculateBB_PctB(kalmanCloses, 20, 2.0);

    int trainSize = std::max(0, (int)history.size() - valSize);
    std::vector<int> trainIndices, valIndices;

    std::vector<double> allBuyLabels(history.size(), 0.0);
    std::vector<double> allSellLabels(history.size(), 0.0);

    int buyWins = 0, sellWins = 0, totalTrain = 0;

    for (int i = config.inputTimeSteps + 50; i < history.size() - Config::TARGET_CANDLES - 1; ++i) {
        double currentClose = closes[i];
        int targetIdx = i + Config::TARGET_CANDLES;
        double targetClose = closes[targetIdx];

        double bLabel = 0.0;
        double sLabel = 0.0;

        double movePct = ((targetClose - currentClose) / currentClose) * 100.0;

        if (i < trainSize) {
            bLabel = (movePct > Config::MIN_MOVEMENT_PCT) ? 1.0 : 0.0;
            sLabel = (movePct < -Config::MIN_MOVEMENT_PCT) ? 1.0 : 0.0;
            
            trainIndices.push_back(i);
            if (bLabel == 1.0) buyWins++;
            if (sLabel == 1.0) sellWins++;
            totalTrain++;
        } else {
            bLabel = (targetClose > currentClose) ? 1.0 : 0.0;
            sLabel = (targetClose < currentClose) ? 1.0 : 0.0;
            
            valIndices.push_back(i);
        }

        allBuyLabels[i] = bLabel;
        allSellLabels[i] = sLabel;
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
        trainInputs[i] = Utils::generateInputTensor(idx, kalmanHistory, kalmanCloses, ema20, rsi, atr, adx, bbPct, config.inputTimeSteps, config.inputFeatures);
        trainBuyTargets[i] = allBuyLabels[idx];
        trainSellTargets[i] = allSellLabels[idx];
    }

    std::vector<Tensor> valInputs(valIndices.size());
    std::vector<double> valBuyTargets(valIndices.size()), valSellTargets(valIndices.size());

    int valBuyCount = 0, valSellCount = 0;

    for (size_t i = 0; i < valIndices.size(); ++i) {
        int idx = valIndices[i];
        valInputs[i] = Utils::generateInputTensor(idx, kalmanHistory, kalmanCloses, ema20, rsi, atr, adx, bbPct, config.inputTimeSteps, config.inputFeatures);
        valBuyTargets[i] = allBuyLabels[idx];
        valSellTargets[i] = allSellLabels[idx];
        
        if (valBuyTargets[i] == 1.0) valBuyCount++;
        if (valSellTargets[i] == 1.0) valSellCount++;
    }

    int minBuyTrades = std::max(40, static_cast<int>(valBuyCount * 0.10));
    int minSellTrades = std::max(40, static_cast<int>(valSellCount * 0.10));

    CNN buyModel(config), sellModel(config);
    double currentLR = Config::LEARNING_RATE;
    double bestBuyAcc = 0.0;
    double bestSellAcc = 0.0;
    int timeStep = 0;

    if (argc > 1) {
        std::ifstream fBuy(outputBuyFile, std::ios::binary);
        std::ifstream fSell(outputSellFile, std::ios::binary);
        
        bool buyLoaded = fBuy.is_open() && buyModel.load(fBuy);
        bool sellLoaded = fSell.is_open() && sellModel.load(fSell);
        
        if (buyLoaded && sellLoaded) {
            std::cout << "Successfully loaded, Fine-Tuning..." << std::endl;
            currentLR = Config::LEARNING_RATE / 10.0;
            timeStep = 1000;
        } else {
            std::cout << "Model files not found, starting fresh training..." << std::endl;
        }
        
        if (fBuy.is_open()) fBuy.close();
        if (fSell.is_open()) fSell.close();
    }

    std::vector<int> shuffleIdx(trainInputs.size());
    std::iota(shuffleIdx.begin(), shuffleIdx.end(), 0);

    std::cout << "Starting INDEPENDENT DUAL-MODEL training..." << std::endl;
    
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
        
        auto endClock = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epochTime = endClock - startClock;

        std::cout << "[Epoch " << std::setw(3) << epoch << "] Time: " 
                  << std::fixed << std::setprecision(2) << epochTime.count() << "s | "
                  << "Buy_Val: " << std::fixed << std::setw(5) << std::setprecision(2) << bValAcc << "% [" 
                  << std::setw(4) << bCorrect << "/" << std::setw(4) << bGiven << "] | "
                  << "Sell_Val: " << std::fixed << std::setw(5) << std::setprecision(2) << sValAcc << "% [" 
                  << std::setw(4) << sCorrect << "/" << std::setw(4) << sGiven << "]";

        if (epoch > 4) {
            if (bValAcc > bestBuyAcc && bGiven > minBuyTrades) {
                bestBuyAcc = bValAcc;
                std::ofstream file(outputBuyFile, std::ios::binary);
                if (file.is_open()) {
                    buyModel.save(file);
                    file.close();
                    std::cout << " | [BUY SAVED]";
                }
            }
            if (sValAcc > bestSellAcc && sGiven > minSellTrades) {
                bestSellAcc = sValAcc;
                std::ofstream file(outputSellFile, std::ios::binary);
                if (file.is_open()) {
                    sellModel.save(file);
                    file.close();
                    std::cout << " | [SELL SAVED]";
                }
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Training complete." << std::endl;
    return 0;
}