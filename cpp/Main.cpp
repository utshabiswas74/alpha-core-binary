#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <fstream>
#include "Config.h"
#include "Utils.h"
#include "NeuralNetwork.h"

void parseLiveCSV(std::istream& inputData, std::vector<Candle>& history, std::vector<double>& closes) {
    std::string line;
    std::vector<Candle> tempHistory;
    std::vector<double> tempCloses;

    while (std::getline(inputData, line)) {
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
        c.open = open;
        c.high = high;
        c.low = low;
        c.close = close;

        tempHistory.push_back(c);
        tempCloses.push_back(c.close);
    }

    const size_t keepCount = 2048;
    if (tempHistory.size() > keepCount) {
        history.assign(tempHistory.end() - keepCount, tempHistory.end());
        closes.assign(tempCloses.end() - keepCount, tempCloses.end());
    } else {
        history = std::move(tempHistory);
        closes = std::move(tempCloses);
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::vector<std::unique_ptr<CNN>> buyModels;
    std::vector<std::unique_ptr<CNN>> sellModels;
    std::string baseName = Config::MODEL_FILE_BASE;

    for (int i = 1; i <= Config::MAX_ENSEMBLE_MODELS; ++i) {
        std::string modelPath = baseName + "_v" + std::to_string(i) + ".bin";
        std::ifstream f(modelPath, std::ios::binary);
        if (f.is_open()) {
            auto buyCnn = std::make_unique<CNN>();
            auto sellCnn = std::make_unique<CNN>();
            if (buyCnn->load(f) && sellCnn->load(f)) {
                buyModels.push_back(std::move(buyCnn));
                sellModels.push_back(std::move(sellCnn));
            }
            f.close();
        }
    }

    if (buyModels.empty() || sellModels.empty()) {
        return 1;
    }

    std::vector<Candle> history;
    std::vector<double> closes;

    parseLiveCSV(std::cin, history, closes);

    if (history.size() < 100) {
        return 0;
    }

    std::vector<double> rsi = Utils::calculateRSI(closes, 9);
    std::vector<double> ema20 = Utils::calculateEMA(closes, 20);
    std::vector<double> atr = Utils::calculateATR(history, 14);
    std::vector<double> adx = Utils::calculateADX(history, 14);
    std::vector<double> bbPct = Utils::calculateBB_PctB(closes, 20, 2.0);

    int lastIndex = history.size() - 1; 
    double totalBuyPred = 0.0;
    double totalSellPred = 0.0;
    int validModels = 0;

    for (size_t i = 0; i < buyModels.size(); ++i) {
        int steps = buyModels[i]->config.inputTimeSteps;
        int feats = buyModels[i]->config.inputFeatures;

        if (lastIndex < steps) continue;

        Tensor input = Utils::generateInputTensor(lastIndex, history, closes, ema20, rsi, atr, adx, bbPct, steps, feats);

        if (input.depth > 0) {
            totalBuyPred += buyModels[i]->predict(input);
            totalSellPred += sellModels[i]->predict(input);
            validModels++;
        }
    }

    if (validModels == 0) {
        return 0;
    }

    double avgBuy = totalBuyPred / validModels;
    double avgSell = totalSellPred / validModels;

    std::string signal = "HOLD";
    double finalConfidence = std::max(avgBuy, avgSell) * 100.0;

    if(finalConfidence > 50) {
        if ((avgBuy - avgSell) > 0.09) {
            signal = "BUY";
        } else if ((avgSell - avgBuy) > 0.01) {
            signal = "SELL";
        }
    }

    std::cout << "JSON_START{"
              << "\"signal\": \"" << signal << "\", "
              << "\"confidence\": " << std::fixed << std::setprecision(2) << finalConfidence << ", "
              << "\"prob_buy\": " << std::fixed << std::setprecision(6) << avgBuy << ", "
              << "\"prob_sell\": " << std::fixed << std::setprecision(6) << avgSell << ", "
              << "\"target\": " << Config::TARGET_CANDLES
              << "}JSON_END\n";

    return 0;
}