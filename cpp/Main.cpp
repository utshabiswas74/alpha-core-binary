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

    const size_t keepCount = 4096;
    if (tempHistory.size() > keepCount) {
        history.assign(tempHistory.end() - keepCount, tempHistory.end());
        closes.assign(tempCloses.end() - keepCount, tempCloses.end());
    } else {
        history = std::move(tempHistory);
        closes = std::move(tempCloses);
    }
}

void aggregateCandles(const std::vector<Candle>& history, int multiplier, std::vector<Candle>& aggHistory, std::vector<double>& aggCloses) {
    int n = history.size();
    for (int i = 0; i < n; i += multiplier) {
        Candle agg = history[i];
        int end = std::min(i + multiplier, n);
        for (int j = i + 1; j < end; ++j) {
            agg.high = std::max(agg.high, history[j].high);
            agg.low = std::min(agg.low, history[j].low);
            agg.close = history[j].close;
            agg.hour = history[j].hour;
            agg.minute = history[j].minute;
        }
        aggHistory.push_back(agg);
        aggCloses.push_back(agg.close);
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::vector<std::unique_ptr<CNN>> buyModels;
    std::vector<std::unique_ptr<CNN>> sellModels;
    std::vector<std::unique_ptr<CNN>> extraBuyModels;
    std::vector<std::unique_ptr<CNN>> extraSellModels;

    for (int i = 1; i <= Config::MAX_ENSEMBLE_MODELS; ++i) {
        std::string modelPath = Config::MODEL_FILE_BASE + "_v" + std::to_string(i) + ".bin";
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

        std::string extraPath = Config::MODEL_FILE_EXTRA + "_v" + std::to_string(i) + ".bin";
        std::ifstream fe(extraPath, std::ios::binary);
        if (fe.is_open()) {
            auto buyCnnExtra = std::make_unique<CNN>();
            auto sellCnnExtra = std::make_unique<CNN>();
            if (buyCnnExtra->load(fe) && sellCnnExtra->load(fe)) {
                extraBuyModels.push_back(std::move(buyCnnExtra));
                extraSellModels.push_back(std::move(sellCnnExtra));
            }
            fe.close();
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

    std::string regularSignal = "HOLD";
    double finalConfidence = std::max(avgBuy, avgSell) * 100.0;

    if (finalConfidence > 50) {
        if ((avgBuy - avgSell) > 0.01) {
            regularSignal = "BUY";
        } else if ((avgSell - avgBuy) > 0.01) {
            regularSignal = "SELL";
        }
    }

    std::string extraSignal = regularSignal;

    if (!extraBuyModels.empty() && Config::TIMEFRAME_MULTIPLIER > 1) {
        extraSignal = "HOLD";
        std::vector<Candle> extraHistory;
        std::vector<double> extraCloses;
        
        aggregateCandles(history, Config::TIMEFRAME_MULTIPLIER, extraHistory, extraCloses);

        if (extraHistory.size() >= 50) {
            std::vector<double> extraRsi = Utils::calculateRSI(extraCloses, 9);
            std::vector<double> extraEma20 = Utils::calculateEMA(extraCloses, 20);
            std::vector<double> extraAtr = Utils::calculateATR(extraHistory, 14);
            std::vector<double> extraAdx = Utils::calculateADX(extraHistory, 14);
            std::vector<double> extraBbPct = Utils::calculateBB_PctB(extraCloses, 20, 2.0);

            int extraLastIndex = extraHistory.size() - 1;
            double totalExtraBuy = 0.0;
            double totalExtraSell = 0.0;
            int validExtra = 0;

            for (size_t i = 0; i < extraBuyModels.size(); ++i) {
                int steps = extraBuyModels[i]->config.inputTimeSteps;
                int feats = extraBuyModels[i]->config.inputFeatures;

                if (extraLastIndex < steps) continue;

                Tensor input = Utils::generateInputTensor(extraLastIndex, extraHistory, extraCloses, extraEma20, extraRsi, extraAtr, extraAdx, extraBbPct, steps, feats);

                if (input.depth > 0) {
                    totalExtraBuy += extraBuyModels[i]->predict(input);
                    totalExtraSell += extraSellModels[i]->predict(input);
                    validExtra++;
                }
            }

            if (validExtra > 0) {
                double exAvgBuy = totalExtraBuy / validExtra;
                double exAvgSell = totalExtraSell / validExtra;
                double exConf = std::max(exAvgBuy, exAvgSell) * 100.0;

                if (exConf > 50) {
                    if ((exAvgBuy - exAvgSell) > 0.01) {
                        extraSignal = "BUY";
                    } else if ((exAvgSell - exAvgBuy) > 0.01) {
                        extraSignal = "SELL";
                    }
                }
            }
        }
    }

    std::string finalSignal = "HOLD";
    if (regularSignal == extraSignal && regularSignal != "HOLD") {
        finalSignal = regularSignal;
    }

    std::cout << "JSON_START{"
              << "\"signal\": \"" << finalSignal << "\", "
              << "\"confidence\": " << std::fixed << std::setprecision(2) << finalConfidence << ", "
              << "\"prob_buy\": " << std::fixed << std::setprecision(6) << avgBuy << ", "
              << "\"prob_sell\": " << std::fixed << std::setprecision(6) << avgSell << ", "
              << "\"target\": " << Config::TARGET_CANDLES
              << "}JSON_END\n";

    return 0;
}