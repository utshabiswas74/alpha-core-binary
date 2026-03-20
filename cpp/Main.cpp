#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <fstream>
#include <map>
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

struct ModelGroup {
    int targetCandles;
    std::vector<std::unique_ptr<CNN>> buyModels;
    std::vector<std::unique_ptr<CNN>> sellModels;
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::map<int, ModelGroup> modelGroups;

    for (int t = 1; t <= 30; ++t) {
        for (int i = 1; i <= Config::MAX_ENSEMBLE_MODELS; ++i) {
            std::string buyPath = Config::MODEL_FILE_BASE + "_t" + std::to_string(t) + "_buy_v" + std::to_string(i) + ".bin";
            std::string sellPath = Config::MODEL_FILE_BASE + "_t" + std::to_string(t) + "_sell_v" + std::to_string(i) + ".bin";
            std::ifstream fb(buyPath, std::ios::binary);
            std::ifstream fs(sellPath, std::ios::binary);
            
            if (fb.is_open() && fs.is_open()) {
                auto buyCnn = std::make_unique<CNN>();
                auto sellCnn = std::make_unique<CNN>();
                if (buyCnn->load(fb) && sellCnn->load(fs)) {
                    modelGroups[t].targetCandles = t;
                    modelGroups[t].buyModels.push_back(std::move(buyCnn));
                    modelGroups[t].sellModels.push_back(std::move(sellCnn));
                }
            }
            if (fb.is_open()) fb.close();
            if (fs.is_open()) fs.close();
        }
    }

    std::vector<std::unique_ptr<CNN>> extraBuyModels;
    std::vector<std::unique_ptr<CNN>> extraSellModels;

    for (int i = 1; i <= Config::MAX_ENSEMBLE_MODELS; ++i) {
        std::string exBuyPath = Config::MODEL_FILE_EXTRA + "_t1_buy_v" + std::to_string(i) + ".bin";
        std::string exSellPath = Config::MODEL_FILE_EXTRA + "_t1_sell_v" + std::to_string(i) + ".bin";
        std::ifstream feb(exBuyPath, std::ios::binary);
        std::ifstream fes(exSellPath, std::ios::binary);
        
        if (feb.is_open() && fes.is_open()) {
            auto buyCnnExtra = std::make_unique<CNN>();
            auto sellCnnExtra = std::make_unique<CNN>();
            if (buyCnnExtra->load(feb) && sellCnnExtra->load(fes)) {
                extraBuyModels.push_back(std::move(buyCnnExtra));
                extraSellModels.push_back(std::move(sellCnnExtra));
            }
        }
        if (feb.is_open()) feb.close();
        if (fes.is_open()) fes.close();
    }

    if (modelGroups.empty()) {
        return 1;
    }

    std::vector<Candle> history;
    std::vector<double> closes;

    parseLiveCSV(std::cin, history, closes);

    if (history.size() < 100) {
        return 0;
    }

    std::vector<Candle> kalmanHistory;
    std::vector<double> kalmanCloses;
    Utils::applyKalmanFilter(history, kalmanHistory, kalmanCloses);

    std::vector<double> rsi = Utils::calculateRSI(kalmanCloses, 9);
    std::vector<double> ema20 = Utils::calculateEMA(kalmanCloses, 20);
    std::vector<double> atr = Utils::calculateATR(kalmanHistory, 14);
    std::vector<double> adx = Utils::calculateADX(kalmanHistory, 14);
    std::vector<double> bbPct = Utils::calculateBB_PctB(kalmanCloses, 20, 2.0);

    int lastIndex = kalmanHistory.size() - 1; 

    std::string extraSignal = "ANY";
    
    if (!extraBuyModels.empty() && Config::TIMEFRAME_MULTIPLIER > 1) {
        extraSignal = "HOLD";
        std::vector<Candle> extraHistory;
        std::vector<double> extraCloses;
        
        aggregateCandles(history, Config::TIMEFRAME_MULTIPLIER, extraHistory, extraCloses);

        if (extraHistory.size() >= 50) {
            std::vector<Candle> extraKalmanHistory;
            std::vector<double> extraKalmanCloses;
            Utils::applyKalmanFilter(extraHistory, extraKalmanHistory, extraKalmanCloses);

            std::vector<double> extraRsi = Utils::calculateRSI(extraKalmanCloses, 9);
            std::vector<double> extraEma20 = Utils::calculateEMA(extraKalmanCloses, 20);
            std::vector<double> extraAtr = Utils::calculateATR(extraKalmanHistory, 14);
            std::vector<double> extraAdx = Utils::calculateADX(extraKalmanHistory, 14);
            std::vector<double> extraBbPct = Utils::calculateBB_PctB(extraKalmanCloses, 20, 2.0);

            int extraLastIndex = extraKalmanHistory.size() - 1;
            double totalExtraBuy = 0.0;
            double totalExtraSell = 0.0;
            int validExtra = 0;

            for (size_t i = 0; i < extraBuyModels.size(); ++i) {
                int steps = extraBuyModels[i]->config.inputTimeSteps;
                int feats = extraBuyModels[i]->config.inputFeatures;

                if (extraLastIndex < steps) continue;

                Tensor input = Utils::generateInputTensor(extraLastIndex, extraKalmanHistory, extraKalmanCloses, extraEma20, extraRsi, extraAtr, extraAdx, extraBbPct, steps, feats);

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

    int bestTarget = Config::TARGET_CANDLES;
    double bestConf = 0.0;
    double bestBuyProb = 0.0;
    double bestSellProb = 0.0;
    std::string finalSignal = "HOLD";

    if (extraSignal != "HOLD") {
        for (auto& pair : modelGroups) {
            int t = pair.first;
            auto& group = pair.second;
            
            double totalBuy = 0.0;
            double totalSell = 0.0;
            int valid = 0;

            for (size_t i = 0; i < group.buyModels.size(); ++i) {
                int steps = group.buyModels[i]->config.inputTimeSteps;
                int feats = group.buyModels[i]->config.inputFeatures;

                if (lastIndex < steps) continue;

                Tensor input = Utils::generateInputTensor(lastIndex, kalmanHistory, kalmanCloses, ema20, rsi, atr, adx, bbPct, steps, feats);

                if (input.depth > 0) {
                    totalBuy += group.buyModels[i]->predict(input);
                    totalSell += group.sellModels[i]->predict(input);
                    valid++;
                }
            }

            if (valid > 0) {
                double avgBuy = totalBuy / valid;
                double avgSell = totalSell / valid;
                double conf = std::max(avgBuy, avgSell) * 100.0;
                
                std::string grpSignal = "HOLD";
                if (conf > 50) {
                    if ((avgBuy - avgSell) > 0.01) grpSignal = "BUY";
                    else if ((avgSell - avgBuy) > 0.01) grpSignal = "SELL";
                }

                if (grpSignal != "HOLD" && (extraSignal == "ANY" || grpSignal == extraSignal)) {
                    if (conf > bestConf) {
                        bestConf = conf;
                        bestTarget = t;
                        bestBuyProb = avgBuy;
                        bestSellProb = avgSell;
                        finalSignal = grpSignal;
                    }
                }
            }
        }
    }

    std::cout << "JSON_START{"
              << "\"signal\": \"" << finalSignal << "\", "
              << "\"confidence\": " << std::fixed << std::setprecision(2) << bestConf << ", "
              << "\"prob_buy\": " << std::fixed << std::setprecision(6) << bestBuyProb << ", "
              << "\"prob_sell\": " << std::fixed << std::setprecision(6) << bestSellProb << ", "
              << "\"target\": " << bestTarget
              << "}JSON_END\n";

    return 0;
}