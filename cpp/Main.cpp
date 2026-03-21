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

    std::map<int, ModelGroup> workerGroups;

    for (int t = 1; t <= 10; ++t) {
        for (int i = 1; i <= Config::MAX_ENSEMBLE_MODELS; ++i) {
            std::string buyPath = Config::MODEL_FILE_WORKERS + "_t" + std::to_string(t) + "_buy_v" + std::to_string(i) + ".bin";
            std::string sellPath = Config::MODEL_FILE_WORKERS + "_t" + std::to_string(t) + "_sell_v" + std::to_string(i) + ".bin";
            std::ifstream fb(buyPath, std::ios::binary);
            std::ifstream fs(sellPath, std::ios::binary);
            
            if (fb.is_open() && fs.is_open()) {
                auto buyCnn = std::make_unique<CNN>();
                auto sellCnn = std::make_unique<CNN>();
                if (buyCnn->load(fb) && sellCnn->load(fs)) {
                    workerGroups[t].targetCandles = t;
                    workerGroups[t].buyModels.push_back(std::move(buyCnn));
                    workerGroups[t].sellModels.push_back(std::move(sellCnn));
                }
            }
            if (fb.is_open()) fb.close();
            if (fs.is_open()) fs.close();
        }
    }

    std::vector<std::unique_ptr<CNN>> masterBuyModels;
    std::vector<std::unique_ptr<CNN>> masterSellModels;

    for (int i = 1; i <= Config::MAX_ENSEMBLE_MODELS; ++i) {
        std::string masterBuyPath = Config::MODEL_FILE_MASTERS + "_t1_buy_v" + std::to_string(i) + ".bin";
        std::string masterSellPath = Config::MODEL_FILE_MASTERS + "_t1_sell_v" + std::to_string(i) + ".bin";
        std::ifstream feb(masterBuyPath, std::ios::binary);
        std::ifstream fes(masterSellPath, std::ios::binary);
        
        if (feb.is_open() && fes.is_open()) {
            auto buyCnnMaster = std::make_unique<CNN>();
            auto sellCnnMaster = std::make_unique<CNN>();
            if (buyCnnMaster->load(feb) && sellCnnMaster->load(fes)) {
                masterBuyModels.push_back(std::move(buyCnnMaster));
                masterSellModels.push_back(std::move(sellCnnMaster));
            }
        }
        if (feb.is_open()) feb.close();
        if (fes.is_open()) fes.close();
    }

    if (workerGroups.empty()) {
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

    std::string masterDecision = "ANY";
    
    if (!masterBuyModels.empty() && Config::TIMEFRAME_MULTIPLIER > 1) {
        masterDecision = "HOLD";
        std::vector<Candle> masterHistory;
        std::vector<double> masterCloses;
        
        aggregateCandles(history, Config::TIMEFRAME_MULTIPLIER, masterHistory, masterCloses);

        if (masterHistory.size() >= 50) {
            std::vector<Candle> masterKalmanHistory;
            std::vector<double> masterKalmanCloses;
            Utils::applyKalmanFilter(masterHistory, masterKalmanHistory, masterKalmanCloses);

            std::vector<double> masterRsi = Utils::calculateRSI(masterKalmanCloses, 9);
            std::vector<double> masterEma20 = Utils::calculateEMA(masterKalmanCloses, 20);
            std::vector<double> masterAtr = Utils::calculateATR(masterKalmanHistory, 14);
            std::vector<double> masterAdx = Utils::calculateADX(masterKalmanHistory, 14);
            std::vector<double> masterBbPct = Utils::calculateBB_PctB(masterKalmanCloses, 20, 2.0);

            int masterLastIndex = masterKalmanHistory.size() - 1;
            double totalMasterBuy = 0.0;
            double totalMasterSell = 0.0;
            int validMaster = 0;

            for (size_t i = 0; i < masterBuyModels.size(); ++i) {
                int steps = masterBuyModels[i]->config.inputTimeSteps;
                int feats = masterBuyModels[i]->config.inputFeatures;

                if (masterLastIndex < steps) continue;

                Tensor input = Utils::generateInputTensor(masterLastIndex, masterKalmanHistory, masterKalmanCloses, masterEma20, masterRsi, masterAtr, masterAdx, masterBbPct, steps, feats);

                if (input.depth > 0) {
                    totalMasterBuy += masterBuyModels[i]->predict(input);
                    totalMasterSell += masterSellModels[i]->predict(input);
                    validMaster++;
                }
            }

            if (validMaster > 0) {
                double masterAvgBuy = totalMasterBuy / validMaster;
                double masterAvgSell = totalMasterSell / validMaster;
                double masterConf = std::max(masterAvgBuy, masterAvgSell) * 100.0;

                if (masterConf > 50) {
                    if ((masterAvgBuy - masterAvgSell) > 0.01) {
                        masterDecision = "BUY";
                    } else if ((masterAvgSell - masterAvgBuy) > 0.01) {
                        masterDecision = "SELL";
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

    if (masterDecision != "HOLD") {
        for (auto& pair : workerGroups) {
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
                
                std::string workerProposal = "HOLD";
                if (conf > 50) {
                    if ((avgBuy - avgSell) > 0.01) workerProposal = "BUY";
                    else if ((avgSell - avgBuy) > 0.01) workerProposal = "SELL";
                }

                if (workerProposal != "HOLD" && (masterDecision == "ANY" || workerProposal == masterDecision)) {
                    if (conf > bestConf) {
                        bestConf = conf;
                        bestTarget = t;
                        bestBuyProb = avgBuy;
                        bestSellProb = avgSell;
                        finalSignal = workerProposal;
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