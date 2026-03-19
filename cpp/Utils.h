#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include "Config.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Candle {
    double open, high, low, close;
    int hour, minute;
};

class Tensor {
public:
    int depth, rows, cols;
    std::vector<double> data;

    Tensor() : depth(0), rows(0), cols(0) {}
    Tensor(int d, int r, int c) : depth(d), rows(r), cols(c) {
        data.resize(d * r * c, 0.0);
    }
    
    double& at(int d, int r, int c) { return data[d * (rows * cols) + r * cols + c]; }
    const double& at(int d, int r, int c) const { return data[d * (rows * cols) + r * cols + c]; }
};

class Utils {
public:
    static double safe_div(double n, double d) { return (std::abs(d) < 1e-9) ? 0.0 : n / d; }
    
    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static double d_sigmoid(double out) { return out * (1.0 - out); }
    static double leaky_relu(double x) { return (x > 0) ? x : 0.01 * x; }
    static double d_leaky_relu(double x) { return (x > 0) ? 1.0 : 0.01; }
    
    static std::vector<double> softmax(const std::vector<double>& input) {
        std::vector<double> output(input.size());
        double maxVal = -1e9;
        for(double val : input) if(val > maxVal) maxVal = val;

        double sum = 0.0;
        for(size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - maxVal);
            sum += output[i];
        }
        for(size_t i = 0; i < input.size(); ++i) {
            output[i] /= sum;
        }
        return output;
    }

    static void parseTime(const std::string& timeStr, int& hh, int& mm) {
        size_t firstColon = timeStr.find(':');
        if (firstColon != std::string::npos) {
            hh = std::stoi(timeStr.substr(0, firstColon));
            size_t secondColon = timeStr.find(':', firstColon + 1);
            if (secondColon != std::string::npos) {
                mm = std::stoi(timeStr.substr(firstColon + 1, secondColon - firstColon - 1));
            } else {
                mm = std::stoi(timeStr.substr(firstColon + 1));
            }
        } else {
            hh = 0; mm = 0;
        }
    }

    static void applyKalmanFilter(const std::vector<Candle>& rawHistory, std::vector<Candle>& kalmanHistory, std::vector<double>& kalmanCloses) {
        kalmanHistory.clear();
        kalmanCloses.clear();
        int n = rawHistory.size();
        if (n == 0) return;

        kalmanHistory.reserve(n);
        kalmanCloses.reserve(n);

        double q = Config::KALMAN_PROCESS_NOISE;
        double r = Config::KALMAN_MEASUREMENT_NOISE;

        double x_o = rawHistory[0].open;
        double x_h = rawHistory[0].high;
        double x_l = rawHistory[0].low;
        double x_c = rawHistory[0].close;

        double p_o = 1.0, p_h = 1.0, p_l = 1.0, p_c = 1.0;

        for (int i = 0; i < n; ++i) {
            p_o += q;
            p_h += q;
            p_l += q;
            p_c += q;

            double k_o = p_o / (p_o + r);
            double k_h = p_h / (p_h + r);
            double k_l = p_l / (p_l + r);
            double k_c = p_c / (p_c + r);

            x_o += k_o * (rawHistory[i].open - x_o);
            x_h += k_h * (rawHistory[i].high - x_h);
            x_l += k_l * (rawHistory[i].low - x_l);
            x_c += k_c * (rawHistory[i].close - x_c);

            p_o *= (1.0 - k_o);
            p_h *= (1.0 - k_h);
            p_l *= (1.0 - k_l);
            p_c *= (1.0 - k_c);

            Candle k_candle;
            k_candle.open = x_o;
            k_candle.close = x_c;
            k_candle.high = std::max({x_h, x_o, x_c});
            k_candle.low = std::min({x_l, x_o, x_c});
            k_candle.hour = rawHistory[i].hour;
            k_candle.minute = rawHistory[i].minute;

            kalmanHistory.push_back(k_candle);
            kalmanCloses.push_back(k_candle.close);
        }
    }

    static std::vector<double> calculateRSI(const std::vector<double>& prices, int period) {
        std::vector<double> rsi(prices.size(), 50.0);
        if (prices.size() <= period) return rsi;
        
        double gain = 0.0, loss = 0.0;
        for (int i = 1; i <= period; ++i) {
            double change = prices[i] - prices[i - 1];
            if (change > 0) gain += change; else loss -= change;
        }
        
        gain /= period; loss /= period;
        rsi[period] = (loss == 0) ? 100.0 : 100.0 - (100.0 / (1.0 + gain / loss));

        for (size_t i = period + 1; i < prices.size(); ++i) {
            double change = prices[i] - prices[i - 1];
            double g = (change > 0) ? change : 0.0;
            double l = (change < 0) ? -change : 0.0;
            
            gain = (gain * (period - 1) + g) / period;
            loss = (loss * (period - 1) + l) / period;
            rsi[i] = (loss == 0) ? 100.0 : 100.0 - (100.0 / (1.0 + gain / loss));
        }
        return rsi;
    }

    static std::vector<double> calculateEMA(const std::vector<double>& prices, int period) {
        std::vector<double> ema(prices.size(), 0.0);
        if (prices.empty()) return ema;
        double k = 2.0 / (period + 1);
        ema[0] = prices[0];
        for (size_t i = 1; i < prices.size(); ++i) ema[i] = prices[i] * k + ema[i - 1] * (1.0 - k);
        return ema;
    }

    static std::vector<double> calculateATR(const std::vector<Candle>& history, int period) {
        std::vector<double> atr(history.size(), 0.0);
        if (history.size() <= period) return atr;
        
        std::vector<double> tr(history.size(), 0.0);
        for(size_t i=1; i<history.size(); ++i) {
            double hl = history[i].high - history[i].low;
            double hc = std::abs(history[i].high - history[i-1].close);
            double lc = std::abs(history[i].low - history[i-1].close);
            tr[i] = std::max({hl, hc, lc});
        }

        double sum = 0.0;
        for(int i=1; i<=period; ++i) sum += tr[i];
        atr[period] = sum / period;

        for(size_t i=period+1; i<history.size(); ++i) {
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period;
        }
        return atr;
    }

    static std::vector<double> calculateADX(const std::vector<Candle>& history, int period) {
        std::vector<double> adx(history.size(), 0.0);
        if (history.size() <= period * 2) return adx;

        std::vector<double> tr(history.size(), 0.0);
        std::vector<double> dmPlus(history.size(), 0.0);
        std::vector<double> dmMinus(history.size(), 0.0);

        for (size_t i = 1; i < history.size(); ++i) {
            double hl = history[i].high - history[i].low;
            double hc = std::abs(history[i].high - history[i-1].close);
            double lc = std::abs(history[i].low - history[i-1].close);
            tr[i] = std::max({hl, hc, lc});

            double up = history[i].high - history[i-1].high;
            double down = history[i-1].low - history[i].low;

            if (up > down && up > 0) dmPlus[i] = up;
            else dmPlus[i] = 0.0;
            
            if (down > up && down > 0) dmMinus[i] = down;
            else dmMinus[i] = 0.0;
        }

        double sTR = 0, sP = 0, sM = 0;
        for(int i=1; i<=period; ++i) { sTR += tr[i]; sP += dmPlus[i]; sM += dmMinus[i]; }
        
        std::vector<double> dx(history.size(), 0.0);
        
        for(size_t i=period+1; i<history.size(); ++i) {
            sTR = sTR - (sTR/period) + tr[i];
            sP = sP - (sP/period) + dmPlus[i];
            sM = sM - (sM/period) + dmMinus[i];
            
            double diPlus = 100.0 * safe_div(sP, sTR);
            double diMinus = 100.0 * safe_div(sM, sTR);
            double sumDI = diPlus + diMinus;
            
            if(sumDI > 0) dx[i] = 100.0 * std::abs(diPlus - diMinus) / sumDI;
        }

        double sumDX = 0;
        for(int i=period+1; i<=2*period; ++i) sumDX += dx[i];
        adx[2*period] = sumDX / period;

        for(size_t i=2*period+1; i<history.size(); ++i) {
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period;
        }
        return adx;
    }

    static std::vector<double> calculateBB_PctB(const std::vector<double>& prices, int period, double stdDevMult) {
        std::vector<double> pctB(prices.size(), 0.0);
        if (prices.size() < period) return pctB;

        for (size_t i = period - 1; i < prices.size(); ++i) {
            double sum = 0.0;
            for (int j = 0; j < period; ++j) sum += prices[i - j];
            double sma = sum / period;

            double sqSum = 0.0;
            for (int j = 0; j < period; ++j) sqSum += std::pow(prices[i - j] - sma, 2);
            double stdDev = std::sqrt(sqSum / period);

            double upper = sma + stdDevMult * stdDev;
            double lower = sma - stdDevMult * stdDev;

            pctB[i] = safe_div(prices[i] - lower, upper - lower);
        }
        return pctB;
    }

    static double calculateLinearSlope(const std::vector<double>& prices, int idx, int period) {
        if(idx < period) return 0.0;
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for(int i=0; i<period; ++i) {
            double x = i;
            double y = prices[idx - period + 1 + i];
            sumX += x; sumY += y; sumXY += x * y; sumX2 += x * x;
        }
        double denominator = (period * sumX2 - sumX * sumX);
        if(std::abs(denominator) < 1e-9) return 0.0;
        return (period * sumXY - sumX * sumY) / denominator;
    }

    static Tensor generateInputTensor(int idx, const std::vector<Candle>& history, 
                                    const std::vector<double>& closes, 
                                    const std::vector<double>& ema20,
                                    const std::vector<double>& rsiVals,
                                    const std::vector<double>& atrVals,
                                    const std::vector<double>& adxVals,
                                    const std::vector<double>& bbPctB,
                                    int inputTimesteps, 
                                    int inputFeatures) {
        
        int startIdx = idx - inputTimesteps + 1;
        
        if (startIdx < 50 || idx >= history.size()) return Tensor();

        Tensor image(1, inputTimesteps, inputFeatures);

        for (int i = 0; i < inputTimesteps; ++i) {
            int curr = startIdx + i;
            const Candle& c = history[curr];
            const Candle& prev = history[curr-1];
            
            double atr = atrVals[curr] > 1e-9 ? atrVals[curr] : 1.0;

            if (0 < inputFeatures) image.at(0, i, 0) = (c.close - prev.close) / atr;
            if (1 < inputFeatures) image.at(0, i, 1) = (c.high - c.low) / atr;
            
            double totalLen = c.high - c.low;
            if (2 < inputFeatures) image.at(0, i, 2) = safe_div(c.close - c.low, totalLen);

            double upperWick = c.high - std::max(c.open, c.close);
            double lowerWick = std::min(c.open, c.close) - c.low;
            if (3 < inputFeatures) image.at(0, i, 3) = safe_div(upperWick - lowerWick, totalLen);

            if (Config::IDX_RSI < inputFeatures) image.at(0, i, Config::IDX_RSI) = rsiVals[curr] / 100.0;

            double rsiVel = rsiVals[curr] - rsiVals[curr-1];
            double rsiPrevVel = rsiVals[curr-1] - rsiVals[curr-2];
            if (5 < inputFeatures) image.at(0, i, 5) = (rsiVel - rsiPrevVel) / 100.0;

            if (6 < inputFeatures) image.at(0, i, 6) = (c.close - ema20[curr]) / atr;

            if (7 < inputFeatures) image.at(0, i, 7) = atrVals[curr-1] > 1e-9 ? (atrVals[curr] - atrVals[curr-1]) / atrVals[curr-1] : 0.0;

            int momIdx = curr - 4;
            if (8 < inputFeatures) image.at(0, i, 8) = (momIdx >= 0) ? ((c.close - closes[momIdx]) / atr) : 0.0;

            double vel = c.close - prev.close;
            double prevVel = prev.close - history[curr-2].close;
            if (9 < inputFeatures) image.at(0, i, 9) = (vel - prevVel) / atr;

            if (10 < inputFeatures) image.at(0, i, 10) = calculateLinearSlope(closes, curr, 5) / atr;

            double currentTR = std::max(c.high - c.low, std::abs(c.high - prev.close));
            if (11 < inputFeatures) image.at(0, i, 11) = safe_div(currentTR, atrVals[curr]);

            if (Config::IDX_ADX < inputFeatures) image.at(0, i, Config::IDX_ADX) = adxVals[curr] / 100.0;

            double periodHigh = -1e9, periodLow = 1e9;
            for(int k=0; k<14 && (curr-k)>=0; ++k) {
                periodHigh = std::max(periodHigh, history[curr-k].high);
                periodLow = std::min(periodLow, history[curr-k].low);
            }
            if (Config::IDX_PRICE_POS < inputFeatures) image.at(0, i, Config::IDX_PRICE_POS) = safe_div(c.close - periodLow, periodHigh - periodLow);

            if (Config::IDX_BB_PCT < inputFeatures) image.at(0, i, Config::IDX_BB_PCT) = bbPctB[curr];

            if (15 < inputFeatures) image.at(0, i, 15) = (adxVals[curr] - adxVals[curr-1]) / 100.0;
        }

        return image;
    }
};

#endif