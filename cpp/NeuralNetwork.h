#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include "Config.h"
#include "Utils.h"

class Layer {
public:
    virtual ~Layer() {}
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradOutput) = 0;
    virtual void update(double lr, int t) = 0;
    virtual void scaleGradients(double scale) {}
    virtual void clearGradients() {}
    virtual void setMode(bool training) {}
    virtual void save(std::ofstream& file) {}
    virtual bool load(std::ifstream& file) { return true; }
};

class ConvLayer : public Layer {
private:
    int numFilters, kernelH, kernelW, stride;
    int inputDepth, inputRows, inputCols;
    int kernelSize, filterSize;
    
    std::vector<double> filters;
    std::vector<double> biases;
    Tensor lastInput;
    Tensor lastPreActivation;
    
    std::vector<double> filterGrads;
    std::vector<double> biasGrads;
    
    std::vector<double> filterM;
    std::vector<double> filterV;
    std::vector<double> biasM;
    std::vector<double> biasV;

public:
    ConvLayer(int filtersCount, int kH, int kW, int inDepth, int inRows, int inCols, int s)
        : numFilters(filtersCount), kernelH(kH), kernelW(kW), inputDepth(inDepth), 
          inputRows(inRows), inputCols(inCols), stride(s) {
        
        kernelSize = kernelH * kernelW;
        filterSize = inputDepth * kernelSize;
        
        int totalParams = numFilters * filterSize;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0, std::sqrt(2.0 / filterSize));

        filters.resize(totalParams);
        filterGrads.resize(totalParams, 0.0);
        filterM.resize(totalParams, 0.0);
        filterV.resize(totalParams, 0.0);
        
        biases.resize(numFilters, 0.0);
        biasGrads.resize(numFilters, 0.0);
        biasM.resize(numFilters, 0.0);
        biasV.resize(numFilters, 0.0);

        for (int i = 0; i < totalParams; ++i) {
            filters[i] = dist(gen);
        }
    }

    Tensor forward(const Tensor& input) override {
        lastInput = input;
        int outRows = (inputRows - kernelH) / stride + 1;
        int outCols = (inputCols - kernelW) / stride + 1;
        Tensor output(numFilters, outRows, outCols);
        lastPreActivation = Tensor(numFilters, outRows, outCols);

        for (int f = 0; f < numFilters; ++f) {
            for (int r = 0; r < outRows; ++r) {
                for (int c = 0; c < outCols; ++c) {
                    double sum = 0.0;
                    int startR = r * stride;
                    int startC = c * stride;
                    int filterOffset = f * filterSize;
                    
                    for (int d = 0; d < inputDepth; ++d) {
                        int inputDepthOffset = d * (inputRows * inputCols);
                        int filterDepthOffset = filterOffset + d * kernelSize;
                        
                        for (int kr = 0; kr < kernelH; ++kr) {
                            int inputRowOffset = inputDepthOffset + (startR + kr) * inputCols;
                            int filterRowOffset = filterDepthOffset + kr * kernelW;
                            
                            for (int kc = 0; kc < kernelW; ++kc) {
                                sum += input.data[inputRowOffset + startC + kc] * filters[filterRowOffset + kc];
                            }
                        }
                    }
                    double preAct = sum + biases[f];
                    lastPreActivation.at(f, r, c) = preAct;
                    output.at(f, r, c) = Utils::leaky_relu(preAct);
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        Tensor gradInput(inputDepth, inputRows, inputCols);
        
        for (int f = 0; f < numFilters; ++f) {
            double biasGradAccumulator = 0.0;
            int filterOffset = f * filterSize;
            
            for (int r = 0; r < gradOutput.rows; ++r) {
                for (int c = 0; c < gradOutput.cols; ++c) {
                    double grad = gradOutput.at(f, r, c);
                    double preAct = lastPreActivation.at(f, r, c);
                    grad *= Utils::d_leaky_relu(preAct);
                    biasGradAccumulator += grad;

                    int startR = r * stride;
                    int startC = c * stride;

                    for (int d = 0; d < inputDepth; ++d) {
                        int inputDepthOffset = d * (inputRows * inputCols);
                        int filterDepthOffset = filterOffset + d * kernelSize;

                        for (int kr = 0; kr < kernelH; ++kr) {
                            int inputRowOffset = inputDepthOffset + (startR + kr) * inputCols;
                            int filterRowOffset = filterDepthOffset + kr * kernelW;
                            
                            for (int kc = 0; kc < kernelW; ++kc) {
                                int inputIdx = inputRowOffset + startC + kc;
                                int filterIdx = filterRowOffset + kc;
                                
                                filterGrads[filterIdx] += grad * lastInput.data[inputIdx];
                                gradInput.data[inputIdx] += grad * filters[filterIdx];
                            }
                        }
                    }
                }
            }
            biasGrads[f] += biasGradAccumulator;
        }
        return gradInput;
    }

    void scaleGradients(double scale) override {
        for (size_t i = 0; i < biases.size(); ++i) biasGrads[i] *= scale;
        for (size_t i = 0; i < filters.size(); ++i) filterGrads[i] *= scale;
    }

    void clearGradients() override {
        std::fill(biasGrads.begin(), biasGrads.end(), 0.0);
        std::fill(filterGrads.begin(), filterGrads.end(), 0.0);
    }

    void update(double lr, int t) override {
        for (int f = 0; f < numFilters; ++f) {
            if (biasGrads[f] > 5.0) biasGrads[f] = 5.0;
            if (biasGrads[f] < -5.0) biasGrads[f] = -5.0;

            biasM[f] = Config::BETA1 * biasM[f] + (1.0 - Config::BETA1) * biasGrads[f];
            biasV[f] = Config::BETA2 * biasV[f] + (1.0 - Config::BETA2) * biasGrads[f] * biasGrads[f];
            
            double mHat = biasM[f] / (1.0 - std::pow(Config::BETA1, t));
            double vHat = biasV[f] / (1.0 - std::pow(Config::BETA2, t));
            
            biases[f] -= lr * mHat / (std::sqrt(vHat) + Config::EPSILON);
        }
        
        for (size_t i = 0; i < filters.size(); ++i) {
            if (filterGrads[i] > 5.0) filterGrads[i] = 5.0;
            if (filterGrads[i] < -5.0) filterGrads[i] = -5.0;

            double grad = filterGrads[i];
            filterM[i] = Config::BETA1 * filterM[i] + (1.0 - Config::BETA1) * grad;
            filterV[i] = Config::BETA2 * filterV[i] + (1.0 - Config::BETA2) * grad * grad;
            
            double fmHat = filterM[i] / (1.0 - std::pow(Config::BETA1, t));
            double fvHat = filterV[i] / (1.0 - std::pow(Config::BETA2, t));
            
            filters[i] -= lr * fmHat / (std::sqrt(fvHat) + Config::EPSILON);
        }
    }

    void save(std::ofstream& file) override {
        if (!file.good()) return;
        file.write(reinterpret_cast<const char*>(filters.data()), filters.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(double));
    }

    bool load(std::ifstream& file) override {
        if (!file.good()) return false;
        file.read(reinterpret_cast<char*>(filters.data()), filters.size() * sizeof(double));
        if(!file) return false;
        file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(double));
        if(!file) return false;
        return true;
    }
};

class LayerNormLayer : public Layer {
private:
    double epsilon = 1e-8;
    
    std::vector<double> gamma;
    std::vector<double> beta;
    
    std::vector<double> gammaGrads;
    std::vector<double> betaGrads;
    
    std::vector<double> gammaM, gammaV;
    std::vector<double> betaM, betaV;

    Tensor normalizedInput;
    std::vector<double> sampleMeans;
    std::vector<double> sampleInvStds;

public:
    LayerNormLayer(int size) { 
        gamma.assign(size, 1.0);
        beta.assign(size, 0.0);
        
        gammaGrads.assign(size, 0.0);
        betaGrads.assign(size, 0.0);
        
        gammaM.assign(size, 0.0); gammaV.assign(size, 0.0);
        betaM.assign(size, 0.0); betaV.assign(size, 0.0);
    }

    Tensor forward(const Tensor& input) override {
        normalizedInput = Tensor(input.depth, input.rows, input.cols);
        Tensor output(input.depth, input.rows, input.cols);
        
        int N = input.rows * input.cols;
        sampleMeans.resize(input.depth);
        sampleInvStds.resize(input.depth);
        
        for (int d = 0; d < input.depth; ++d) {
            double sum = 0.0;
            for (int i = 0; i < N; ++i) {
                sum += input.data[d * N + i];
            }
            double mean = sum / N;
            sampleMeans[d] = mean;

            double sqSum = 0.0;
            for (int i = 0; i < N; ++i) {
                double diff = input.data[d * N + i] - mean;
                sqSum += diff * diff;
            }
            double variance = sqSum / N;
            double invStd = 1.0 / std::sqrt(variance + epsilon);
            sampleInvStds[d] = invStd;

            for (int i = 0; i < N; ++i) {
                int idx = d * N + i;
                double normVal = (input.data[idx] - mean) * invStd;
                normalizedInput.data[idx] = normVal;
                
                int paramIdx = ((int)gamma.size() == input.depth) ? d : i; 
                output.data[idx] = gamma[paramIdx] * normVal + beta[paramIdx];
            }
        }
        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        Tensor gradInput(gradOutput.depth, gradOutput.rows, gradOutput.cols);
        int N = gradOutput.rows * gradOutput.cols;
        
        for (int d = 0; d < gradOutput.depth; ++d) {
            for (int i = 0; i < N; ++i) {
                int idx = d * N + i;
                int paramIdx = ((int)gamma.size() == gradOutput.depth) ? d : i;
                double grad = gradOutput.data[idx];
                
                betaGrads[paramIdx] += grad;
                gammaGrads[paramIdx] += grad * normalizedInput.data[idx];
            }
        }
        
        for (int d = 0; d < gradOutput.depth; ++d) {
            double sum_dx_hat = 0.0;
            double sum_dx_hat_x_hat = 0.0;
            
            for (int i = 0; i < N; ++i) {
                int idx = d * N + i;
                int paramIdx = ((int)gamma.size() == gradOutput.depth) ? d : i;
                double dx_hat = gradOutput.data[idx] * gamma[paramIdx];
                sum_dx_hat += dx_hat;
                sum_dx_hat_x_hat += dx_hat * normalizedInput.data[idx];
            }

            double invStd = sampleInvStds[d];
            for (int i = 0; i < N; ++i) {
                int idx = d * N + i;
                int paramIdx = ((int)gamma.size() == gradOutput.depth) ? d : i;
                double dx_hat = gradOutput.data[idx] * gamma[paramIdx];
                
                double term1 = N * dx_hat;
                double term2 = sum_dx_hat;
                double term3 = normalizedInput.data[idx] * sum_dx_hat_x_hat;
                
                gradInput.data[idx] = (invStd / N) * (term1 - term2 - term3);
            }
        }

        return gradInput;
    }

    void scaleGradients(double scale) override {
        for(auto& g : gammaGrads) g *= scale;
        for(auto& g : betaGrads) g *= scale;
    }

    void update(double lr, int t) override {
        for(size_t i = 0; i < gamma.size(); ++i) {
            gammaGrads[i] = std::max(-5.0, std::min(5.0, gammaGrads[i]));
            betaGrads[i] = std::max(-5.0, std::min(5.0, betaGrads[i]));

            gammaM[i] = Config::BETA1 * gammaM[i] + (1.0 - Config::BETA1) * gammaGrads[i];
            gammaV[i] = Config::BETA2 * gammaV[i] + (1.0 - Config::BETA2) * gammaGrads[i] * gammaGrads[i];
            double gmHat = gammaM[i] / (1.0 - std::pow(Config::BETA1, t));
            double gvHat = gammaV[i] / (1.0 - std::pow(Config::BETA2, t));
            gamma[i] -= lr * gmHat / (std::sqrt(gvHat) + Config::EPSILON);

            betaM[i] = Config::BETA1 * betaM[i] + (1.0 - Config::BETA1) * betaGrads[i];
            betaV[i] = Config::BETA2 * betaV[i] + (1.0 - Config::BETA2) * betaGrads[i] * betaGrads[i];
            double bmHat = betaM[i] / (1.0 - std::pow(Config::BETA1, t));
            double bvHat = betaV[i] / (1.0 - std::pow(Config::BETA2, t));
            beta[i] -= lr * bmHat / (std::sqrt(bvHat) + Config::EPSILON);
        }
    }

    void clearGradients() override {
        std::fill(gammaGrads.begin(), gammaGrads.end(), 0.0);
        std::fill(betaGrads.begin(), betaGrads.end(), 0.0);
    }
    
    void save(std::ofstream& file) override {
        if (!file.good()) return;
        file.write(reinterpret_cast<const char*>(gamma.data()), gamma.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(beta.data()), beta.size() * sizeof(double));
    }

    bool load(std::ifstream& file) override {
        if (!file.good()) return false;
        file.read(reinterpret_cast<char*>(gamma.data()), gamma.size() * sizeof(double));
        file.read(reinterpret_cast<char*>(beta.data()), beta.size() * sizeof(double));
        return true;
    }
};

class LSTMLayer : public Layer {
private:
    int inputSize, hiddenSize;
    int seqLen, lastDepth, lastRows, lastCols;
    
    std::vector<double> W, U, b;
    std::vector<double> gradW, gradU, gradB;
    std::vector<double> mW, vW, mU, vU, mB, vB;

    std::vector<std::vector<double>> x_hist;
    std::vector<std::vector<double>> h_hist;
    std::vector<std::vector<double>> c_hist;
    std::vector<std::vector<double>> i_hist;
    std::vector<std::vector<double>> f_hist;
    std::vector<std::vector<double>> c_tilde_hist;
    std::vector<std::vector<double>> o_hist;

    std::vector<double> A;
    std::vector<double> dA;
    std::vector<double> dx;
    std::vector<double> dh;

public:
    LSTMLayer(int inSize, int hSize) : inputSize(inSize), hiddenSize(hSize) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> distW(0, std::sqrt(1.0 / inputSize));
        std::normal_distribution<> distU(0, std::sqrt(1.0 / hiddenSize));

        W.resize(inputSize * 4 * hiddenSize);
        gradW.resize(inputSize * 4 * hiddenSize, 0.0);
        mW.resize(inputSize * 4 * hiddenSize, 0.0);
        vW.resize(inputSize * 4 * hiddenSize, 0.0);

        U.resize(hiddenSize * 4 * hiddenSize);
        gradU.resize(hiddenSize * 4 * hiddenSize, 0.0);
        mU.resize(hiddenSize * 4 * hiddenSize, 0.0);
        vU.resize(hiddenSize * 4 * hiddenSize, 0.0);

        b.resize(4 * hiddenSize, 0.0);
        gradB.resize(4 * hiddenSize, 0.0);
        mB.resize(4 * hiddenSize, 0.0);
        vB.resize(4 * hiddenSize, 0.0);

        A.resize(4 * hiddenSize, 0.0);
        dA.resize(4 * hiddenSize, 0.0);
        dx.resize(inputSize, 0.0);
        dh.resize(hiddenSize, 0.0);

        for (auto& val : W) val = distW(gen);
        for (auto& val : U) val = distU(gen);
    }

    Tensor forward(const Tensor& input) override {
        seqLen = input.rows;
        lastDepth = input.depth;
        lastRows = input.rows;
        lastCols = input.cols;

        if (x_hist.size() != seqLen) {
            x_hist.assign(seqLen, std::vector<double>(inputSize, 0.0));
            h_hist.assign(seqLen + 1, std::vector<double>(hiddenSize, 0.0));
            c_hist.assign(seqLen + 1, std::vector<double>(hiddenSize, 0.0));
            i_hist.assign(seqLen, std::vector<double>(hiddenSize, 0.0));
            f_hist.assign(seqLen, std::vector<double>(hiddenSize, 0.0));
            c_tilde_hist.assign(seqLen, std::vector<double>(hiddenSize, 0.0));
            o_hist.assign(seqLen, std::vector<double>(hiddenSize, 0.0));
        } else {
            for(auto& v : x_hist) std::fill(v.begin(), v.end(), 0.0);
            for(auto& v : h_hist) std::fill(v.begin(), v.end(), 0.0);
            for(auto& v : c_hist) std::fill(v.begin(), v.end(), 0.0);
            for(auto& v : i_hist) std::fill(v.begin(), v.end(), 0.0);
            for(auto& v : f_hist) std::fill(v.begin(), v.end(), 0.0);
            for(auto& v : c_tilde_hist) std::fill(v.begin(), v.end(), 0.0);
            for(auto& v : o_hist) std::fill(v.begin(), v.end(), 0.0);
        }

        for (int t = 0; t < seqLen; ++t) {
            int idx = 0;
            for (int d = 0; d < input.depth; ++d) {
                for (int c = 0; c < input.cols; ++c) {
                    x_hist[t][idx++] = input.at(d, t, c);
                }
            }

            std::fill(A.begin(), A.end(), 0.0);
            for (int k = 0; k < 4 * hiddenSize; ++k) A[k] = b[k];

            for (int j = 0; j < inputSize; ++j) {
                double xj = x_hist[t][j];
                for (int k = 0; k < 4 * hiddenSize; ++k) {
                    A[k] += xj * W[j * 4 * hiddenSize + k];
                }
            }

            for (int j = 0; j < hiddenSize; ++j) {
                double hj = h_hist[t][j];
                for (int k = 0; k < 4 * hiddenSize; ++k) {
                    A[k] += hj * U[j * 4 * hiddenSize + k];
                }
            }

            for (int j = 0; j < hiddenSize; ++j) {
                i_hist[t][j] = Utils::sigmoid(A[j]);
                f_hist[t][j] = Utils::sigmoid(A[hiddenSize + j]);
                c_tilde_hist[t][j] = std::tanh(A[2 * hiddenSize + j]);
                o_hist[t][j] = Utils::sigmoid(A[3 * hiddenSize + j]);

                c_hist[t + 1][j] = f_hist[t][j] * c_hist[t][j] + i_hist[t][j] * c_tilde_hist[t][j];
                h_hist[t + 1][j] = o_hist[t][j] * std::tanh(c_hist[t + 1][j]);
            }
        }

        Tensor output(1, seqLen, hiddenSize);
        for(int t = 0; t < seqLen; ++t) {
            for (int j = 0; j < hiddenSize; ++j) {
                output.at(0, t, j) = h_hist[t + 1][j];
            }
        }
        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        Tensor gradInput(lastDepth, lastRows, lastCols);
        std::vector<double> dh_next(hiddenSize, 0.0);
        std::vector<double> dc_next(hiddenSize, 0.0);

        for (int t = seqLen - 1; t >= 0; --t) {
            std::fill(dA.begin(), dA.end(), 0.0);
            std::fill(dx.begin(), dx.end(), 0.0);
            std::fill(dh.begin(), dh.end(), 0.0);

            for (int j = 0; j < hiddenSize; ++j) {
                double grad_ht = gradOutput.at(0, t, j) + dh_next[j];
                
                double tanh_c = std::tanh(c_hist[t + 1][j]);
                double dc = dc_next[j] + grad_ht * o_hist[t][j] * (1.0 - tanh_c * tanh_c);

                double do_val = grad_ht * tanh_c;
                double di_val = dc * c_tilde_hist[t][j];
                double df_val = dc * c_hist[t][j];
                double dct_val = dc * i_hist[t][j];

                dA[j] = di_val * i_hist[t][j] * (1.0 - i_hist[t][j]);
                dA[hiddenSize + j] = df_val * f_hist[t][j] * (1.0 - f_hist[t][j]);
                dA[2 * hiddenSize + j] = dct_val * (1.0 - c_tilde_hist[t][j] * c_tilde_hist[t][j]);
                dA[3 * hiddenSize + j] = do_val * o_hist[t][j] * (1.0 - o_hist[t][j]);

                dc_next[j] = dc * f_hist[t][j];
            }

            for (int k = 0; k < 4 * hiddenSize; ++k) {
                double da_k = dA[k];
                gradB[k] += da_k;

                for (int j = 0; j < inputSize; ++j) {
                    gradW[j * 4 * hiddenSize + k] += x_hist[t][j] * da_k;
                    dx[j] += da_k * W[j * 4 * hiddenSize + k];
                }

                for (int j = 0; j < hiddenSize; ++j) {
                    gradU[j * 4 * hiddenSize + k] += h_hist[t][j] * da_k;
                    dh[j] += da_k * U[j * 4 * hiddenSize + k];
                }
            }

            dh_next = dh;

            int idx = 0;
            for (int d = 0; d < lastDepth; ++d) {
                for (int c = 0; c < lastCols; ++c) {
                    gradInput.at(d, t, c) = dx[idx++];
                }
            }
        }
        return gradInput;
    }

    void scaleGradients(double scale) override {
        for (size_t i = 0; i < gradW.size(); ++i) gradW[i] *= scale;
        for (size_t i = 0; i < gradU.size(); ++i) gradU[i] *= scale;
        for (size_t i = 0; i < gradB.size(); ++i) gradB[i] *= scale;
    }

    void clearGradients() override {
        std::fill(gradW.begin(), gradW.end(), 0.0);
        std::fill(gradU.begin(), gradU.end(), 0.0);
        std::fill(gradB.begin(), gradB.end(), 0.0);
    }

    void update(double lr, int t) override {
        for (size_t i = 0; i < b.size(); ++i) {
            if (gradB[i] > 5.0) gradB[i] = 5.0;
            if (gradB[i] < -5.0) gradB[i] = -5.0;

            mB[i] = Config::BETA1 * mB[i] + (1.0 - Config::BETA1) * gradB[i];
            vB[i] = Config::BETA2 * vB[i] + (1.0 - Config::BETA2) * gradB[i] * gradB[i];
            double mHat = mB[i] / (1.0 - std::pow(Config::BETA1, t));
            double vHat = vB[i] / (1.0 - std::pow(Config::BETA2, t));
            b[i] -= lr * mHat / (std::sqrt(vHat) + Config::EPSILON);
        }

        for (size_t i = 0; i < W.size(); ++i) {
            if (gradW[i] > 5.0) gradW[i] = 5.0;
            if (gradW[i] < -5.0) gradW[i] = -5.0;

            mW[i] = Config::BETA1 * mW[i] + (1.0 - Config::BETA1) * gradW[i];
            vW[i] = Config::BETA2 * vW[i] + (1.0 - Config::BETA2) * gradW[i] * gradW[i];
            double mHat = mW[i] / (1.0 - std::pow(Config::BETA1, t));
            double vHat = vW[i] / (1.0 - std::pow(Config::BETA2, t));
            W[i] -= lr * mHat / (std::sqrt(vHat) + Config::EPSILON);
        }

        for (size_t i = 0; i < U.size(); ++i) {
            if (gradU[i] > 5.0) gradU[i] = 5.0;
            if (gradU[i] < -5.0) gradU[i] = -5.0;

            mU[i] = Config::BETA1 * mU[i] + (1.0 - Config::BETA1) * gradU[i];
            vU[i] = Config::BETA2 * vU[i] + (1.0 - Config::BETA2) * gradU[i] * gradU[i];
            double mHat = mU[i] / (1.0 - std::pow(Config::BETA1, t));
            double vHat = vU[i] / (1.0 - std::pow(Config::BETA2, t));
            U[i] -= lr * mHat / (std::sqrt(vHat) + Config::EPSILON);
        }
    }

    void save(std::ofstream& file) override {
        if (!file.good()) return;
        file.write(reinterpret_cast<const char*>(W.data()), W.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(U.data()), U.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(b.data()), b.size() * sizeof(double));
    }

    bool load(std::ifstream& file) override {
        if (!file.good()) return false;
        file.read(reinterpret_cast<char*>(W.data()), W.size() * sizeof(double));
        if(!file) return false;
        file.read(reinterpret_cast<char*>(U.data()), U.size() * sizeof(double));
        if(!file) return false;
        file.read(reinterpret_cast<char*>(b.data()), b.size() * sizeof(double));
        if(!file) return false;
        return true;
    }
};

class AttentionLayer : public Layer {
private:
    int inputSize;
    std::vector<double> attWeights;
    std::vector<double> attGrads;
    std::vector<double> attM, attV;
    std::vector<double> lastScores;
    std::vector<double> lastAlpha;
    std::vector<double> gradAlpha;
    Tensor lastInput;

public:
    AttentionLayer(int size) : inputSize(size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0, std::sqrt(2.0 / inputSize));
        
        attWeights.resize(inputSize);
        attGrads.resize(inputSize, 0.0);
        attM.resize(inputSize, 0.0);
        attV.resize(inputSize, 0.0);

        for(auto& w : attWeights) w = dist(gen);
    }

    Tensor forward(const Tensor& input) override {
        lastInput = input;
        int seqLen = input.rows;
        int hiddenSize = input.cols;
        
        lastScores.resize(seqLen);
        lastAlpha.resize(seqLen);
        gradAlpha.resize(seqLen);

        double maxVal = -1e9;
        for(int t = 0; t < seqLen; ++t) {
            double score = 0.0;
            for(int i = 0; i < hiddenSize; ++i) {
                score += input.at(0, t, i) * attWeights[i];
            }
            lastScores[t] = score; 
            if (score > maxVal) maxVal = score;
        }

        double sum = 0.0;
        for(int t = 0; t < seqLen; ++t) {
            lastAlpha[t] = std::exp(lastScores[t] - maxVal);
            sum += lastAlpha[t];
        }
        for(int t = 0; t < seqLen; ++t) {
            lastAlpha[t] /= sum;
        }

        Tensor output(1, hiddenSize, 1);
        for(int i = 0; i < hiddenSize; ++i) {
            double sumVal = 0.0;
            for(int t = 0; t < seqLen; ++t) {
                sumVal += input.at(0, t, i) * lastAlpha[t];
            }
            output.data[i] = sumVal;
        }
        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        int seqLen = lastInput.rows;
        int hiddenSize = lastInput.cols;
        Tensor gradInput(1, seqLen, hiddenSize);

        for (int t = 0; t < seqLen; ++t) {
            double dot = 0.0;
            for (int i = 0; i < hiddenSize; ++i) {
                dot += gradOutput.data[i] * lastInput.at(0, t, i);
            }
            gradAlpha[t] = dot;
        }

        double sumGradAlphaAlpha = 0.0;
        for (int t = 0; t < seqLen; ++t) {
            sumGradAlphaAlpha += gradAlpha[t] * lastAlpha[t];
        }

        for (int t = 0; t < seqLen; ++t) {
            double gScore = lastAlpha[t] * (gradAlpha[t] - sumGradAlphaAlpha);
            double alpha = lastAlpha[t];
            
            for (int i = 0; i < hiddenSize; ++i) {
                attGrads[i] += gScore * lastInput.at(0, t, i);
                gradInput.at(0, t, i) = (gradOutput.data[i] * alpha) + (gScore * attWeights[i]);
            }
        }
        
        return gradInput;
    }

    void scaleGradients(double scale) override {
        for(auto& g : attGrads) g *= scale;
    }

    void clearGradients() override {
        std::fill(attGrads.begin(), attGrads.end(), 0.0);
    }

    void update(double lr, int t) override {
        for(size_t i = 0; i < attWeights.size(); ++i) {
            if(attGrads[i] > 5.0) attGrads[i] = 5.0;
            if(attGrads[i] < -5.0) attGrads[i] = -5.0;

            attM[i] = Config::BETA1 * attM[i] + (1.0 - Config::BETA1) * attGrads[i];
            attV[i] = Config::BETA2 * attV[i] + (1.0 - Config::BETA2) * attGrads[i] * attGrads[i];
            double mHat = attM[i] / (1.0 - std::pow(Config::BETA1, t));
            double vHat = attV[i] / (1.0 - std::pow(Config::BETA2, t));
            attWeights[i] -= lr * mHat / (std::sqrt(vHat) + Config::EPSILON);
        }
    }

    void save(std::ofstream& file) override {
        if(!file.good()) return;
        file.write(reinterpret_cast<const char*>(attWeights.data()), attWeights.size() * sizeof(double));
    }

    bool load(std::ifstream& file) override {
        if(!file.good()) return false;
        file.read(reinterpret_cast<char*>(attWeights.data()), attWeights.size() * sizeof(double));
        return true;
    }
};

class DenseLayer : public Layer {
private:
    int inputSize, outputSize;
    std::vector<double> weights;
    std::vector<double> biases;
    std::vector<double> weightGrads;
    std::vector<double> biasGrads;
    std::vector<double> weightM;
    std::vector<double> weightV;
    std::vector<double> biasM;
    std::vector<double> biasV;
    Tensor lastInput;
    Tensor lastPreActivation;
    bool isOutput, useSigmoid;

public:
    DenseLayer(int inSize, int outSize, bool isOut = false, bool sigmoid = false)
        : inputSize(inSize), outputSize(outSize), isOutput(isOut), useSigmoid(sigmoid) {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0, std::sqrt(2.0 / inputSize));

        weights.resize(inputSize * outputSize);
        weightGrads.resize(inputSize * outputSize, 0.0);
        weightM.resize(inputSize * outputSize, 0.0);
        weightV.resize(inputSize * outputSize, 0.0);
        
        biases.resize(outputSize, 0.0);
        biasGrads.resize(outputSize, 0.0);
        biasM.resize(outputSize, 0.0);
        biasV.resize(outputSize, 0.0);

        lastPreActivation = Tensor(1, outputSize, 1);

        for (auto& w : weights) w = dist(gen);
    }

    Tensor forward(const Tensor& input) override {
        lastInput = input;
        Tensor output(1, outputSize, 1);
        
        for (int i = 0; i < outputSize; ++i) {
            double sum = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                sum += input.data[j] * weights[j * outputSize + i];
            }
            lastPreActivation.data[i] = sum;

            if (isOutput && useSigmoid)
                output.data[i] = Utils::sigmoid(sum);
            else
                output.data[i] = Utils::leaky_relu(sum);
        }
        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        Tensor gradInput(1, inputSize, 1);
        std::vector<double> deltas(outputSize);

        for (int i = 0; i < outputSize; ++i) {
            double preAct = lastPreActivation.data[i];
            double grad = gradOutput.data[i];
            
            if (isOutput && useSigmoid) {
            } else {
                grad *= Utils::d_leaky_relu(preAct);
            }
            deltas[i] = grad;
            biasGrads[i] += grad;
        }

        for (int j = 0; j < inputSize; ++j) {
            double inVal = lastInput.data[j];
            double gradSum = 0.0;
            for (int i = 0; i < outputSize; ++i) {
                int idx = j * outputSize + i;
                
                weightGrads[idx] += deltas[i] * inVal;
                gradSum += deltas[i] * weights[idx];
            }
            gradInput.data[j] = gradSum;
        }
        return gradInput;
    }

    void scaleGradients(double scale) override {
        for(size_t i=0; i<weightGrads.size(); ++i) weightGrads[i] *= scale;
        for(size_t i=0; i<biasGrads.size(); ++i) biasGrads[i] *= scale;
    }

    void clearGradients() override {
        std::fill(weightGrads.begin(), weightGrads.end(), 0.0);
        std::fill(biasGrads.begin(), biasGrads.end(), 0.0);
    }

    void update(double lr, int t) override {
        for (int i = 0; i < outputSize; ++i) {
            if (biasGrads[i] > 5.0) biasGrads[i] = 5.0;
            if (biasGrads[i] < -5.0) biasGrads[i] = -5.0;

            biasM[i] = Config::BETA1 * biasM[i] + (1.0 - Config::BETA1) * biasGrads[i];
            biasV[i] = Config::BETA2 * biasV[i] + (1.0 - Config::BETA2) * biasGrads[i] * biasGrads[i];
            
            double mHat = biasM[i] / (1.0 - std::pow(Config::BETA1, t));
            double vHat = biasV[i] / (1.0 - std::pow(Config::BETA2, t));
            
            biases[i] -= lr * mHat / (std::sqrt(vHat) + Config::EPSILON);
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            if (weightGrads[i] > 5.0) weightGrads[i] = 5.0;
            if (weightGrads[i] < -5.0) weightGrads[i] = -5.0;

            double grad = weightGrads[i];
            weightM[i] = Config::BETA1 * weightM[i] + (1.0 - Config::BETA1) * grad;
            weightV[i] = Config::BETA2 * weightV[i] + (1.0 - Config::BETA2) * grad * grad;
            
            double mHat = weightM[i] / (1.0 - std::pow(Config::BETA1, t));
            double vHat = weightV[i] / (1.0 - std::pow(Config::BETA2, t));
            
            weights[i] -= lr * mHat / (std::sqrt(vHat) + Config::EPSILON);
        }
    }

    void save(std::ofstream& file) override {
        if (!file.good()) return;
        file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(double));
    }

    bool load(std::ifstream& file) override {
        if (!file.good()) return false;
        file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(double));
        if(!file) return false;
        file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(double));
        if(!file) return false;
        return true;
    }
};

class DropoutLayer : public Layer {
private:
    double rate;
    Tensor mask;
    bool isTraining;
    std::mt19937 gen;

public:
    DropoutLayer(double r) : rate(r), isTraining(false) {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    void setMode(bool training) override { isTraining = training; }

    Tensor forward(const Tensor& input) override {
        if (!isTraining) return input;
        
        mask = Tensor(input.depth, input.rows, input.cols);
        Tensor output(input.depth, input.rows, input.cols);
        std::bernoulli_distribution d(1.0 - rate);

        for (size_t i = 0; i < input.data.size(); ++i) {
            if (d(gen)) {
                mask.data[i] = 1.0;
                output.data[i] = input.data[i] / (1.0 - rate);
            } else {
                mask.data[i] = 0.0;
                output.data[i] = 0.0;
            }
        }
        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (!isTraining) return gradOutput;
        Tensor gradInput = gradOutput;
        for (size_t i = 0; i < gradInput.data.size(); ++i) {
            gradInput.data[i] *= mask.data[i] / (1.0 - rate);
        }
        return gradInput;
    }

    void update(double lr, int t) override {}
};

class CNN {
public:
    Config::ModelConfig config; 
private:
    std::vector<std::unique_ptr<Layer>> layers;

    void buildLayers() {
        layers.clear();
        
        int c1InRows = config.inputTimeSteps;
        int c1InCols = config.inputFeatures;
        
        layers.push_back(std::make_unique<ConvLayer>(
            config.conv1Filters, config.conv1KernH, config.conv1KernW, 
            1, c1InRows, c1InCols, config.conv1Stride));
        
        int c1OutRows = (c1InRows - config.conv1KernH) / config.conv1Stride + 1;
        int c1OutCols = (c1InCols - config.conv1KernW) / config.conv1Stride + 1;
        
        layers.push_back(std::make_unique<LayerNormLayer>(config.conv1Filters));

        layers.push_back(std::make_unique<ConvLayer>(
            config.conv2Filters, config.conv2KernH, config.conv2KernW, 
            config.conv1Filters, c1OutRows, c1OutCols, config.conv2Stride));
        
        int c2OutRows = (c1OutRows - config.conv2KernH) / config.conv2Stride + 1;
        int c2OutCols = (c1OutCols - config.conv2KernW) / config.conv2Stride + 1;
        
        layers.push_back(std::make_unique<LayerNormLayer>(config.conv2Filters));

        int lstmInputSize = config.conv2Filters * c2OutCols;

        layers.push_back(std::make_unique<LSTMLayer>(lstmInputSize, config.lstmHiddenSize));
        layers.push_back(std::make_unique<AttentionLayer>(config.lstmHiddenSize));

        layers.push_back(std::make_unique<DenseLayer>(config.lstmHiddenSize, config.hiddenNeurons1, false));
        layers.push_back(std::make_unique<LayerNormLayer>(config.hiddenNeurons1));
        layers.push_back(std::make_unique<DropoutLayer>(Config::DROPOUT_RATE_1));
        
        layers.push_back(std::make_unique<DenseLayer>(config.hiddenNeurons1, config.hiddenNeurons2, false));
        layers.push_back(std::make_unique<LayerNormLayer>(config.hiddenNeurons2));
        layers.push_back(std::make_unique<DropoutLayer>(Config::DROPOUT_RATE_2));
        
        layers.push_back(std::make_unique<DenseLayer>(config.hiddenNeurons2, config.outputSize, true, true));
    }

public:
    CNN(Config::ModelConfig cfg) : config(cfg) {
        buildLayers();
    }
    
    CNN() {
        buildLayers();
    }

    double predict(const Tensor& input) {
        for(auto& l : layers) l->setMode(false);
        Tensor out = input;
        for(auto& l : layers) out = l->forward(out);
        return out.data[0];
    }

    double train(const Tensor& input, double target, double weight = 1.0) {
        for(auto& l : layers) l->setMode(true);
        Tensor out = input;
        for(auto& l : layers) out = l->forward(out);

        double prediction = out.data[0];
        double error = (prediction - target) * weight;
        
        Tensor grad(1, 1, 1);
        grad.data[0] = error;

        for(int i = layers.size() - 1; i >= 0; --i) grad = layers[i]->backward(grad);
        
        return prediction;
    }

    void averageGradients(int batchSize) {
        if(batchSize <= 0) return;
        double scale = 1.0 / batchSize;
        for(auto& l : layers) l->scaleGradients(scale);
    }

    void clearGradients() {
        for(auto& l : layers) l->clearGradients();
    }

    void updateParams(double lr, int t) {
        for(auto& l : layers) l->update(lr, t);
    }

    bool save(std::ofstream& file) {
        if(!file.is_open()) return false;
        
        file.write(reinterpret_cast<const char*>(&config.inputTimeSteps), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.inputFeatures), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.conv1Filters), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.conv1KernH), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.conv1KernW), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.conv1Stride), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.conv2Filters), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.conv2KernH), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.conv2KernW), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.conv2Stride), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.lstmHiddenSize), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.hiddenNeurons1), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.hiddenNeurons2), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.outputSize), sizeof(int));
        file.write(reinterpret_cast<const char*>(&config.targetCandles), sizeof(int));
        
        for(auto& l : layers) l->save(file);
        return true;
    }

    bool load(std::ifstream& file) {
        if(!file.is_open()) return false;
        
        file.read(reinterpret_cast<char*>(&config.inputTimeSteps), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.inputFeatures), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.conv1Filters), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.conv1KernH), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.conv1KernW), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.conv1Stride), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.conv2Filters), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.conv2KernH), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.conv2KernW), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.conv2Stride), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.lstmHiddenSize), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.hiddenNeurons1), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.hiddenNeurons2), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.outputSize), sizeof(int));
        file.read(reinterpret_cast<char*>(&config.targetCandles), sizeof(int));

        if(!file) return false;
        
        buildLayers();

        for(auto& l : layers) {
            if(!l->load(file)) return false;
        }
        return true;
    }
};

#endif