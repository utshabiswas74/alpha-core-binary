#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace Config {
    struct ModelConfig {
        int inputTimeSteps = 24;
        int inputFeatures = 16;

        int conv1Filters = 24;
        int conv1KernH = 5;
        int conv1KernW = 16;
        int conv1Stride = 1;

        int conv2Filters = 48;
        int conv2KernH = 5;
        int conv2KernW = 1;
        int conv2Stride = 1;

        int lstmHiddenSize = 48;
        int hiddenNeurons1 = 128;
        int hiddenNeurons2 = 128;
        int outputSize = 1;
    };
    
    const int EPOCHS = 128;
    const int BATCH_SIZE = 128;
    const double LEARNING_RATE = 0.0001;
    const double INPUT_NOISE = 0.01;
    
    const double DROPOUT_RATE_1 = 0.40;
    const double DROPOUT_RATE_2 = 0.40;
    
    const double BETA1 = 0.9;
    const double BETA2 = 0.999;
    const double EPSILON = 1e-8;

    const int IDX_RSI = 4;
    const int IDX_ADX = 12;
    const int IDX_PRICE_POS = 13;
    const int IDX_BB_PCT = 14;

    const int TARGET_CANDLES = 4;
    const int MAX_ENSEMBLE_MODELS = 6;
    const int TIMEFRAME_MULTIPLIER = 6;
    const double MIN_MOVEMENT_PCT = 0.01;
    
    const double KALMAN_PROCESS_NOISE = 1e-4;
    const double KALMAN_MEASUREMENT_NOISE = 1e-2;

    const std::string MODEL_FILE_BASE = "data/main_brain/model_brain";
    const std::string MODEL_FILE_EXTRA = "data/extra_brain/model_brain";
    const std::string TRAIN_DATA_FILE = "data/training_data.csv";
}

#endif