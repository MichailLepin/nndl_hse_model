/**
 * LSTM модель для прогнозирования спроса на велосипеды
 * Использует TensorFlow.js для создания и обучения нейронной сети
 */

export class LSTMModel {
    constructor() {
        this.model = null;
        this.isCompiled = false;
        this.trainingHistory = [];
    }

    /**
     * Создает архитектуру LSTM модели
     * @param {number} sequenceLength - Длина входной последовательности (24 часа)
     * @param {number} numFeatures - Количество признаков
     * @returns {tf.LayersModel} - Созданная модель
     */
    createModel(sequenceLength, numFeatures) {
        console.log(`Создание LSTM модели: ${sequenceLength} временных шагов, ${numFeatures} признаков`);
        
        this.model = tf.sequential({
            layers: [
                // Первый LSTM слой с dropout
                tf.layers.lstm({
                    units: 64,
                    returnSequences: true,
                    inputShape: [sequenceLength, numFeatures],
                    dropout: 0.2,
                    recurrentDropout: 0.2
                }),
                
                // Второй LSTM слой
                tf.layers.lstm({
                    units: 32,
                    returnSequences: false,
                    dropout: 0.2,
                    recurrentDropout: 0.2
                }),
                
                // Dense слой для выхода
                tf.layers.dense({
                    units: 24, // 24 часа прогноза
                    activation: 'linear'
                })
            ]
        });

        console.log('Архитектура модели создана');
        return this.model;
    }

    /**
     * Компилирует модель с оптимизатором и функцией потерь
     * @param {number} learningRate - Скорость обучения
     */
    compileModel(learningRate = 0.001) {
        if (!this.model) {
            throw new Error('Модель не создана. Сначала вызовите createModel()');
        }

        console.log('Компиляция модели...');
        
        this.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'meanSquaredError',
            metrics: ['mae'] // Mean Absolute Error
        });

        this.isCompiled = true;
        console.log('Модель скомпилирована успешно');
    }

    /**
     * Обучает модель на предоставленных данных
     * @param {tf.Tensor} trainX - Обучающие входные данные
     * @param {tf.Tensor} trainY - Обучающие метки
     * @param {tf.Tensor} valX - Валидационные входные данные
     * @param {tf.Tensor} valY - Валидационные метки
     * @param {Object} options - Параметры обучения
     * @returns {Promise<Object>} - История обучения
     */
    async train(trainX, trainY, valX, valY, options = {}) {
        if (!this.isCompiled) {
            throw new Error('Модель не скомпилирована. Сначала вызовите compileModel()');
        }

        const {
            epochs = 50,
            batchSize = 32,
            validationSplit = 0.2,
            onEpochEnd = null,
            onProgress = null
        } = options;

        console.log(`Начинаем обучение: ${epochs} эпох, размер батча ${batchSize}`);

        this.trainingHistory = [];

        try {
            const history = await this.model.fit(trainX, trainY, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: valX && valY ? [valX, valY] : undefined,
                validationSplit: valX ? undefined : validationSplit,
                shuffle: false, // Не перемешиваем временные ряды
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        const epochData = {
                            epoch: epoch + 1,
                            loss: logs.loss,
                            valLoss: logs.val_loss,
                            mae: logs.mae,
                            valMae: logs.val_mae
                        };
                        
                        this.trainingHistory.push(epochData);
                        
                        console.log(`Эпоха ${epoch + 1}/${epochs}: loss=${logs.loss.toFixed(4)}, val_loss=${logs.val_loss?.toFixed(4)}, mae=${logs.mae?.toFixed(4)}`);
                        
                        if (onEpochEnd) {
                            onEpochEnd(epochData);
                        }
                        
                        if (onProgress) {
                            const progress = ((epoch + 1) / epochs) * 100;
                            onProgress(progress);
                        }
                    }
                }
            });

            console.log('Обучение завершено успешно');
            return history;

        } catch (error) {
            console.error('Ошибка при обучении модели:', error);
            throw error;
        }
    }

    /**
     * Делает прогноз на основе входных данных
     * @param {tf.Tensor} inputData - Входные данные для прогноза
     * @returns {tf.Tensor} - Прогноз
     */
    async predict(inputData) {
        if (!this.model) {
            throw new Error('Модель не создана');
        }

        if (!this.isCompiled) {
            throw new Error('Модель не скомпилирована');
        }

        try {
            console.log('Выполняем прогноз...');
            const prediction = this.model.predict(inputData);
            return prediction;
        } catch (error) {
            console.error('Ошибка при прогнозировании:', error);
            throw error;
        }
    }

    /**
     * Вычисляет метрики качества модели
     * @param {tf.Tensor} yTrue - Истинные значения
     * @param {tf.Tensor} yPred - Предсказанные значения
     * @returns {Object} - Метрики
     */
    calculateMetrics(yTrue, yPred) {
        console.log('Вычисляем метрики качества...');
        
        // Mean Absolute Error (MAE)
        const mae = tf.losses.absoluteDifference(yTrue, yPred).mean();
        
        // Mean Squared Error (MSE)
        const mse = tf.losses.meanSquaredError(yTrue, yPred).mean();
        
        // Root Mean Squared Error (RMSE)
        const rmse = tf.sqrt(mse);
        
        // Mean Absolute Percentage Error (MAPE)
        const epsilon = 1e-8; // Малое значение для избежания деления на ноль
        const mape = tf.div(
            tf.abs(tf.sub(yTrue, yPred)),
            tf.add(tf.abs(yTrue), epsilon)
        ).mean().mul(100);

        return {
            mae: mae,
            mse: mse,
            rmse: rmse,
            mape: mape
        };
    }

    /**
     * Получает метрики как числа (синхронно)
     * @param {tf.Tensor} yTrue - Истинные значения
     * @param {tf.Tensor} yPred - Предсказанные значения
     * @returns {Promise<Object>} - Метрики как числа
     */
    async getMetrics(yTrue, yPred) {
        const metrics = this.calculateMetrics(yTrue, yPred);
        
        const result = {
            mae: await metrics.mae.data(),
            mse: await metrics.mse.data(),
            rmse: await metrics.rmse.data(),
            mape: await metrics.mape.data()
        };

        // Очищаем промежуточные тензоры
        Object.values(metrics).forEach(tensor => tensor.dispose());

        return result;
    }

    /**
     * Сохраняет модель в IndexedDB
     * @param {string} name - Имя модели
     * @returns {Promise<void>}
     */
    async saveModel(name = 'bike-demand-lstm') {
        if (!this.model) {
            throw new Error('Модель не создана');
        }

        try {
            console.log(`Сохранение модели: ${name}`);
            await this.model.save(`indexeddb://${name}`);
            console.log('Модель сохранена успешно');
        } catch (error) {
            console.error('Ошибка при сохранении модели:', error);
            throw error;
        }
    }

    /**
     * Загружает модель из IndexedDB
     * @param {string} name - Имя модели
     * @returns {Promise<tf.LayersModel>}
     */
    async loadModel(name = 'bike-demand-lstm') {
        try {
            console.log(`Загрузка модели: ${name}`);
            this.model = await tf.loadLayersModel(`indexeddb://${name}`);
            this.isCompiled = true;
            console.log('Модель загружена успешно');
            return this.model;
        } catch (error) {
            console.error('Ошибка при загрузке модели:', error);
            throw error;
        }
    }

    /**
     * Получает информацию о модели
     * @returns {Object} - Информация о модели
     */
    getModelInfo() {
        if (!this.model) {
            return null;
        }

        return {
            inputShape: this.model.inputShape,
            outputShape: this.model.outputShape,
            totalParams: this.model.countParams(),
            layers: this.model.layers.length,
            isCompiled: this.isCompiled
        };
    }

    /**
     * Получает историю обучения
     * @returns {Array} - История обучения
     */
    getTrainingHistory() {
        return this.trainingHistory;
    }

    /**
     * Очищает память от модели и тензоров
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.isCompiled = false;
        this.trainingHistory = [];
    }
}
