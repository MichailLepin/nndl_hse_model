/**
 * Основное приложение для прогнозирования спроса на велосипеды
 * Связывает все компоненты: загрузку данных, модель и интерфейс
 */

import { DataLoader } from './data-loader.js';
import { LSTMModel } from './lstm-model.js';

export class BikeDemandApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new LSTMModel();
        this.isDataLoaded = false;
        this.isModelTrained = false;
        this.processedData = null;
        this.charts = {};
        
        // Элементы DOM
        this.elements = {};
    }

    /**
     * Инициализирует приложение и настраивает обработчики событий
     */
    initialize() {
        console.log('Инициализация приложения...');
        
        // Получаем ссылки на элементы DOM
        this.elements = {
            csvFile: document.getElementById('csvFile'),
            loadDataBtn: document.getElementById('loadDataBtn'),
            trainBtn: document.getElementById('trainBtn'),
            predictBtn: document.getElementById('predictBtn'),
            resetBtn: document.getElementById('resetBtn'),
            status: document.getElementById('status'),
            progress: document.getElementById('progress'),
            progressText: document.getElementById('progressText'),
            metrics: document.getElementById('metrics'),
            lossValue: document.getElementById('lossValue'),
            maeValue: document.getElementById('maeValue'),
            mapeValue: document.getElementById('mapeValue'),
            predictionChart: document.getElementById('predictionChart'),
            trainingChart: document.getElementById('trainingChart')
        };

        // Настраиваем обработчики событий
        this.setupEventListeners();
        
        // Инициализируем графики
        this.initializeCharts();
        
        this.showStatus('Application ready. Please load a CSV file with data.', 'info');
        console.log('Application initialized');
    }

    /**
     * Настраивает обработчики событий для элементов интерфейса
     */
    setupEventListeners() {
        // Загрузка данных
        this.elements.loadDataBtn.addEventListener('click', () => this.loadData());
        
        // Обучение модели
        this.elements.trainBtn.addEventListener('click', () => this.trainModel());
        
        // Прогнозирование
        this.elements.predictBtn.addEventListener('click', () => this.makePrediction());
        
        // Сброс
        this.elements.resetBtn.addEventListener('click', () => this.reset());
        
        // Обработка выбора файла
        this.elements.csvFile.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.elements.loadDataBtn.disabled = false;
                this.showStatus(`File selected: ${e.target.files[0].name}`, 'info');
            }
        });
    }

    /**
     * Инициализирует графики Chart.js
     */
    initializeCharts() {
        // График прогнозов
        this.charts.prediction = new Chart(this.elements.predictionChart, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Actual Values',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }, {
                    label: 'Prediction',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Bikes'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Bike Demand Forecast'
                    }
                }
            }
        });

        // График обучения
        this.charts.training = new Chart(this.elements.trainingChart, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss (MSE)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    yAxisID: 'y'
                }, {
                    label: 'MAE',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Loss (MSE)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'MAE'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Training Process'
                    }
                }
            }
        });
    }

    /**
     * Загружает и предобрабатывает данные из CSV файла
     */
    async loadData() {
        try {
            this.showStatus('Loading and preprocessing data...', 'info');
            this.elements.loadDataBtn.disabled = true;
            
            const file = this.elements.csvFile.files[0];
            if (!file) {
                throw new Error('No file selected');
            }

            // Загружаем и предобрабатываем данные
            this.processedData = await this.dataLoader.loadAndPreprocess(file);
            
            this.isDataLoaded = true;
            this.elements.trainBtn.disabled = false;
            
            this.showStatus(`Data loaded successfully! Training records: ${this.processedData.trainData.sequences.length}, test records: ${this.processedData.testData.sequences.length}`, 'success');
            
            console.log('Data loaded:', this.processedData);
            
        } catch (error) {
            console.error('Error loading data:', error);
            this.showStatus(`Error loading data: ${error.message}`, 'error');
            this.elements.loadDataBtn.disabled = false;
        }
    }

    /**
     * Обучает LSTM модель
     */
    async trainModel() {
        if (!this.isDataLoaded) {
            this.showStatus('Please load data first', 'error');
            return;
        }

        try {
            this.showStatus('Starting model training...', 'info');
            this.elements.trainBtn.disabled = true;
            
            // Конвертируем данные в тензоры
            const trainX = tf.tensor3d(this.processedData.trainData.sequences);
            const trainY = tf.tensor2d(this.processedData.trainData.labels);
            const testX = tf.tensor3d(this.processedData.testData.sequences);
            const testY = tf.tensor2d(this.processedData.testData.labels);

            // Создаем модель
            const sequenceLength = 24;
            const numFeatures = this.processedData.featureNames.length;
            
            this.model.createModel(sequenceLength, numFeatures);
            this.model.compileModel(0.001);

            // Обучаем модель
            await this.model.train(trainX, trainY, testX, testY, {
                epochs: 30,
                batchSize: 32,
                onEpochEnd: (epochData) => {
                    this.updateTrainingChart(epochData);
                },
                onProgress: (progress) => {
                    this.updateProgress(progress);
                }
            });

            this.isModelTrained = true;
            this.elements.predictBtn.disabled = false;
            
            this.showStatus('Training completed successfully!', 'success');
            this.updateProgress(100);
            
            // Очищаем память
            trainX.dispose();
            trainY.dispose();
            testX.dispose();
            testY.dispose();
            
        } catch (error) {
            console.error('Error training model:', error);
            this.showStatus(`Training error: ${error.message}`, 'error');
            this.elements.trainBtn.disabled = false;
        }
    }

    /**
     * Делает прогноз и отображает результаты
     */
    async makePrediction() {
        if (!this.isModelTrained) {
            this.showStatus('Please train the model first', 'error');
            return;
        }

        try {
            this.showStatus('Making prediction...', 'info');
            this.elements.predictBtn.disabled = true;

            // Используем тестовые данные для прогноза
            const testX = tf.tensor3d(this.processedData.testData.sequences);
            const testY = tf.tensor2d(this.processedData.testData.labels);

            // Делаем прогноз
            const predictions = await this.model.predict(testX);
            
            // Вычисляем метрики
            const metrics = await this.model.getMetrics(testY, predictions);
            
            // Обновляем метрики в интерфейсе
            this.updateMetrics(metrics);
            
            // Подготавливаем данные для графика
            const actualData = await testY.data();
            const predictedData = await predictions.data();
            
            // Создаем метки времени (упрощенно)
            const timeLabels = Array.from({length: Math.min(actualData.length, 100)}, (_, i) => `Hour ${i + 1}`);
            
            // Обновляем график прогнозов
            this.updatePredictionChart(timeLabels, actualData, predictedData);
            
            this.showStatus('Prediction completed successfully!', 'success');
            
            // Очищаем память
            testX.dispose();
            testY.dispose();
            predictions.dispose();
            
        } catch (error) {
            console.error('Error making prediction:', error);
            this.showStatus(`Prediction error: ${error.message}`, 'error');
        } finally {
            this.elements.predictBtn.disabled = false;
        }
    }

    /**
     * Обновляет график обучения
     * @param {Object} epochData - Данные эпохи
     */
    updateTrainingChart(epochData) {
        if (epochData && 
            typeof epochData.epoch === 'number' && 
            typeof epochData.loss === 'number' && 
            !isNaN(epochData.loss) &&
            typeof epochData.mae === 'number' && 
            !isNaN(epochData.mae)) {
            
            this.charts.training.data.labels.push(epochData.epoch);
            this.charts.training.data.datasets[0].data.push(epochData.loss);
            this.charts.training.data.datasets[1].data.push(epochData.mae);
            this.charts.training.update();
        } else {
            console.warn('Пропускаем обновление графика из-за некорректных данных эпохи:', epochData);
        }
    }

    /**
     * Обновляет график прогнозов
     * @param {Array} labels - Метки времени
     * @param {Array} actual - Фактические значения
     * @param {Array} predicted - Прогнозируемые значения
     */
    updatePredictionChart(labels, actual, predicted) {
        // Ограничиваем количество точек для лучшей визуализации
        const maxPoints = 100;
        const step = Math.max(1, Math.floor(actual.length / maxPoints));
        
        const sampledLabels = labels.filter((_, i) => i % step === 0);
        const sampledActual = actual.filter((_, i) => i % step === 0);
        const sampledPredicted = predicted.filter((_, i) => i % step === 0);
        
        this.charts.prediction.data.labels = sampledLabels;
        this.charts.prediction.data.datasets[0].data = sampledActual;
        this.charts.prediction.data.datasets[1].data = sampledPredicted;
        this.charts.prediction.update();
    }

    /**
     * Обновляет метрики в интерфейсе
     * @param {Object} metrics - Вычисленные метрики
     */
    updateMetrics(metrics) {
        this.elements.lossValue.textContent = metrics.mse[0].toFixed(4);
        this.elements.maeValue.textContent = metrics.mae[0].toFixed(2);
        this.elements.mapeValue.textContent = metrics.mape[0].toFixed(2);
        this.elements.metrics.classList.remove('hidden');
    }

    /**
     * Обновляет прогресс-бар
     * @param {number} progress - Процент выполнения (0-100)
     */
    updateProgress(progress) {
        this.elements.progress.style.width = `${progress}%`;
        this.elements.progressText.textContent = `${Math.round(progress)}%`;
    }

    /**
     * Показывает статусное сообщение
     * @param {string} message - Сообщение
     * @param {string} type - Тип сообщения (info, success, error)
     */
    showStatus(message, type = 'info') {
        this.elements.status.textContent = message;
        this.elements.status.className = `status ${type}`;
        this.elements.status.classList.remove('hidden');
    }

    /**
     * Сбрасывает приложение в исходное состояние
     */
    reset() {
        console.log('Сброс приложения...');
        
        // Очищаем данные
        this.dataLoader.dispose();
        this.model.dispose();
        
        // Сбрасываем состояние
        this.isDataLoaded = false;
        this.isModelTrained = false;
        this.processedData = null;
        
        // Сбрасываем интерфейс
        this.elements.csvFile.value = '';
        this.elements.loadDataBtn.disabled = true;
        this.elements.trainBtn.disabled = true;
        this.elements.predictBtn.disabled = true;
        
        // Очищаем графики
        this.charts.prediction.data.labels = [];
        this.charts.prediction.data.datasets[0].data = [];
        this.charts.prediction.data.datasets[1].data = [];
        this.charts.prediction.update();
        
        this.charts.training.data.labels = [];
        this.charts.training.data.datasets[0].data = [];
        this.charts.training.data.datasets[1].data = [];
        this.charts.training.update();
        
        // Скрываем метрики
        this.elements.metrics.classList.add('hidden');
        
        // Сбрасываем прогресс
        this.updateProgress(0);
        
        // Показываем статус
        this.showStatus('Application reset. Load a new file to start working.', 'info');
        
        console.log('Application reset');
    }
}
