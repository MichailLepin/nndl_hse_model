/**
 * Модуль для загрузки и предобработки данных о велосипедах
 * Обрабатывает CSV файлы, нормализует данные и создает последовательности для LSTM
 */

export class DataLoader {
    constructor() {
        this.rawData = null;
        this.processedData = null;
        this.normalizationParams = {};
        this.sequences = null;
        this.labels = null;
        this.trainData = null;
        this.testData = null;
        this.featureNames = [];
    }

    /**
     * Загружает и парсит CSV файл
     * @param {File} file - CSV файл
     * @returns {Promise<Object>} - Парсированные данные
     */
    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    const lines = csv.split('\n');
                    
                    // Исправляем заголовки с проблемными символами
                    const rawHeaders = lines[0].split(',');
                    const headers = rawHeaders.map(h => {
                        const trimmed = h.trim();
                        // Заменяем проблемные символы на стандартные
                        return trimmed
                            .replace(/[^\x00-\x7F]/g, '') // Удаляем не-ASCII символы
                            .replace(/Temperature.*C/, 'Temperature(°C)')
                            .replace(/Humidity.*%/, 'Humidity(%)')
                            .replace(/Wind speed.*m\/s/, 'Wind speed (m/s)')
                            .replace(/Visibility.*10m/, 'Visibility (10m)')
                            .replace(/Dew point temperature.*C/, 'Dew point temperature(°C)')
                            .replace(/Solar Radiation.*MJ\/m2/, 'Solar Radiation (MJ/m2)')
                            .replace(/Rainfall.*mm/, 'Rainfall(mm)')
                            .replace(/Snowfall.*cm/, 'Snowfall (cm)');
                    });
                    
                    console.log('Оригинальные заголовки:', rawHeaders);
                    console.log('Исправленные заголовки:', headers);
                    
                    const data = [];
                    for (let i = 1; i < lines.length; i++) {
                        if (lines[i].trim()) {
                            const values = lines[i].split(',');
                            if (values.length === headers.length) {
                                const row = {};
                                headers.forEach((header, index) => {
                                    row[header] = values[index].trim();
                                });
                                data.push(row);
                            }
                        }
                    }
                    
                    this.rawData = data;
                    resolve(data);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Ошибка чтения файла'));
            reader.readAsText(file, 'UTF-8');
        });
    }

    /**
     * Предобрабатывает данные: конвертирует типы, кодирует категориальные переменные
     * @param {Array} data - Сырые данные
     * @returns {Array} - Обработанные данные
     */
    preprocessData(data) {
        console.log('Начинаем предобработку данных...');
        console.log('Первая строка данных:', data[0]);
        console.log('Количество строк:', data.length);
        
        const processed = data.map((row, index) => {
            if (index < 3) {
                console.log(`Строка ${index}:`, row);
            }
            const processedRow = {};
            
            // Конвертируем дату
            processedRow.date = new Date(row.Date);
            
            // Конвертируем числовые значения
            processedRow.hour = parseInt(row.Hour);
            processedRow.rented_bike_count = parseInt(row['Rented Bike Count']);
            processedRow.temperature = parseFloat(row['Temperature(°C)']);
            processedRow.humidity = parseFloat(row['Humidity(%)']);
            processedRow.wind_speed = parseFloat(row['Wind speed (m/s)']);
            processedRow.visibility = parseFloat(row['Visibility (10m)']);
            processedRow.dew_point_temperature = parseFloat(row['Dew point temperature(°C)']);
            processedRow.solar_radiation = parseFloat(row['Solar Radiation (MJ/m2)']);
            processedRow.rainfall = parseFloat(row['Rainfall(mm)']);
            processedRow.snowfall = parseFloat(row['Snowfall (cm)']);
            
            // Кодируем категориальные переменные
            processedRow.seasons = row.Seasons;
            processedRow.holiday = row.Holiday === 'Holiday' ? 1 : 0;
            processedRow.functioning_day = row['Functioning Day'] === 'Yes' ? 1 : 0;
            
            return processedRow;
        }).filter((row, index) => {
            // Check for NaN in all numeric features
            const isValid = !isNaN(row.rented_bike_count) &&
                   !isNaN(row.temperature) &&
                   !isNaN(row.humidity) &&
                   !isNaN(row.wind_speed) &&
                   !isNaN(row.visibility) &&
                   !isNaN(row.dew_point_temperature) &&
                   !isNaN(row.solar_radiation) &&
                   !isNaN(row.rainfall) &&
                   !isNaN(row.snowfall);
            
            if (index < 5 && !isValid) {
                console.log(`Строка ${index} отфильтрована:`, {
                    rented_bike_count: row.rented_bike_count,
                    temperature: row.temperature,
                    humidity: row.humidity,
                    wind_speed: row.wind_speed,
                    visibility: row.visibility,
                    dew_point_temperature: row.dew_point_temperature,
                    solar_radiation: row.solar_radiation,
                    rainfall: row.rainfall,
                    snowfall: row.snowfall
                });
            }
            
            return isValid;
        });

        // Сортируем по дате и времени
        processed.sort((a, b) => {
            if (a.date.getTime() === b.date.getTime()) {
                return a.hour - b.hour;
            }
            return a.date.getTime() - b.date.getTime();
        });

        console.log(`Обработано ${processed.length} записей`);
        this.processedData = processed;
        return processed;
    }

    /**
     * Создает циклические признаки для часа (синус и косинус)
     * @param {number} hour - Час (0-23)
     * @returns {Object} - Объект с циклическими признаками
     */
    createCyclicalFeatures(hour) {
        const hourRad = (hour * 2 * Math.PI) / 24;
        return {
            hour_sin: Math.sin(hourRad),
            hour_cos: Math.cos(hourRad)
        };
    }

    /**
     * One-hot кодирование для сезонов
     * @param {string} season - Сезон
     * @returns {Object} - Объект с one-hot кодированием
     */
    oneHotEncodeSeason(season) {
        const seasons = ['Spring', 'Summer', 'Autumn', 'Winter'];
        const encoded = {};
        seasons.forEach(s => {
            encoded[`season_${s.toLowerCase()}`] = s === season ? 1 : 0;
        });
        return encoded;
    }

    /**
     * Нормализует числовые признаки (min-max scaling)
     * @param {Array} data - Данные для нормализации
     * @returns {Array} - Нормализованные данные
     */
    normalizeFeatures(data) {
        console.log('Нормализация признаков...');
        
        const numericFeatures = [
            'temperature', 'humidity', 'wind_speed', 'visibility',
            'dew_point_temperature', 'solar_radiation', 'rainfall', 'snowfall'
        ];
        
        // Вычисляем min и max для каждого признака
        this.normalizationParams = {};
        numericFeatures.forEach(feature => {
            const values = data.map(row => row[feature]).filter(v => !isNaN(v));
            this.normalizationParams[feature] = {
                min: Math.min(...values),
                max: Math.max(...values)
            };
        });

        // Нормализуем данные
        const normalized = data.map(row => {
            const normalizedRow = { ...row };
            
            // Нормализуем числовые признаки
            numericFeatures.forEach(feature => {
                const params = this.normalizationParams[feature];
                const range = params.max - params.min;
                if (range === 0) {
                    normalizedRow[feature] = 0;
                } else {
                    const normalizedValue = (row[feature] - params.min) / range;
                    // Проверяем на NaN и Infinity
                    if (isNaN(normalizedValue) || !isFinite(normalizedValue)) {
                        console.warn(`Некорректное значение для признака ${feature}: ${row[feature]}, заменяем на 0`);
                        normalizedRow[feature] = 0;
                    } else {
                        normalizedRow[feature] = normalizedValue;
                    }
                }
            });
            
            // Добавляем циклические признаки для часа
            const cyclical = this.createCyclicalFeatures(row.hour);
            normalizedRow.hour_sin = cyclical.hour_sin;
            normalizedRow.hour_cos = cyclical.hour_cos;
            
            // Добавляем one-hot кодирование для сезонов
            const seasonEncoded = this.oneHotEncodeSeason(row.seasons);
            Object.assign(normalizedRow, seasonEncoded);
            
            return normalizedRow;
        });

        // Определяем имена всех признаков
        this.featureNames = [
            'temperature', 'humidity', 'wind_speed', 'visibility',
            'dew_point_temperature', 'solar_radiation', 'rainfall', 'snowfall',
            'hour_sin', 'hour_cos', 'holiday', 'functioning_day',
            'season_spring', 'season_summer', 'season_autumn', 'season_winter'
        ];

        console.log(`Создано ${this.featureNames.length} признаков`);
        return normalized;
    }

    /**
     * Создает последовательности для LSTM (24 часа входных данных -> 24 часа прогноза)
     * @param {Array} data - Нормализованные данные
     * @returns {Object} - Объект с последовательностями и метками
     */
    createSequences(data) {
        console.log('Создание последовательностей для LSTM...');
        
        const sequences = [];
        const labels = [];
        const sequenceLength = 24; // 24 часа входных данных
        const predictionLength = 24; // 24 часа прогноза
        
        for (let i = sequenceLength; i < data.length - predictionLength; i++) {
            // Создаем последовательность входных данных (24 часа)
            const sequence = [];
            for (let j = i - sequenceLength; j < i; j++) {
                const features = this.featureNames.map(name => {
                    const value = data[j][name];
                    // Проверяем на NaN и Infinity
                    if (isNaN(value) || !isFinite(value)) {
                        console.warn(`Некорректное значение в последовательности: ${name} = ${value}, заменяем на 0`);
                        return 0;
                    }
                    return value;
                });
                sequence.push(features);
            }
            
            // Создаем метки (следующие 24 часа)
            const label = [];
            for (let j = i; j < i + predictionLength; j++) {
                const value = data[j].rented_bike_count;
                if (isNaN(value) || !isFinite(value)) {
                    console.warn(`Некорректное значение метки: ${value}, заменяем на 0`);
                    label.push(0);
                } else {
                    label.push(value);
                }
            }
            
            sequences.push(sequence);
            labels.push(label);
        }
        
        console.log(`Создано ${sequences.length} последовательностей`);
        this.sequences = sequences;
        this.labels = labels;
        
        return { sequences, labels };
    }

    /**
     * Разделяет данные на обучающую и тестовую выборки
     * @param {Array} sequences - Последовательности
     * @param {Array} labels - Метки
     * @param {number} testRatio - Доля тестовых данных (по умолчанию 0.2)
     * @returns {Object} - Разделенные данные
     */
    splitData(sequences, labels, testRatio = 0.2) {
        console.log('Разделение данных на обучающую и тестовую выборки...');
        
        const splitIndex = Math.floor(sequences.length * (1 - testRatio));
        
        this.trainData = {
            sequences: sequences.slice(0, splitIndex),
            labels: labels.slice(0, splitIndex)
        };
        
        this.testData = {
            sequences: sequences.slice(splitIndex),
            labels: labels.slice(splitIndex)
        };
        
        console.log(`Обучающих последовательностей: ${this.trainData.sequences.length}`);
        console.log(`Тестовых последовательностей: ${this.testData.sequences.length}`);
        
        return { trainData: this.trainData, testData: this.testData };
    }

    /**
     * Конвертирует данные в тензоры TensorFlow.js
     * @param {Array} sequences - Последовательности
     * @param {Array} labels - Метки
     * @returns {Object} - Тензоры
     */
    convertToTensors(sequences, labels) {
        console.log('Конвертация в тензоры TensorFlow.js...');
        
        const X = tf.tensor3d(sequences);
        const y = tf.tensor2d(labels);
        
        console.log(`Форма входных данных: ${X.shape}`);
        console.log(`Форма меток: ${y.shape}`);
        
        return { X, y };
    }

    /**
     * Полный процесс загрузки и предобработки данных
     * @param {File} file - CSV файл
     * @returns {Promise<Object>} - Обработанные данные
     */
    async loadAndPreprocess(file) {
        try {
            console.log('Начинаем загрузку и предобработку данных...');
            
            // Загружаем CSV
            const rawData = await this.loadCSV(file);
            console.log(`Загружено ${rawData.length} записей`);
            
            // Предобрабатываем данные
            const processedData = this.preprocessData(rawData);
            
            // Нормализуем признаки
            const normalizedData = this.normalizeFeatures(processedData);
            
            // Создаем последовательности
            const { sequences, labels } = this.createSequences(normalizedData);
            
            // Разделяем на train/test
            const { trainData, testData } = this.splitData(sequences, labels);
            
            console.log('Предобработка данных завершена успешно!');
            
            return {
                trainData,
                testData,
                featureNames: this.featureNames,
                normalizationParams: this.normalizationParams,
                rawData: this.processedData
            };
            
        } catch (error) {
            console.error('Ошибка при загрузке и предобработке данных:', error);
            throw error;
        }
    }

    /**
     * Очищает память от тензоров
     */
    dispose() {
        if (this.sequences) {
            this.sequences = null;
        }
        if (this.labels) {
            this.labels = null;
        }
        if (this.trainData) {
            this.trainData = null;
        }
        if (this.testData) {
            this.testData = null;
        }
    }
}
