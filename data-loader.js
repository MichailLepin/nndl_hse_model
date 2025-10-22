// data-loader.js
// Handles CSV parsing, preprocessing, sequence creation, and train/test split

export class DataLoader {
  constructor() {
    this.rawData = [];
    this.processedData = [];
    this.sequences = [];
    this.labels = [];
    this.trainX = null;
    this.trainY = null;
    this.testX = null;
    this.testY = null;
    this.scaleParams = {};
    this.numFeatures = 0;
    this.testActuals = [];
  }

  /**
   * Parse CSV file from uploaded file input
   */
  async loadCSV(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        this.parseCSV(text);
        resolve();
      };
      reader.onerror = reject;
      reader.readAsText(file);
    });
  }

  /**
   * Parse CSV text into raw data array
   */
  parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',');
    
    this.rawData = [];
    for (let i = 1; i < lines.length; i++) {
      const values = this.parseCSVLine(lines[i]);
      if (values.length === headers.length) {
        const row = {};
        headers.forEach((header, idx) => {
          row[header.trim()] = values[idx].trim();
        });
        this.rawData.push(row);
      }
    }
  }

  /**
   * Parse a single CSV line, handling quoted values
   */
  parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        result.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    result.push(current);
    return result;
  }

  /**
   * Preprocess the data: encode features, normalize, create sequences
   */
  preprocessData(sequenceLength = 24, trainSplit = 0.8) {
    // Convert and encode features
    this.processedData = this.rawData.map(row => {
      // Parse date
      const dateParts = row['Date'].split('/');
      const date = new Date(`${dateParts[2]}-${dateParts[1]}-${dateParts[0]}`);
      
      // Parse numeric values
      const hour = parseInt(row['Hour']);
      const rentedBikeCount = parseFloat(row['Rented Bike Count']);
      const temperature = parseFloat(row['Temperature(°C)']);
      const humidity = parseFloat(row['Humidity(%)']);
      const windSpeed = parseFloat(row['Wind speed (m/s)']);
      const visibility = parseFloat(row['Visibility (10m)']);
      const dewPoint = parseFloat(row['Dew point temperature(°C)']);
      const solarRadiation = parseFloat(row['Solar Radiation (MJ/m2)']);
      const rainfall = parseFloat(row['Rainfall(mm)']);
      const snowfall = parseFloat(row['Snowfall (cm)']);
      
      // Cyclical encoding for hour
      const hourSin = Math.sin(2 * Math.PI * hour / 24);
      const hourCos = Math.cos(2 * Math.PI * hour / 24);
      
      // One-hot encode seasons
      const seasons = row['Seasons'].trim();
      const seasonSpring = seasons === 'Spring' ? 1 : 0;
      const seasonSummer = seasons === 'Summer' ? 1 : 0;
      const seasonAutumn = seasons === 'Autumn' ? 1 : 0;
      const seasonWinter = seasons === 'Winter' ? 1 : 0;
      
      // One-hot encode holiday
      const holiday = row['Holiday'].trim() === 'Holiday' ? 1 : 0;
      
      // One-hot encode functioning day
      const functioningDay = row['Functioning Day'].trim() === 'Yes' ? 1 : 0;
      
      return {
        date,
        rentedBikeCount,
        hourSin,
        hourCos,
        temperature,
        humidity,
        windSpeed,
        visibility,
        dewPoint,
        solarRadiation,
        rainfall,
        snowfall,
        seasonSpring,
        seasonSummer,
        seasonAutumn,
        seasonWinter,
        holiday,
        functioningDay
      };
    });

    // Normalize numeric features
    this.normalizeFeatures();

    // Create sequences
    this.createSequences(sequenceLength);

    // Split into train/test
    this.splitData(trainSplit);
  }

  /**
   * Normalize features using min-max scaling
   */
  normalizeFeatures() {
    const featuresToNormalize = [
      'temperature', 'humidity', 'windSpeed', 'visibility',
      'dewPoint', 'solarRadiation', 'rainfall', 'snowfall', 'rentedBikeCount'
    ];

    featuresToNormalize.forEach(feature => {
      const values = this.processedData.map(row => row[feature]);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min || 1; // Avoid division by zero

      this.scaleParams[feature] = { min, max, range };

      this.processedData.forEach(row => {
        row[feature] = (row[feature] - min) / range;
      });
    });
  }

  /**
   * Create sequences of length sequenceLength for LSTM input
   */
  createSequences(sequenceLength) {
    this.sequences = [];
    this.labels = [];
    this.testActuals = [];

    const featureKeys = [
      'hourSin', 'hourCos', 'temperature', 'humidity', 'windSpeed',
      'visibility', 'dewPoint', 'solarRadiation', 'rainfall', 'snowfall',
      'seasonSpring', 'seasonSummer', 'seasonAutumn', 'seasonWinter',
      'holiday', 'functioningDay', 'rentedBikeCount'
    ];

    this.numFeatures = featureKeys.length;

    // Create sequences: use previous 24 hours to predict next 24 hours
    for (let i = sequenceLength; i < this.processedData.length - sequenceLength; i++) {
      const sequence = [];
      
      // Get previous 24 hours
      for (let j = i - sequenceLength; j < i; j++) {
        const features = featureKeys.map(key => this.processedData[j][key]);
        sequence.push(features);
      }

      // Label: next 24 hours of rented bike count
      const label = [];
      for (let j = i; j < i + sequenceLength; j++) {
        label.push(this.processedData[j]['rentedBikeCount']);
      }

      this.sequences.push(sequence);
      this.labels.push(label);
    }
  }

  /**
   * Split data chronologically into train and test sets
   */
  splitData(trainSplit) {
    const splitIndex = Math.floor(this.sequences.length * trainSplit);

    const trainSequences = this.sequences.slice(0, splitIndex);
    const trainLabels = this.labels.slice(0, splitIndex);
    const testSequences = this.sequences.slice(splitIndex);
    const testLabels = this.labels.slice(splitIndex);

    // Convert to tensors
    this.trainX = tf.tensor3d(trainSequences);
    this.trainY = tf.tensor2d(trainLabels);
    this.testX = tf.tensor3d(testSequences);
    this.testY = tf.tensor2d(testLabels);

    // Store test actuals for plotting (denormalized)
    this.testActuals = testLabels.map(label => 
      label.map(val => this.denormalize(val, 'rentedBikeCount'))
    );
  }

  /**
   * Denormalize a value
   */
  denormalize(normalizedValue, feature) {
    const { min, range } = this.scaleParams[feature];
    return normalizedValue * range + min;
  }

  /**
   * Get number of features
   */
  getNumFeatures() {
    return this.numFeatures;
  }

  /**
   * Get training data
   */
  getTrainData() {
    return { x: this.trainX, y: this.trainY };
  }

  /**
   * Get test data
   */
  getTestData() {
    return { x: this.testX, y: this.testY };
  }

  /**
   * Get test actuals (denormalized)
   */
  getTestActuals() {
    return this.testActuals;
  }

  /**
   * Clean up tensors
   */
  dispose() {
    if (this.trainX) this.trainX.dispose();
    if (this.trainY) this.trainY.dispose();
    if (this.testX) this.testX.dispose();
    if (this.testY) this.testY.dispose();
  }
}


