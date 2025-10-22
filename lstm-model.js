// lstm-model.js
// Defines and trains the LSTM model using TensorFlow.js

export class LSTMModel {
  constructor(sequenceLength, numFeatures) {
    this.sequenceLength = sequenceLength;
    this.numFeatures = numFeatures;
    this.model = null;
    this.buildModel();
  }

  /**
   * Build the LSTM model architecture
   */
  buildModel() {
    this.model = tf.sequential();

    // First LSTM layer with 64 units, return sequences for next LSTM layer
    this.model.add(tf.layers.lstm({
      units: 64,
      returnSequences: true,
      inputShape: [this.sequenceLength, this.numFeatures],
      dropout: 0.2,
      recurrentDropout: 0.2
    }));

    // Second LSTM layer with 32 units
    this.model.add(tf.layers.lstm({
      units: 32,
      returnSequences: false,
      dropout: 0.2,
      recurrentDropout: 0.2
    }));

    // Dense output layer with 24 units (predicting next 24 hours)
    this.model.add(tf.layers.dense({
      units: this.sequenceLength,
      activation: 'linear'
    }));

    // Compile model with Adam optimizer and MSE loss
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mae']
    });
  }

  /**
   * Train the model
   */
  async train(trainX, trainY, epochs, batchSize, validationSplit, callbacks) {
    const history = await this.model.fit(trainX, trainY, {
      epochs: epochs,
      batchSize: batchSize,
      validationSplit: validationSplit,
      shuffle: false, // Keep chronological order
      callbacks: callbacks
    });

    return history;
  }

  /**
   * Make predictions
   */
  predict(testX) {
    return this.model.predict(testX);
  }

  /**
   * Evaluate the model on test data
   */
  evaluate(testX, testY) {
    const result = this.model.evaluate(testX, testY);
    return result;
  }

  /**
   * Get model summary
   */
  summary() {
    this.model.summary();
  }

  /**
   * Dispose model
   */
  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
}


