// app.js
// Main application logic - ties together data loading, model training, and prediction

import { DataLoader } from './data-loader.js';
import { LSTMModel } from './lstm-model.js';

class BikedemandApp {
  constructor() {
    this.dataLoader = null;
    this.model = null;
    this.sequenceLength = 24;
    this.epochs = 50;
    this.batchSize = 32;
    this.validationSplit = 0.2;
    this.predictions = null;
    this.chart = null;
  }

  /**
   * Initialize the application
   */
  init() {
    // Set up event listeners
    document.getElementById('file-input').addEventListener('change', (e) => this.handleFileUpload(e));
    document.getElementById('train-btn').addEventListener('click', () => this.startTraining());
    document.getElementById('predict-btn').addEventListener('click', () => this.makePrediction());
    
    // Initially disable buttons
    document.getElementById('train-btn').disabled = true;
    document.getElementById('predict-btn').disabled = true;
    
    this.updateStatus('Ready. Please upload the CSV file.');
  }

  /**
   * Handle CSV file upload
   */
  async handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    this.updateStatus('Loading and preprocessing data...');
    this.showProgress(true);
    
    try {
      // Create new data loader
      this.dataLoader = new DataLoader();
      
      // Load and preprocess CSV
      await this.dataLoader.loadCSV(file);
      this.dataLoader.preprocessData(this.sequenceLength, 0.8);
      
      this.updateStatus(`Data loaded successfully. ${this.dataLoader.sequences.length} sequences created.`);
      document.getElementById('train-btn').disabled = false;
      this.showProgress(false);
      
    } catch (error) {
      this.updateStatus(`Error loading data: ${error.message}`);
      this.showProgress(false);
    }
  }

  /**
   * Start model training
   */
  async startTraining() {
    if (!this.dataLoader) {
      alert('Please upload data first!');
      return;
    }

    this.updateStatus('Building and training model...');
    this.showProgress(true);
    document.getElementById('train-btn').disabled = true;
    document.getElementById('predict-btn').disabled = true;

    try {
      // Dispose old model if exists
      if (this.model) {
        this.model.dispose();
      }

      // Create new model
      const numFeatures = this.dataLoader.getNumFeatures();
      this.model = new LSTMModel(this.sequenceLength, numFeatures);
      
      // Get training data
      const { x: trainX, y: trainY } = this.dataLoader.getTrainData();

      // Set up training callbacks
      const callbacks = {
        onEpochEnd: (epoch, logs) => {
          const progress = ((epoch + 1) / this.epochs) * 100;
          document.getElementById('progress-bar').style.width = `${progress}%`;
          document.getElementById('progress-bar').textContent = `${Math.round(progress)}%`;
          
          const lossText = `Epoch ${epoch + 1}/${this.epochs} - Loss: ${logs.loss.toFixed(4)} - MAE: ${logs.mae.toFixed(4)}`;
          const valText = logs.val_loss ? ` - Val Loss: ${logs.val_loss.toFixed(4)} - Val MAE: ${logs.val_mae.toFixed(4)}` : '';
          this.updateStatus(lossText + valText);
        },
        onTrainEnd: () => {
          this.updateStatus('Training complete!');
          this.showProgress(false);
          document.getElementById('train-btn').disabled = false;
          document.getElementById('predict-btn').disabled = false;
        }
      };

      // Train model
      await this.model.train(
        trainX,
        trainY,
        this.epochs,
        this.batchSize,
        this.validationSplit,
        callbacks
      );

    } catch (error) {
      this.updateStatus(`Error during training: ${error.message}`);
      this.showProgress(false);
      document.getElementById('train-btn').disabled = false;
    }
  }

  /**
   * Make predictions and display results
   */
  async makePrediction() {
    if (!this.model) {
      alert('Please train the model first!');
      return;
    }

    this.updateStatus('Making predictions...');

    try {
      // Get test data
      const { x: testX, y: testY } = this.dataLoader.getTestData();
      
      // Make predictions
      const predictionsTensor = this.model.predict(testX);
      const predictionsArray = await predictionsTensor.array();
      predictionsTensor.dispose();

      // Denormalize predictions
      this.predictions = predictionsArray.map(pred =>
        pred.map(val => this.dataLoader.denormalize(val, 'rentedBikeCount'))
      );

      // Get actual values
      const actuals = this.dataLoader.getTestActuals();

      // Calculate metrics
      const mae = this.calculateMAE(actuals, this.predictions);
      const mape = this.calculateMAPE(actuals, this.predictions);

      this.updateStatus(`Prediction complete! MAE: ${mae.toFixed(2)}, MAPE: ${mape.toFixed(2)}%`);

      // Plot results
      this.plotResults(actuals, this.predictions);

    } catch (error) {
      this.updateStatus(`Error during prediction: ${error.message}`);
      console.error(error);
    }
  }

  /**
   * Calculate Mean Absolute Error
   */
  calculateMAE(actuals, predictions) {
    let totalError = 0;
    let count = 0;

    for (let i = 0; i < actuals.length; i++) {
      for (let j = 0; j < actuals[i].length; j++) {
        totalError += Math.abs(actuals[i][j] - predictions[i][j]);
        count++;
      }
    }

    return totalError / count;
  }

  /**
   * Calculate Mean Absolute Percentage Error
   */
  calculateMAPE(actuals, predictions) {
    let totalError = 0;
    let count = 0;

    for (let i = 0; i < actuals.length; i++) {
      for (let j = 0; j < actuals[i].length; j++) {
        if (actuals[i][j] !== 0) {
          totalError += Math.abs((actuals[i][j] - predictions[i][j]) / actuals[i][j]);
          count++;
        }
      }
    }

    return (totalError / count) * 100;
  }

  /**
   * Plot actual vs predicted values using Chart.js
   */
  plotResults(actuals, predictions) {
    // Flatten arrays for plotting
    const flatActuals = actuals.flat();
    const flatPredictions = predictions.flat();

    // Limit to first 500 points for better visualization
    const maxPoints = Math.min(500, flatActuals.length);
    const labels = Array.from({ length: maxPoints }, (_, i) => i);
    const actualData = flatActuals.slice(0, maxPoints);
    const predictionData = flatPredictions.slice(0, maxPoints);

    // Destroy previous chart if exists
    if (this.chart) {
      this.chart.destroy();
    }

    // Create new chart
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Actual Bike Count',
            data: actualData,
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
          },
          {
            label: 'Predicted Bike Count',
            data: predictionData,
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Actual vs Predicted Bike Rental Demand (Test Set - First 500 Hours)',
            font: {
              size: 16
            }
          },
          legend: {
            display: true,
            position: 'top'
          }
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'Hour Index'
            }
          },
          y: {
            display: true,
            title: {
              display: true,
              text: 'Bike Count'
            }
          }
        }
      }
    });
  }

  /**
   * Update status message
   */
  updateStatus(message) {
    document.getElementById('status').textContent = message;
  }

  /**
   * Show/hide progress bar
   */
  showProgress(show) {
    const progressContainer = document.getElementById('progress-container');
    progressContainer.style.display = show ? 'block' : 'none';
    
    if (!show) {
      document.getElementById('progress-bar').style.width = '0%';
      document.getElementById('progress-bar').textContent = '0%';
    }
  }

  /**
   * Clean up resources
   */
  dispose() {
    if (this.dataLoader) {
      this.dataLoader.dispose();
    }
    if (this.model) {
      this.model.dispose();
    }
    if (this.chart) {
      this.chart.destroy();
    }
  }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  const app = new BikedemandApp();
  app.init();
  
  // Store app instance globally for debugging
  window.bikeApp = app;
});

