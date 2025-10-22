// app.js
// Orchestrates UI interactions, data loading, training, evaluation, and visualization.

import { DataLoader } from "./data-loader.js";
import { createLstmModel, trainModel, predictHorizon } from "./lstm-model.js";

class BikeDemandApp {
  constructor() {
    // UI elements
    this.fileInput = document.getElementById("csvFile");
    this.trainBtn = document.getElementById("trainBtn");
    this.predictBtn = document.getElementById("predictBtn");
    this.progressBar = document.getElementById("progress");
    this.statusEl = document.getElementById("status");
    this.metricsEl = document.getElementById("metrics");

    this.trainChartCtx = document.getElementById("trainChart").getContext("2d");
    this.testChartCtx = document.getElementById("testChart").getContext("2d");

    this.trainChart = null;
    this.testChart = null;

    // State
    this.loader = new DataLoader();
    this.dataset = null;
    this.model = null;

    // Bind events
    this.fileInput.addEventListener("change", (e) => this.onFileSelected(e));
    this.trainBtn.addEventListener("click", () => this.onTrain());
    this.predictBtn.addEventListener("click", () => this.onPredict());

    // Initial UI state
    this.setProgress(0);
    this.disableActions(true);
  }

  setProgress(pct) {
    this.progressBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    this.progressBar.setAttribute("aria-valuenow", String(Math.round(pct)));
  }

  setStatus(text) {
    this.statusEl.textContent = text;
  }

  setMetrics(text) {
    this.metricsEl.textContent = text;
  }

  disableActions(disabled) {
    this.trainBtn.disabled = disabled;
    this.predictBtn.disabled = disabled;
  }

  async onFileSelected(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    // Parse and preprocess
    this.setStatus("Parsing CSV…");
    this.setProgress(5);

    try {
      const dataObj = await this.loader.loadAndPrepare(file);
      // Dispose old tensors if any
      if (this.dataset) {
        this.disposeDataset(this.dataset);
      }
      this.dataset = dataObj;

      this.setProgress(50);
      this.setStatus(
        `Loaded. Features: ${dataObj.numFeatures}, Train samples: ${dataObj.xTrain.shape[0]}, Test samples: ${dataObj.xTest.shape[0]}`
      );
      this.disableActions(false);
    } catch (err) {
      console.error(err);
      this.setStatus(`Error: ${err.message}`);
      this.disableActions(true);
    } finally {
      this.setProgress(0);
    }
  }

  disposeDataset(ds) {
    try {
      ds.xTrain?.dispose?.();
      ds.yTrain?.dispose?.();
      ds.xTest?.dispose?.();
      ds.yTest?.dispose?.();
    } catch (_) {}
  }

  async onTrain() {
    if (!this.dataset) {
      this.setStatus("Please upload a CSV first.");
      return;
    }
    // Create or re-create model
    if (this.model) {
      this.model.dispose();
      this.model = null;
      await tf.nextFrame();
    }
    this.model = createLstmModel(this.dataset.numFeatures);
    this.setStatus("Training…");
    this.setProgress(1);
    this.disableActions(true);

    // Training chart (loss over epochs)
    const lossData = [];
    const maeData = [];
    if (this.trainChart) this.trainChart.destroy();
    this.trainChart = new Chart(this.trainChartCtx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          { label: "Loss (MSE)", data: [], tension: 0.2 },
          { label: "MAE", data: [], tension: 0.2 },
        ],
      },
      options: {
        responsive: true,
        animation: false,
        scales: {
          x: { title: { display: true, text: "Epoch" } },
          y: { title: { display: true, text: "Value" } },
        },
      },
    });

    const epochs = Number(document.getElementById("epochs").value) || 20;
    const batchSize = Number(document.getElementById("batchSize").value) || 64;

    try {
      await trainModel(this.model, this.dataset.xTrain, this.dataset.yTrain, {
        epochs,
        batchSize,
        onEpoch: (epoch, logs) => {
          const loss = Number.isFinite(logs.loss) ? logs.loss : NaN;
          const mae = Number.isFinite(logs.mae) ? logs.mae : NaN;
          lossData.push(loss);
          maeData.push(mae);
          this.trainChart.data.labels.push(epoch + 1);
          this.trainChart.data.datasets[0].data = lossData;
          this.trainChart.data.datasets[1].data = maeData;
          this.trainChart.update();

          const lossStr = Number.isFinite(loss) ? loss.toFixed(5) : 'NaN';
          const maeStr = Number.isFinite(mae) ? mae.toFixed(5) : 'NaN';
          this.setMetrics(`Epoch ${epoch + 1}/${epochs} — loss: ${lossStr} | MAE: ${maeStr}`);
          this.setProgress(5 + Math.min(90, ((epoch + 1) / epochs) * 90));
        },
      });
      this.setStatus("Training complete.");
    } catch (err) {
      console.error(err);
      this.setStatus(`Training error: ${err.message}`);
    } finally {
      this.disableActions(false);
      this.setProgress(0);
      await tf.nextFrame();
    }
  }

  async onPredict() {
    if (!this.dataset || !this.model) {
      this.setStatus("Please upload data and train the model first.");
      return;
    }
    this.setStatus("Predicting on test set…");
    this.setProgress(10);
    this.disableActions(true);

    let yPred = null;
    try {
      yPred = predictHorizon(this.model, this.dataset.xTest); // Tensor2D [N, 24], normalized
      const yTrue = this.dataset.yTest; // Tensor2D [N, 24], normalized

      // Convert to arrays
      const [predArr, trueArr] = await Promise.all([yPred.array(), yTrue.array()]);
      // Flatten both and compute MAE & MAPE in original scale
      const flatPred = [];
      const flatTrue = [];
      for (let i = 0; i < predArr.length; i++) {
        for (let t = 0; t < predArr[i].length; t++) {
          flatPred.push(predArr[i][t]);
          flatTrue.push(trueArr[i][t]);
        }
      }
      // Denormalize
      const predDenorm = this.loader.inverseScaleLabels(Float32Array.from(flatPred));
      const trueDenorm = this.loader.inverseScaleLabels(Float32Array.from(flatTrue));

      let mae = 0;
      let mapeAccum = 0;
      let mapeCount = 0;
      for (let i = 0; i < predDenorm.length; i++) {
        const a = trueDenorm[i];
        const p = predDenorm[i];
        mae += Math.abs(p - a);
        if (a !== 0) {
          mapeAccum += Math.abs((p - a) / a);
          mapeCount++;
        }
      }
      mae /= predDenorm.length;
      const mape = mapeCount > 0 ? (mapeAccum / mapeCount) * 100 : NaN;

      this.setMetrics(`Test MAE: ${mae.toFixed(2)} | MAPE: ${isNaN(mape) ? "n/a" : mape.toFixed(2) + "%"} (denormalized)`);

      // Build a time series for plotting the NON-OVERLAPPING test blocks
      // Each test window corresponds to 24 labels with a stored timestamp for the first label
      const labels = [];
      const seriesTrue = [];
      const seriesPred = [];
      const meta = this.dataset.testMeta;

      let idx = 0;
      for (let i = 0; i < predArr.length; i++) {
        const startTs = meta.labelTimestamps[i];
        for (let h = 0; h < 24; h++, idx++) {
          const ts = startTs + h * 3600000;
          labels.push(new Date(ts));
          seriesTrue.push(trueDenorm[idx]);
          seriesPred.push(predDenorm[idx]);
        }
      }

      // Render chart
      if (this.testChart) this.testChart.destroy();
      this.testChart = new Chart(this.testChartCtx, {
        type: "line",
        data: {
          labels,
          datasets: [
            { label: "Actual (Test)", data: seriesTrue, pointRadius: 0, borderWidth: 1, tension: 0.1 },
            { label: "Predicted", data: seriesPred, pointRadius: 0, borderWidth: 1, tension: 0.1 },
          ],
        },
        options: {
          responsive: true,
          animation: false,
          scales: {
            x: {
              type: "time",
              time: { unit: "hour", tooltipFormat: "yyyy-MM-dd HH:mm" },
              title: { display: true, text: "Time (Test period)" },
            },
            y: {
              title: { display: true, text: "Rented bike count" },
            },
          },
          plugins: {
            tooltip: { mode: "index", intersect: false },
            legend: { position: "top" },
          },
        },
      });

      this.setStatus("Prediction complete.");
    } catch (err) {
      console.error(err);
      this.setStatus(`Prediction error: ${err.message}`);
    } finally {
      // Dispose prediction tensor
      try {
        yPred?.dispose();
      } catch (_) {}
      this.disableActions(false);
      this.setProgress(0);
      await tf.nextFrame();
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  // eslint-disable-next-line no-new
  new BikeDemandApp();
});
