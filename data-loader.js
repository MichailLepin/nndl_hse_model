// data-loader.js
// Utility module to load, preprocess, and window the bike-sharing data entirely in-browser.
// - Parses CSV (expects headers).
// - Casts date and hour, sorts chronologically.
// - One-hot encodes seasons, holiday, functioning_day.
// - Adds cyclical hour features (sin/cos).
// - Min–max normalizes numeric features using TRAIN-ONLY stats.
// - Builds sequences: lookback=24, horizon=24.
// - Splits chronologically into train/test (default 80/20).

export class DataLoader {
  constructor() {
    this.lookback = 24;
    this.horizon = 24;
    this.trainSplit = 0.8;

    this.numericCols = [
      "temperature",
      "humidity",
      "wind_speed",
      "visibility",
      "dew_point_temperature",
      "solar_radiation",
      "rainfall",
      "snowfall",
    ];

    this.catCols = ["seasons", "holiday", "functioning_day"];

    // Fitted on training set only
    this.featureMin = null;
    this.featureMax = null;
    this.labelMin = null;
    this.labelMax = null;

    this.featureNames = []; // after expansion (num + hour_sin/cos + one-hots)
  }

  async parseCSVFile(file) {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => resolve(results.data),
        error: (err) => reject(err),
      });
    });
  }

  // Normalize with min-max; in-place if arrays provided
  static _minMaxNormalize(value, min, max) {
    if (max === min) return 0; // avoid NaN
    return (value - min) / (max - min);
  }
  static _minMaxDenormalize(value, min, max) {
    return value * (max - min) + min;
  }

  _ensureTypesAndSort(rows) {
    // Ensure date and hour types; sort by timestamp asc
    const out = rows
      .map((r) => {
        const dateStr = r.date || r.Date || r.dates || r["Date"];
        const hourVal = r.hour ?? r.Hour ?? r["hour"] ?? r["Hour"];
        const dateObj = new Date(dateStr);
        const hourNum = typeof hourVal === "number" ? hourVal : parseInt(hourVal, 10);
        const rented = r.rented_bike_count ?? r["rented_bike_count"] ?? r["Rented Bike Count"];
        return {
          ...r,
          _date: dateObj,
          _hour: hourNum,
          _ts: dateObj.getTime() + (hourNum ?? 0) * 3600000,
          _label: Number(rented),
        };
      })
      .filter((r) => Number.isFinite(r._ts) && Number.isFinite(r._hour) && Number.isFinite(r._label))
      .sort((a, b) => a._ts - b._ts);
    return out;
  }

  _collectCategories(rows) {
    const cats = {};
    for (const col of this.catCols) {
      cats[col] = new Set();
    }
    rows.forEach((r) => {
      for (const col of this.catCols) {
        const v = r[col];
        if (v !== undefined && v !== null && v !== "") cats[col].add(String(v));
      }
    });
    // Freeze order to arrays
    const catMaps = {};
    for (const col of this.catCols) {
      catMaps[col] = Array.from(cats[col]).sort();
    }
    return catMaps;
  }

  _buildFeatureVectors(rows, catMaps) {
    // Construct feature vectors per row:
    // - numericCols
    // - hour sin/cos
    // - one-hot(seasons, holiday, functioning_day)
    const featureNames = [];

    // Numeric
    for (const n of this.numericCols) featureNames.push(n);

    // Hour cyclical
    featureNames.push("hour_sin", "hour_cos");

    // Categorical one-hots
    for (const col of this.catCols) {
      for (const v of catMaps[col]) {
        featureNames.push(`${col}__${v}`);
      }
    }
    this.featureNames = featureNames;

    const X = new Array(rows.length);
    const y = new Float32Array(rows.length);
    const timestamps = new Array(rows.length);

    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const feats = [];

      // Numeric
      for (const n of this.numericCols) {
        feats.push(Number(r[n]));
      }
      // Hour sin/cos
      const h = Number(r._hour) % 24;
      const angle = (2 * Math.PI * h) / 24;
      feats.push(Math.sin(angle), Math.cos(angle));

      // One-hots
      for (const col of this.catCols) {
        const options = catMaps[col];
        const rv = String(r[col]);
        for (const opt of options) {
          feats.push(rv === opt ? 1 : 0);
        }
      }
      X[i] = feats;
      y[i] = r._label;
      timestamps[i] = r._ts;
    }
    return { X, y, timestamps, featureNames };
  }

  _splitChronologically(len) {
    const splitIdx = Math.floor(len * this.trainSplit);
    return { splitIdx };
  }

  _fitFeatureScalers(Xtrain) {
    const F = Xtrain[0].length;
    const fmin = new Float32Array(F).fill(Number.POSITIVE_INFINITY);
    const fmax = new Float32Array(F).fill(Number.NEGATIVE_INFINITY);
    for (const row of Xtrain) {
      for (let j = 0; j < F; j++) {
        const v = row[j];
        if (Number.isFinite(v)) {
          if (v < fmin[j]) fmin[j] = v;
          if (v > fmax[j]) fmax[j] = v;
        }
      }
    }
    this.featureMin = fmin;
    this.featureMax = fmax;
  }

  _fitLabelScaler(yTrain) {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const v of yTrain) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    this.labelMin = min;
    this.labelMax = max;
  }

  _normalizeFeatures(X) {
    const F = this.featureMin.length;
    const out = new Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const row = X[i];
      const nr = new Float32Array(F);
      for (let j = 0; j < F; j++) {
        nr[j] = DataLoader._minMaxNormalize(row[j], this.featureMin[j], this.featureMax[j]);
      }
      out[i] = nr;
    }
    return out;
  }

  _normalizeLabels(y) {
    const out = new Float32Array(y.length);
    for (let i = 0; i < y.length; i++) {
      out[i] = DataLoader._minMaxNormalize(y[i], this.labelMin, this.labelMax);
    }
    return out;
  }

  _buildWindows(X, y, timestamps, lookback, horizon, stride) {
    // Returns {xs:[Tensors? no; raw arrays], ys:[], meta:{labelStartIdx:[], labelTimestamps:[]}}
    // We'll convert to tensors later to control memory.
    const xs = [];
    const ys = [];
    const labelStartIdx = [];
    const labelTimestamps = [];
    for (let t = lookback; t + horizon <= X.length; t += stride) {
      const xSlice = X.slice(t - lookback, t); // length lookback
      const ySlice = y.slice(t, t + horizon); // next 24 labels
      xs.push(xSlice);
      ys.push(ySlice);
      labelStartIdx.push(t);
      labelTimestamps.push(timestamps[t]); // timestamp at first label step
    }
    return { xs, ys, meta: { labelStartIdx, labelTimestamps } };
  }

  inverseScaleLabels(normalizedArray) {
    // Accepts Float32Array or number[]
    const min = this.labelMin;
    const max = this.labelMax;
    const out = new Float32Array(normalizedArray.length);
    for (let i = 0; i < normalizedArray.length; i++) {
      out[i] = DataLoader._minMaxDenormalize(normalizedArray[i], min, max);
    }
    return out;
  }

  async loadAndPrepare(file) {
    const raw = await this.parseCSVFile(file);
    const rows = this._ensureTypesAndSort(raw);
    if (rows.length < this.lookback + this.horizon + 1) {
      throw new Error("Dataset is too small after cleaning to build sequences.");
    }

    const catMaps = this._collectCategories(rows);
    const { X, y, timestamps, featureNames } = this._buildFeatureVectors(rows, catMaps);
    const { splitIdx } = this._splitChronologically(X.length);

    // Split raw series
    const XtrainRaw = X.slice(0, splitIdx);
    const yTrainRaw = y.slice(0, splitIdx);
    const tTrain = timestamps.slice(0, splitIdx);

    const XtestRaw = X.slice(splitIdx);
    const yTestRaw = y.slice(splitIdx);
    const tTest = timestamps.slice(splitIdx);

    // Fit scalers on training
    this._fitFeatureScalers(XtrainRaw);
    this._fitLabelScaler(yTrainRaw);

    // Normalize features & labels
    const Xtrain = this._normalizeFeatures(XtrainRaw);
    const yTrain = this._normalizeLabels(yTrainRaw);
    const Xtest = this._normalizeFeatures(XtestRaw);
    const yTest = this._normalizeLabels(yTestRaw);

    // Build windows
    // - Train: stride=1 for more samples
    // - Test: stride=24 for non-overlapping day blocks (easier plotting)
    const trainWin = this._buildWindows(
      Xtrain,
      yTrain,
      tTrain,
      this.lookback,
      this.horizon,
      1
    );
    const testWin = this._buildWindows(
      Xtest,
      yTest,
      tTest,
      this.lookback,
      this.horizon,
      24
    );

    // Convert to tensors
    const numFeatures = featureNames.length;
    const toTensor3D = (blocks) => {
      // blocks: array of [lookback][features] Float32Arrays
      const n = blocks.length;
      const buf = new Float32Array(n * this.lookback * numFeatures);
      let o = 0;
      for (let i = 0; i < n; i++) {
        const seq = blocks[i];
        for (let t = 0; t < this.lookback; t++) {
          const row = seq[t];
          for (let f = 0; f < numFeatures; f++) {
            buf[o++] = row[f];
          }
        }
      }
      return tf.tensor3d(buf, [n, this.lookback, numFeatures]);
    };
    const toTensor2D = (blocks) => {
      const n = blocks.length;
      const buf = new Float32Array(n * this.horizon);
      let o = 0;
      for (let i = 0; i < n; i++) {
        const row = blocks[i];
        for (let t = 0; t < this.horizon; t++) buf[o++] = row[t];
      }
      return tf.tensor2d(buf, [n, this.horizon]);
    };

    const xTrain = toTensor3D(trainWin.xs);
    const yTrain = toTensor2D(trainWin.ys);
    const xTest = toTensor3D(testWin.xs);
    const yTest = toTensor2D(testWin.ys);

    return {
      xTrain,
      yTrain,
      xTest,
      yTest,
      featureNames,
      numFeatures,
      lookback: this.lookback,
      horizon: this.horizon,
      testMeta: testWin.meta,
      scalers: {
        featureMin: this.featureMin,
        featureMax: this.featureMax,
        labelMin: this.labelMin,
        labelMax: this.labelMax,
      },
    };
  }
}
