// lstm-model.js
// Defines a 2-layer LSTM sequence-to-sequence regressor in TensorFlow.js.
// Input: [batch, 24, num_features]
// Output: [batch, 24] (next 24 hours of demand; linear activation)

export function createLstmModel(numFeatures) {
  const model = tf.sequential();

  // First LSTM returns sequences for stacking
  model.add(
    tf.layers.lstm({
      units: 64,
      returnSequences: true,
      inputShape: [24, numFeatures],
      kernelInitializer: 'glorotUniform', // Xavier инициализация
      recurrentInitializer: 'orthogonal',  // Ортогональная инициализация для рекуррентных весов
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), // L2 регуляризация
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Second LSTM reduces sequence to vector
  model.add(
    tf.layers.lstm({
      units: 32,
      returnSequences: false,
      kernelInitializer: 'glorotUniform',
      recurrentInitializer: 'orthogonal',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Dense to 24 outputs (horizon)
  model.add(
    tf.layers.dense({
      units: 24,
      activation: "linear",
      kernelInitializer: 'glorotUniform',
    })
  );

  const optimizer = tf.train.adam(0.0001); // Уменьшенный learning rate для стабильности
  
  model.compile({
    optimizer: optimizer,
    loss: "meanSquaredError",
    metrics: ["mae"],
  });

  return model;
}

// Train the model with callbacks to stream progress.
export async function trainModel(model, xTrain, yTrain, { epochs = 20, batchSize = 64, onEpoch, onBatch } = {}) {
  // Проверяем данные на NaN перед обучением
  const xStats = tf.tidy(() => {
    const hasNaN = tf.any(tf.isNaN(xTrain));
    const hasInf = tf.any(tf.isInf(xTrain));
    return { hasNaN: hasNaN.dataSync()[0], hasInf: hasInf.dataSync()[0] };
  });
  
  const yStats = tf.tidy(() => {
    const hasNaN = tf.any(tf.isNaN(yTrain));
    const hasInf = tf.any(tf.isInf(yTrain));
    return { hasNaN: hasNaN.dataSync()[0], hasInf: hasInf.dataSync()[0] };
  });
  
  if (xStats.hasNaN || xStats.hasInf || yStats.hasNaN || yStats.hasInf) {
    console.warn('⚠️ Обнаружены NaN или Infinity в данных!', { xStats, yStats });
  }
  
  const history = await model.fit(xTrain, yTrain, {
    epochs,
    batchSize,
    shuffle: false, // time series: do NOT shuffle across time
    validationSplit: 0.1, // добавляем валидацию для мониторинга
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Проверяем на NaN в метриках
        if (!Number.isFinite(logs.loss) || !Number.isFinite(logs.mae)) {
          console.error('❌ NaN обнаружен в метриках на эпохе', epoch + 1, logs);
        }
        if (onEpoch) onEpoch(epoch, logs);
        await tf.nextFrame();
      },
      onBatchEnd: async (batch, logs) => {
        if (onBatch) onBatch(batch, logs);
        // Yield to UI thread
        await tf.nextFrame();
      },
    },
  });
  return history;
}

// Predict for a batch of test sequences. Returns a Tensor2D [N, 24]
export function predictHorizon(model, xTest) {
  return tf.tidy(() => model.predict(xTest));
}
