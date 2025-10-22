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
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Second LSTM reduces sequence to vector
  model.add(
    tf.layers.lstm({
      units: 32,
      returnSequences: false,
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Dense to 24 outputs (horizon)
  model.add(
    tf.layers.dense({
      units: 24,
      activation: "linear",
    })
  );

  model.compile({
    optimizer: tf.train.adam(),
    loss: "meanSquaredError",
    metrics: ["mae"],
  });

  return model;
}

// Train the model with callbacks to stream progress.
export async function trainModel(model, xTrain, yTrain, { epochs = 20, batchSize = 64, onEpoch, onBatch } = {}) {
  const history = await model.fit(xTrain, yTrain, {
    epochs,
    batchSize,
    shuffle: false, // time series: do NOT shuffle across time
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
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
