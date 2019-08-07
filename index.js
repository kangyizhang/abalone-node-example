require('@tensorflow/tfjs-node');

const tf = require('@tensorflow/tfjs');
const fs = require('fs');

function getModel() {
  // build neural network
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [8],
    activation: 'sigmoid',
    units: 50,
  }));
  model.add(tf.layers.dense({
    activation: 'sigmoid',
    units: 50,
  }));
  model.add(tf.layers.dense({
    units: 1,
  }));
  model.compile({optimizer: tf.train.adam(0.005), loss: 'meanSquaredError'});
  return model;
}

async function run() {
  const dataset = tf.data.csv(
      'file://Abalone.csv',
      {hasHeader: true, columnConfigs: {'rings': {isLabel: true}}});
  const convertedDataset =
      dataset
          .map(row => {
            const rawFeatures = row['xs'];
            const rawLabel = row['ys'];
            // const convertedFeatures = Object.values(rawFeatures);
            const convertedFeatures = Object.keys(rawFeatures).map(key => {
              switch (rawFeatures[key]) {
                case 'F':
                  return 0;
                case 'M':
                  return 1;
                case 'I':
                  return 2;
                default:
                  return Number(rawFeatures[key]);
              }
            });
            const convertedLabel = [rawLabel['rings']];
            return {xs: convertedFeatures, ys: convertedLabel};
          })
          .shuffle(1000)
          .batch(100);

  // const data = await iter.toArray();
  // const feature = tf.tensor2d(data.map((row: {xs: any, ys: any}) => {
  //                                   return row['xs'];
  //                                 })
  //                                 .map((row: {[key: string]: string}) => {
  //                                   return Object.keys(row).map(key => {
  //                                     switch (row[key]) {
  //                                       case 'F':
  //                                         return 0;
  //                                       case 'M':
  //                                         return 1;
  //                                       case 'I':
  //                                         return 2;
  //                                       default:
  //                                         return Number(row[key]);
  //                                     }
  //                                   });
  //                                 }));

  // const target =
  //     tf.tensor2d(data.map((row: {xs: any, ys: any}) => {
  //                       return row['xs'];
  //                     })
  //                     .map((row: {[key: string]: string}) => {
  //                       return Object.keys(row).map(key => Number(row[key]));
  //                     }));

  const model = getModel();

  // await model.fit(feature, target, {
  //   epochs: 100,
  //   validationSplit: 0.2,
  //   callbacks: {
  //     onEpochEnd: async (epoch, logs) => {
  //       // console.log(`Epoch ${epoch + 1} of ${100} completed.`);
  //     }
  //   }
  // });
  await model.fitDataset(convertedDataset, {
    epochs: 100,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} of ${100} completed.`);
      }
    }
  });

  // 9,9,14
  // console.log((model.predict(tf.tensor2d([
  //               [1, 0.355, 0.29, 0.09, 0.3275, 0.134, 0.086, 0.09],
  //               [0, 0.45, 0.335, 0.105, 0.425, 0.1865, 0.091, 0.115],
  //               [0, 0.55, 0.425, 0.135, 0.8515, 0.362, 0.196, 0.27]
  //             ]))
  //                 .dataSync()));

  // 9, 10, 11, 13
  console.log((model
                   .predict(tf.tensor2d([
                     [1, 0.4,0.29,0.115,0.2795,0.1115,0.0575,0.075],
                     [1, 0.705,0.56,0.165,1.675,0.797,0.4095,0.388],
                     [1, 0.63,0.505,0.15,1.3165,0.6325,0.2465,0.37],
                     [1, 0.655,0.525,0.18,1.402,0.624,0.2935,0.365]
                   ]))
                   .dataSync()));
}

run();
