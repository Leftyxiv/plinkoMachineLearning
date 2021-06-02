require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const _ = require('lodash')

const loadCSV = require('../load-csv');
const LogisticRegression = require('./logisticRegression');
const plot = require('node-remote-plot')

const { features, labels, testFeatures, testLabels } = loadCSV('data/cars.csv', {
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg'],
  shuffle: true,
  splitTest: 50,
  converters: {
    mpg: (value) => {
      const mpg = parseFloat(value);
      if (mpg < 15){
        return [1, 0, 0]
      } else if (mpg < 30) {
        return [0, 1, 0]
      } else {
        return [0, 0, 1]
      }
    }
  }
});


const regression = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50,
  // decisionBoundary: .6,
})

regression.train()


console.log(regression.test(testFeatures, _.flatMap(testLabels)))