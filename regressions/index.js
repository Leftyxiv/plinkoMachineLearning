require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const plot = require('node-remote-plot');

const LinerRegression = require("./linearRegression");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffleL: true,
  splitTest: 50,
  dataColumns: ["horsepower", "displacement", "weight"],
  labelColumns: ["mpg"],
});

const regression = new LinerRegression(features, labels, {
  learningRate: 10,
  iterations: 100,
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
})
console.log('r2', r2)