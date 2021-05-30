require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

const LinerRegression = require("./linearRegression");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffleL: true,
  splitTest: 50,
  dataColumns: ["horsepower"],
  labelColumns: ["mpg"],
});

const regression = new LinerRegression(features, labels, {
  learningRate: 0.0001,
  iterations: 100,
});

regression.train();

console.log(regression.weights.get(1, 0), "-----m-----", regression.weights.get(0, 0), "-----b-------");
