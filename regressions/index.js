require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

const LinerRegression = require("./linearRegression");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffleL: true,
  splitTest: 50,
  dataColumns: ["horsepower", "displacement", "weight"],
  labelColumns: ["mpg"],
});

const regression = new LinerRegression(features, labels, {
  learningRate: 0.1,
  iterations: 100,
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);
console.log('r2', r2)