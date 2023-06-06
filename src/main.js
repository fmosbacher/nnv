import { NeuralNet, Matrix, Activation, Chart } from './lib.js';

const MAX_EPOCHS = 100_000;
const LEARNING_RATE = 0.2;
const COST_THRESHOLD = 1e-4;
const K = 0.1;

const nn = new NeuralNet(
  2,
  { activation: Activation.TanH, neurons: 4 },
  { activation: Activation.TanH, neurons: 2 },
  { activation: Activation.TanH, neurons: 1 }
);
const chart = new Chart();

const xs = [
  new Matrix(1, 2, [0, 0]),
  new Matrix(1, 2, [0, 1]),
  new Matrix(1, 2, [1, 0]),
  new Matrix(1, 2, [1, 1]),
];

const ys = [
  new Matrix(1, 1, [0]),
  new Matrix(1, 1, [1]),
  new Matrix(1, 1, [1]),
  new Matrix(1, 1, [0]),
];

let epoch = 0;
let cost;

requestAnimationFrame(run);

function run() {
  epoch += 1;

  cost = nn.cost(xs, ys);

  if (cost < COST_THRESHOLD || isNaN(cost)) {
    printResults();
    return;
  }

  chart.addData(cost);
  chart.plot();

  for (let sample = 0; sample < xs.length; sample++) {
    nn.backprop(xs[sample], ys[sample], LEARNING_RATE);
  }

  if (epoch === MAX_EPOCHS) {
    printResults();
    return;
  }

  requestAnimationFrame(run);
}

function printResults() {
  console.log(`epoch: ${epoch}, cost: ${cost}`);

  for (let sample = 0; sample < xs.length; sample++) {
    console.log(xs[sample].values, nn.forward(xs[sample]).values);
  }
}
