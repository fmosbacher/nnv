export class Matrix {
  constructor(rows, cols, values) {
    this.rows = rows;
    this.cols = cols;
    this.values =
      values ||
      Array.from({ length: rows * cols }, () => Math.random() * 2 - 1);
  }

  clone() {
    return new Matrix(this.rows, this.cols, [...this.values]);
  }

  get(row, col) {
    return this.values[row * this.cols + col];
  }

  dot(other) {
    const values = [];

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < other.cols; j++) {
        let sum = 0;

        for (let k = 0; k < this.cols; k++) {
          sum += this.get(i, k) * other.get(k, j);
        }

        values.push(sum);
      }
    }

    return new Matrix(this.rows, other.cols, values);
  }

  transpose() {
    const values = [];

    for (let col = 0; col < this.cols; col++) {
      for (let row = 0; row < this.rows; row++) {
        values.push(this.get(row, col));
      }
    }

    return new Matrix(this.cols, this.rows, values);
  }

  map(fn) {
    return new Matrix(this.rows, this.cols, this.values.map(fn));
  }

  broadcast(other, op) {
    const values = [];
    const rows = Math.max(this.rows, other.rows);
    const cols = Math.max(this.cols, other.cols);

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const value = op(
          this.get(row % this.rows, col % this.cols),
          other.get(row % other.rows, col % other.cols)
        );

        values.push(value);
      }
    }

    return new Matrix(rows, cols, values);
  }

  add(other) {
    return this.broadcast(other, (a, b) => a + b);
  }

  sub(other) {
    return this.broadcast(other, (a, b) => a - b);
  }

  mult(other) {
    return this.broadcast(other, (a, b) => a * b);
  }

  div(other) {
    return this.broadcast(other, (a, b) => a / b);
  }
}

function sigmoid(x) {
  return 1 / (1 + Math.E ** -x);
}

function dsigmoid(x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

function relu(k) {
  return (x) => (x > 0 ? x : x * k);
}

function drelu(k) {
  return (x) => (x > 0 ? 1 : k);
}

function tanh(x) {
  Math.tanh;
}

export const Activation = {
  Sigmoid: { fn: sigmoid, deriv: dsigmoid },
  ReLU: { fn: relu(0), deriv: drelu(0) },
  LeakyReLU: (k) => ({ fn: relu(k), deriv: drelu(k) }),
  Sin: { fn: Math.sin, deriv: Math.cos },
  TanH: { fn: Math.tanh, deriv: (x) => 1 - Math.tanh(x) ** 2 },
};

class Layer {
  constructor(inputsCount, outputsCount, activation) {
    this.weights = new Matrix(inputsCount, outputsCount);
    this.biases = new Matrix(1, outputsCount);
    this.activation = activation;
    this.zs;
    this.as;
  }

  activate(inputs) {
    this.zs = inputs.dot(this.weights).add(this.biases);
    this.as = this.zs.map((z) => this.activation.fn(z));
    return this.as;
  }
}

export class NeuralNet {
  constructor(inputNeuronsCount, ...layersConfig) {
    this.neurons = [
      inputNeuronsCount,
      ...layersConfig.map((config) => config.neurons),
    ];

    this.layers = layersConfig.map((config, i) => {
      return new Layer(
        i === 0 ? inputNeuronsCount : layersConfig[i - 1].neurons,
        config.neurons,
        config.activation
      );
    });
  }

  clone() {
    const cloned = NeuralNet([...this.neurons]);

    cloned.weights = this.weights.map((w) => w.clone());
    cloned.biases = this.biases.map((b) => b.clone());

    return cloned;
  }

  backprop(inputs, expectedOutputs, learningRate) {
    const outputs = this.forward(inputs);
    let delta;

    for (let i = this.layers.length - 1; i >= 0; i--) {
      const isLastLayer = i === this.layers.length - 1;
      const isFirstLayer = i === 0;
      const prevActiv = isFirstLayer ? inputs : this.layers[i - 1].as;
      const currZsDeriv = this.layers[i].zs.map((z) =>
        this.layers[i].activation.deriv(z)
      );

      if (isLastLayer) {
        delta = outputs.sub(expectedOutputs).mult(currZsDeriv);
      } else {
        const nextW = this.layers[i + 1].weights;
        delta = delta.dot(nextW.transpose()).mult(currZsDeriv);
      }

      const dw = prevActiv
        .transpose()
        .dot(delta)
        .map((d) => d * learningRate);
      this.layers[i].weights = this.layers[i].weights.sub(dw);

      const db = delta.map((d) => d * learningRate);
      this.layers[i].biases = this.layers[i].biases.sub(db);
    }
  }

  forward(inputs) {
    let lastActivation = inputs;

    this.layers.forEach((layer) => {
      lastActivation = layer.activate(lastActivation);
    });

    return lastActivation;
  }

  cost(inputsBatch, outputsBatch) {
    const outputs = inputsBatch.map((inputs) => this.forward(inputs));
    const samplesCount = inputsBatch.length;

    return (
      outputs
        .map((output, i) => output.sub(outputsBatch[i]))
        .map((diffs) => diffs.map((diff) => diff ** 2))
        .map((squaredDiffs) => squaredDiffs.values)
        .flat()
        .reduce((a, b) => a + b) / samplesCount
    );
  }
}

export class Chart {
  constructor() {
    this.data = [];
    this.ctx = document.getElementsByTagName('canvas')[0].getContext('2d');
  }

  addData(d) {
    this.data.push(d);
  }

  clear() {
    this.ctx.fillStyle = '#222';
    this.ctx.rect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
    this.ctx.fill();
  }

  plot() {
    const padding = 20;
    const max = Math.max(...this.data);
    const width = this.ctx.canvas.width - padding;
    const height = this.ctx.canvas.height - padding;
    const xOffset = width / (this.data.length - 1 || 1);

    this.clear();

    for (let i = 0; i < this.data.length; i++) {
      const yPrev =
        i === 0
          ? padding / 2
          : (1 - this.data[i - 1] / max) * height + padding / 2;
      const xPrev = i === 0 ? padding / 2 : (i - 1) * xOffset + padding / 2;
      const y = (1 - this.data[i] / max) * height + padding / 2;
      const x = i * xOffset + padding / 2;

      this.ctx.strokeStyle = '#e33';
      this.ctx.beginPath();
      this.ctx.moveTo(xPrev, yPrev);
      this.ctx.lineTo(x, y);
      this.ctx.lineWidth = 5;
      this.ctx.lineCap = 'round';
      this.ctx.stroke();
    }

    this.ctx.font = '30px Monospace';
    this.ctx.fillStyle = '#eee';
    this.ctx.fillText(
      `Cost: ${this.data[this.data.length - 1].toFixed(6)}`,
      width - 350,
      50
    );
    this.ctx.fillText(`Epoch: ${this.data.length}`, width - 350, 100);
  }
}
