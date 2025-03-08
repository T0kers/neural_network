use serde::{Deserialize, Serialize};
use csv::Writer;
use std::{fs::File, io, vec};

use nalgebra::{DMatrix, DVector};
use rand::Rng;

pub trait DatapointRef {
    fn input(&self) -> &DVector<f32>;
    fn expected(&self) -> &DVector<f32>;
}

pub trait TrainingData<D: DatapointRef>: Iterator<Item = D> {}

pub struct ArithmeticData {
    current: usize,
    amount: usize,
}

impl ArithmeticData {
    pub fn new(amount: usize) -> Self {
        Self { current: 0, amount }
    }
}

pub struct Datapoint {
    pub input: DVector<f32>,
    pub expected: DVector<f32>,
}

impl DatapointRef for Datapoint {
    fn input(&self) -> &DVector<f32> {
        &self.input
    }
    fn expected(&self) -> &DVector<f32> {
        &self.expected
    }
}

impl Iterator for ArithmeticData {
    type Item = Datapoint;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.amount {
            return None;
        }
        self.current += 1;

        let mut rng = rand::rng();
        let a = rng.random::<f32>() / 2.0;
        let b = rng.random::<f32>() / 2.0;
        Some(Datapoint {
            input: DVector::from_vec(vec![a, b]),
            expected: DVector::from_vec(vec![a + b, a * b]),
        })
    }
}

impl TrainingData<Datapoint> for ArithmeticData {}

#[derive(Debug, Serialize, Deserialize, Copy, Clone)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Softmax,
}

impl ActivationFunction {
    pub fn apply(&self, x: &DVector<f32>) -> DVector<f32> {
        match self {
            Self::ReLU => x.map(|e| e.max(0.0)), // ReLU: f(x) = max(0, x)
            Self::Sigmoid => x.map(|e| 1.0 / (1.0 + (-e.clamp(-10.0, 10.0)).exp())), // clamping to prevent exploding values and NaN.
            Self::Softmax => {
                let exp_x = x.map(|elem| elem.exp());
                let sum_exp_x = exp_x.sum();
                exp_x / sum_exp_x
            }
        }
    }
    pub fn derivative(&self, y: &DVector<f32>, slope: &mut DVector<f32>) {
        match self {
            Self::ReLU => slope.component_mul_assign(&y.map(|e| if e > 0.0 { 1.0 } else { 0.0 })), // ReLU derivative
            Self::Sigmoid => slope.component_mul_assign(&y.map(|e| e * (1.0 - e))), // clamping to prevent exploding values and NaN.
            Self::Softmax => {
                let len = y.len();
                for i in 0..len {
                    for j in 0..len {
                        if i == j {
                            slope[i] = y[i] * (1.0 - y[i]);
                        } else {
                            slope[i] -= y[i] * y[j];
                        }
                    }
                }
            }
        }
    }
}


#[derive(Debug, Serialize, Deserialize)]
struct Layer {
    weights: DMatrix<f32>,
    biases: DVector<f32>,

    // fields below are only used for training, and are not stored in file
    #[serde(skip)]
    input: DVector<f32>,
    #[serde(skip)]
    pre_activation: DVector<f32>, // todo remove redundency
    #[serde(skip)]
    post_activation: DVector<f32>, // todo remove redundency
    activation_function: ActivationFunction,
}

impl Layer {
    const LEARNING_RATE: f32 = 0.1;
    fn new(nodes_out: usize, nodes_in: usize, activation_function: ActivationFunction) -> Self {
        let mut rng = rand::rng();
        let variance = (2.0 / (nodes_in as f32 + nodes_out as f32)).sqrt(); // Xavier (Glorot) initialization
        Self {
            weights: DMatrix::from_fn(nodes_out, nodes_in, |_, _| {
                rng.random_range(-variance..variance)
            }),
            biases: DVector::from_fn(nodes_out, |_, _| 0.0),
            input: DVector::zeros(nodes_in),
            pre_activation: DVector::zeros(nodes_out),
            post_activation: DVector::zeros(nodes_out),
            activation_function,
        }
    }
    fn calculate_activation(&mut self, input: DVector<f32>) -> DVector<f32> {
        self.activation_function.apply(&(&self.weights * input + &self.biases))
    }
    fn train_calculate_activation(&mut self, input: &DVector<f32>) -> &DVector<f32> {
        self.input = input.clone();
        self.pre_activation = &self.weights * &self.input + &self.biases;
        self.post_activation = self.activation_function.apply(&self.pre_activation);
        &self.post_activation
    }
    pub fn update_weights_and_biases(&mut self, slope: &DVector<f32>) {
        self.biases -= slope * Self::LEARNING_RATE;
        self.weights -= slope * self.input.transpose() * Self::LEARNING_RATE;
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CostData {
    runs: usize,
    cost: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_node_count: &[usize], activation_functions: &[ActivationFunction]) -> Self {
        let mut layers = vec![];

        for i in 0..&layer_node_count.len() - 1 {
            layers.push(Layer::new(layer_node_count[i + 1], layer_node_count[i], activation_functions[i]));
        }

        Self { layers }
    }
    pub fn load_from_file(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let network = serde_json::from_reader(file)?;
        Ok(network)
    }
    pub fn save_to_file(&self, filename: &str) -> io::Result<()> {
        let file = File::create(filename)?;
        serde_json::to_writer(file, self)?;
        Ok(())
    }
    pub fn forward(&mut self, input: &DVector<f32>) -> DVector<f32> {
        let mut result = input.clone();
        // println!("{}", result);
        for layer in &mut self.layers {
            result = layer.calculate_activation(result);
            // println!("{}", result);
        }
        result
    }

    fn train_forward(&mut self, input: &DVector<f32>) -> DVector<f32> {
        let mut result = input;
        for layer in &mut self.layers {
            result = layer.train_calculate_activation(result);
        }
        result.clone()
    }

    pub fn train<T: TrainingData<impl DatapointRef>>(&mut self, dataset: &mut T) {
        for data in dataset {
            let output = self.train_forward(&data.input());
            let mut slope = Self::cost_derivative(&output, &data.expected());
            for layer in self.layers.iter_mut().rev() {
                layer.activation_function.derivative(&layer.post_activation, &mut slope);
                layer.update_weights_and_biases(&slope);
                slope = &layer.weights.transpose() * slope;
            }
        }
    }

    pub fn train_cost_data<T: TrainingData<impl DatapointRef>>(&mut self, dataset: &mut T, filename: &str) -> io::Result<()> {
        let file = File::create(filename)?;
        let mut wtr = Writer::from_writer(file);

        for (i, data) in dataset.enumerate() {
            let output = self.train_forward(&data.input());
            wtr.serialize(CostData {runs: i + 1, cost: Self::cost_size(&output, &data.expected())})?;
            let mut slope = Self::cost_derivative(&output, &data.expected());
            for layer in self.layers.iter_mut().rev() {
                layer.activation_function.derivative(&layer.post_activation, &mut slope);
                layer.update_weights_and_biases(&slope);
                slope = &layer.weights.transpose() * slope;
            }
        }
        wtr.flush()?;
        Ok(())
    }

    fn cost(output: &DVector<f32>, expected: &DVector<f32>) -> DVector<f32> {
        output.zip_map(&expected, |o, e| (o - e).powi(2))
    }
    fn cost_size(output: &DVector<f32>, expected: &DVector<f32>) -> f32 {
        Self::cost(output, expected).magnitude()
    }
    // derivative w.r.t. output
    fn cost_derivative(output: &DVector<f32>, expected: &DVector<f32>) -> DVector<f32> {
        output.zip_map(&expected, |o, e| 2.0 * (o - e))
    }
    pub fn last_layer_debug(&self) {
        println!("{}", self.layers.last().unwrap().weights);
        println!("{}", self.layers.last().unwrap().biases);
    }
}
