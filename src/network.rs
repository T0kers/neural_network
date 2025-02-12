use std::{ops::Mul, vec};

use nalgebra::{DMatrix, DVector};
use rand::Rng;

pub struct Datapoint {
    pub input: DVector<f32>,
    pub expected: DVector<f32>,
}

pub trait TrainingData: Iterator<Item = Datapoint> {}

pub struct MultiplicationData {
    current: usize,
    amount: usize,
}

impl MultiplicationData {
    pub fn new(amount: usize) -> Self {
        Self {
            current: 0,
            amount
        }
    }
}

impl Iterator for &mut MultiplicationData {
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
            expected: DVector::from_vec(vec![a + b]),
        })
    }
}

impl TrainingData for &mut MultiplicationData {}

#[derive(Debug)]
struct Layer {
    weights: DMatrix<f32>,
    biases: DVector<f32>,
    input: DVector<f32>,
}

impl Layer {
    const LEARNING_RATE: f32 = 0.1;
    fn new(nodes_out: usize, nodes_in: usize) -> Self {
        let mut rng = rand::rng();
        Self {
            weights: DMatrix::from_fn(nodes_out, nodes_in, |_, _| rng.random_range(-1.0..1.0)),
            biases: DVector::from_fn(nodes_out, |_, _| rng.random_range(-1.0..1.0)),
            input: DVector::from_vec(vec![0.0; nodes_in]),
        }
    }
    pub fn calculate_output(&mut self, input: DVector<f32>) -> DVector<f32> {
        self.input = input;
        &self.weights * &self.input + &self.biases
    }
    pub fn update_weights_and_biases(&mut self, slope: &DVector<f32>) {
        self.biases += slope * Self::LEARNING_RATE;
        self.weights += slope * self.input.transpose() * Self::LEARNING_RATE;
    }
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>
}

impl Network {
    pub fn new(layer_node_count: &Vec<usize>) -> Self {
        let mut layers = vec![];

        for i in 0..&layer_node_count.len() - 1 {
            layers.push(Layer::new(layer_node_count[i + 1], layer_node_count[i]));
        }

        Self {
            layers
        }
    }
}

impl Network {
    pub fn forward(&mut self, input: &DVector<f32>) -> DVector<f32> {
        let mut result = input.clone();
        for layer in &mut self.layers {
            result = Self::activation(&layer.calculate_output(result));
        }
        result
    }
    pub fn train<T: TrainingData>(&mut self, dataset: &mut T) {
        for data in dataset {
            let output = self.forward(&data.input);
            let mut slope = Self::cost_derivative(&output, &data.expected);
            for layer in self.layers.iter_mut().rev() {
                slope *= Self::activation_derivative(&output);
                layer.update_weights_and_biases(&slope);
                slope = &layer.weights * slope;
            }
        }
    }

    // sigmoid function
    fn activation(x: &DVector<f32>) -> DVector<f32> {
        x.map(|e| 1.0 / (1.0 + (-e).exp()))
        
    }
    fn activation_derivative(y: &DVector<f32>) -> DVector<f32> { // notice the input is activation(x)
        y.map(|e| e * (1.0 - e))
    }
    fn cost(output: &DVector<f32>, expected: &DVector<f32>) -> DVector<f32> {
        output.zip_map(&expected, |o, e| (o - e).powi(2))
    }
    // derivative w.r.t. output
    fn cost_derivative(output: &DVector<f32>, expected: &DVector<f32>) -> DVector<f32> {
        output.zip_map(&expected, |o, e| 2.0 * (o - e))
    }
}