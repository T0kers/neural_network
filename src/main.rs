use std::env;

use nalgebra::DVector;
use network::{MultiplicationData, Network};

pub mod network;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut network = Network::new(&vec![2, 3, 1]);
    println!("{:?}", network);
    let mut dataset = MultiplicationData::new(100000);
    network.train(&mut &mut dataset);

    let mut test_set = MultiplicationData::new(10);
    for test in &mut test_set {
        println!("Input: {} + {}. Prediction: {}. Answer {}.", test.input[0], test.input[1], network.forward(&test.input)[0], test.expected[0]);
    }
}
