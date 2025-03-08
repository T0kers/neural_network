pub mod digits;

use std::{env, io};

use digits::{load_and_test_digit_network, train_and_save_digit_network};
use network::{ActivationFunction, ArithmeticData, Network};

pub mod network;

fn main() -> io::Result<()> {
    env::set_var("RUST_BACKTRACE", "1");

    // train_and_save_digit_network("digit_network.json", Some("cost_data.csv"))?;
    load_and_test_digit_network("digit_network.json")?;
    // addition_test();
    Ok(())
}

fn addition_test() {
    let mut network = Network::new(&[2, 6, 3, 2], &[ActivationFunction::ReLU, ActivationFunction::ReLU, ActivationFunction::ReLU]);
    println!("{:?}", network);
    let mut dataset = ArithmeticData::new(100);
    network.train(&mut dataset);

    let mut test_set = ArithmeticData::new(10);
    for test in &mut test_set {
        println!(
            "Input: {} +* {}. \n* Prediction: {}. Answer {}.\n+ Prediction: {}. Answer: {}.",
            test.input[0],
            test.input[1],
            network.forward(&test.input)[0],
            test.expected[0],
            network.forward(&test.input)[1],
            test.expected[1]
        );
    }
    println!("{:?}", network);
}
