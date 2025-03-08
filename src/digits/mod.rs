use crate::network::{self, Datapoint, Network, TrainingData};
use image::GrayImage;
use nalgebra::DVector;
use rand::seq::SliceRandom;
use rand::{rng, Rng};
use std::fs::File;
use std::io::{self, Read};
use crate::network::ActivationFunction;

pub fn train_and_save_digit_network(network_filename: &str, cost_data_filename: Option<&str>) -> io::Result<()> {
    let mut network = Network::new(&[28 * 28, 100, 10], &[ActivationFunction::ReLU, ActivationFunction::Softmax]);
    let mut dataset = DigitData::load_from_files(&[
        ("src/digits/data/data0", 0),
        ("src/digits/data/data1", 1),
        ("src/digits/data/data2", 2),
        ("src/digits/data/data3", 3),
        ("src/digits/data/data4", 4),
        ("src/digits/data/data5", 5),
        ("src/digits/data/data6", 6),
        ("src/digits/data/data7", 7),
        ("src/digits/data/data8", 8),
        ("src/digits/data/data9", 9),
    ]);
    if let Some(cost_data_filename) = cost_data_filename {
        network.train_cost_data(&mut dataset, &cost_data_filename)?;
    }
    else {
        network.train(&mut dataset);
    }
    network.save_to_file(network_filename)
}

pub fn load_and_test_digit_network(network_filename: &str) -> io::Result<()> {
    let mut network = Network::load_from_file(network_filename)?;
    println!("{:?}", network);
    let dataset = DigitData::load_from_files(&[
        ("src/digits/data/data0", 0),
        ("src/digits/data/data1", 1),
        ("src/digits/data/data2", 2),
        ("src/digits/data/data3", 3),
        ("src/digits/data/data4", 4),
        ("src/digits/data/data5", 5),
        ("src/digits/data/data6", 6),
        ("src/digits/data/data7", 7),
        ("src/digits/data/data8", 8),
        ("src/digits/data/data9", 9),
    ]);
    for data in dataset.take(10) {
        let output = network.forward(&data.input);
        let mut highest_num = 0;
        let mut highest_confidence = 0.0;
        for (num, confidence) in output.iter().enumerate() {
            println!(" - Number: {}. Confidence: {}.", num, confidence);
            if *confidence > highest_confidence {
                highest_num = num;
                highest_confidence = *confidence;
            }
        }
        println!("AI guess: {}. Confidence: {}.", highest_num, highest_confidence);
        println!("Correct: {}", data.expected);
    }
    Ok(())
}

pub struct DigitData {
    digits: Vec<(Vec<u8>, u8)>,
    current: usize,
}
impl DigitData {
    fn read_digit_file(file_path: &str) -> io::Result<Vec<Vec<u8>>> {
        let mut file = File::open(file_path)?;
        let mut data = Vec::new();

        // Each image is 28x28, so we need to read 28 * 28 bytes per image
        let mut buffer = [0u8; 28 * 28]; // Buffer for one image

        // Read 1000 images from the file
        for _ in 0..1000 {
            file.read_exact(&mut buffer)?;
            data.push(buffer.to_vec());
        }

        Ok(data)
    }
    pub fn load_from_files(paths_and_expected: &[(&str, u8)]) -> Self {
        let mut digits: Vec<(Vec<u8>, u8)> = vec![];
        for p_e in paths_and_expected {
            digits.extend(
                Self::read_digit_file(p_e.0)
                    .unwrap()
                    .into_iter()
                    .map(|e| (e, p_e.1)),
            );
        }

        digits.shuffle(&mut rng());

        Self { digits, current: 0 }
    }
    pub fn display_random(&self, file_name: &str) {
        let i = rng().random_range(0..self.digits.len());
        let image = GrayImage::from_raw(28, 28, self.digits[i].0.clone()).unwrap();
        image
            .save(format!("{}.png", file_name).to_string())
            .unwrap(); // Save it as a PNG for viewing
        println!("{}: {}", file_name, self.digits[i].1);
    }
}
impl Iterator for DigitData {
    type Item = Datapoint;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.digits.len() {
            return None;
        }
        let mut expected_vec = vec![0.0f32; 10];
        expected_vec[self.digits[self.current].1 as usize] = 1.0;

        let point = Some(Datapoint {
            input: DVector::from_iterator(
                self.digits[self.current].0.len(),
                self.digits[self.current]
                    .0
                    .iter()
                    .map(|e| (*e as f32) / 255.0),
            ),
            expected: DVector::from_vec(expected_vec),
        });
        self.current += 1;
        point
    }
}
impl TrainingData<Datapoint> for DigitData {}
