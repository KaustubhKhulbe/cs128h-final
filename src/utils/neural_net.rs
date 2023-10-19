use nalgebra::{DVector, DMatrix};

use super::{layer::Layer, activation::Activation};

pub struct NeuralNetwork {
    layers: Vec<Layer>
}

impl NeuralNetwork {
    pub fn new(input: DVector<f64>) -> Self{
        NeuralNetwork {
            layers: vec![Layer::input_layer(input)],
        }
    }

    pub fn add_layer(&mut self, size: usize, activation: Activation) {
        let previous_layer = self.layers.last().unwrap();
        let previous_layer_size = previous_layer.computed_.nrows();
        let layer = Layer::new(size, previous_layer_size, activation, previous_layer.computed_.clone()); // todo! clone has large space complexity
        self.layers.push(layer);
    }

    pub fn forward_propogation(&mut self) {
        for i in 1..self.layers.len() {
            let _ = &self.layers[i].single_layer_forward_propogation();
        }
    }

    pub fn print_network(&self) {
        for layer in &self.layers {
            println!("{}", layer.computed_);
        }
    }

    pub fn get_cost_val(&self, y_hat: &DVector<f64>, y: &DVector<f64>) -> f64 {
        let m = y_hat.nrows();
        println!("{}", m);
        let y_hat_ln = y_hat.clone().map(|x| { x.ln()});
        let one_minus_y = y.clone().map(|x| { 1.0 - x});
        let y_hat_one_minus_x_ln = y_hat.clone().map(|x| { (1.0 - x).ln()});

        let cost = (-1.0 / m as f64) * (
            y.dot(&y_hat_ln) + one_minus_y.dot(&y_hat_one_minus_x_ln) // check later
        );

        return cost;
    }

    pub fn get_accuracy_val(&self, y_hat: DVector<f64>, y: DVector<f64>) -> f64{
        let y_hat_ = self.convert_prob_into_class(y_hat);
        let mut correct = 0;
        for i in 0..y_hat_.nrows() {
            correct += (y_hat_[i] == y[i] as i32) as i32;
        }
        return correct as f64 / (y_hat_.nrows() as f64);
    }

    pub fn convert_prob_into_class(&self, probs: DVector<f64>) -> DVector<i32> {
        let mut probs_ = DVector::zeros(probs.nrows());
        for i in 0..probs.nrows() {
            probs_[i] = probs[i].round() as i32;
        }
        return probs_;
    }
}

