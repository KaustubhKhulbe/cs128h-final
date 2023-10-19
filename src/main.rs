use nalgebra::DVector;

use crate::utils::neural_net::NeuralNetwork;

mod utils;
fn main() {
    println!("Hello, world!");
    let mut network = NeuralNetwork::new(DVector::from(vec![1.0, 2.0, 3.0]));
    // network.add_layer(5, utils::activation::Activation::Relu);
    // network.add_layer(10, utils::activation::Activation::Relu);
    // network.add_layer(19, utils::activation::Activation::Relu);
    // network.add_layer(12, utils::activation::Activation::Relu);
    // network.add_layer(1, utils::activation::Activation::Sigmoid);
    // network.forward_propogation();
    // network.print_network();
    let y_hat = DVector::from(vec![0.49, 0.99999999, 0.000001, 0.000001, 0.000001]);
    let y = DVector::from(vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    let c = network.get_cost_val(&y_hat, &y);
    let a = network.get_accuracy_val(y_hat, y);
    println!("{}", a);
}
