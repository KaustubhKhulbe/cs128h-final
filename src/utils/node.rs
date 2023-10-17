use nalgebra::{DVector, DMatrix};

use super::activation::Activation;

pub struct Layer {
    weights_matrix_: DMatrix<f64>,
    bias_vector_: DVector<f64>,
    nodes_: Vec<Node>,
    activation_function_: Activation,
}

impl Layer {
    pub fn new(num_layers: usize, num_prev_layers: usize, activation: Activation) -> Self{
        let weights = DMatrix::<f64>::new_random(num_layers, num_prev_layers);
        let bias = DVector::<f64>::new_random(num_layers);
        Layer {
            weights_matrix_: weights,
            bias_vector_: bias,
            nodes_: vec![],
            activation_function_: activation,
        }
    }
}

pub struct Node {
}