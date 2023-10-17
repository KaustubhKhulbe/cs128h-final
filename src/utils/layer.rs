use std::ops::Mul;
use nalgebra::{DVector, DMatrix};
use super::activation::Activation;

pub struct Layer {
    weights_matrix_: DMatrix<f64>,
    bias_vector_: DVector<f64>,
    nodes_: DVector<f64>, // previous layer values
    activation_function_: Activation,
    computed_: DVector<f64>
}

impl Layer {
    pub fn new(num_layers: usize, num_prev_layers: usize, activation: Activation, previous_layer: DVector<f64>) -> Self{
        let weights = DMatrix::<f64>::new_random(num_layers, num_prev_layers);
        let bias = DVector::<f64>::new_random(num_layers);
        Layer {
            weights_matrix_: weights,
            bias_vector_: bias,
            nodes_: previous_layer,
            activation_function_: activation,
            computed_: DVector::<f64>::zeros(num_layers),
        }
    }

    /// z = W /cdot l_[n-1] + b
    /// a = g(z)
    pub fn compute(mut self) {
        let z = self.weights_matrix_.mul(&self.nodes_) + self.bias_vector_;
        
        self.computed_ = match self.activation_function_ {
            Activation::Relu => z.map(|x: f64| {Activation::relu(x)}),
            Activation::Sigmoid => z.map(|x: f64| {Activation::sigmoid(x)}),
            Activation::Tanx => todo!(),
        };
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::utils::activation::Activation;

    use super::Layer;

    #[test]
    fn initialize() {
        let prev_layer = DVector::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let l = Layer::new(5, prev_layer.nrows(), Activation::Relu, prev_layer);
        assert_eq!(l.weights_matrix_.shape(), (5, 5));
        assert_eq!(l.bias_vector_.shape(), (5, 1));
        assert_eq!(l.computed_.shape(), (5, 1));
        assert_eq!(l.nodes_.shape(), (5, 1));

        let prev_layer = DVector::from(vec![0.0, 1.0, 2.0, 3.0]);
        let l = Layer::new(5, prev_layer.nrows(), Activation::Relu, prev_layer);
        assert_eq!(l.weights_matrix_.shape(), (5, 4));
        assert_eq!(l.bias_vector_.shape(), (5, 1));
        assert_eq!(l.computed_.shape(), (5, 1));
        assert_eq!(l.nodes_.shape(), (4, 1));
    }
}
