use std::ops::Mul;
use nalgebra::{DVector, DMatrix};
use super::activation::Activation;

pub struct Layer {
    pub weights_matrix_: DMatrix<f64>,
    pub bias_vector_: DVector<f64>,
    pub nodes_: DVector<f64>, // previous layer values
    pub activation_function_: Activation,
    pub computed_: DVector<f64>,
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

    pub fn input_layer(vec: DVector<f64>) -> Self{
        Layer {
            weights_matrix_: DMatrix::zeros(0, 0),
            bias_vector_: DVector::zeros(0),
            nodes_: DVector::zeros(0),
            activation_function_: Activation::Input,
            computed_: vec,
        }
    }

    /// z = W /cdot l_[n-1] + b
    /// a = g(z)
    pub fn single_layer_forward_propogation(&mut self) {
        let z = (&self.weights_matrix_).mul(&self.nodes_) + &self.bias_vector_;
        
        self.computed_ = match self.activation_function_ {
            Activation::Relu => z.map(|x: f64| {Activation::relu(x)}),
            Activation::Sigmoid => z.map(|x: f64| {Activation::sigmoid(x)}),
            Activation::Tanx => todo!(),
            Activation::Input => todo!(),
        };
    }

    pub fn linear_back(&mut self) {
        let a_prev = &self.nodes_;
        let w = &self.weights_matrix_;
        let b = &self.bias_vector_;

        //let linear_cache = (a_prev, w, b);

        println!("try");
        // println!("computed{:?}", &self.computed_.shape());
        // println!("w{:?}", w.shape());
        // println!("a_prev{:?}", a_prev.shape());
        // println!("computed transpose{:?}", &self.computed_.transpose().shape());
        let m = a_prev.shape().1 as f64;
        let dw = (1.0 / m) * (&self.computed_.dot(&a_prev));
        // println!("finished dw");
        let db = (1.0 / m) * &self.computed_.sum();
        //let w_vec = w.clone().into_owned().reshape_generic(DMatrix::nrows(&w) * DMatrix::ncols(&w), 1);
        // println!("w_vec{:?}", w_vec.shape());
        // let da_prev = w_vec.dot(&self.computed_);
        // println!("finished a_prev");

        // let m = a_prev.shape().1 as f64;
        // let dw = (1.0 / m) * (&self.computed_.dot(&a_prev));
        // let db = (1.0 / m) * &self.computed_.sum();
        // let da_prev = w.dot(&self.computed_);

        self.weights_matrix_ = self.weights_matrix_.add_scalar(&dw*(-0.1));
        self.bias_vector_ = self.bias_vector_.add_scalar(&db*(-0.1));
        //self.nodes_ = self.nodes_.add_scalar(&da_prev*(-0.1));
    }

    pub fn single_layer_back_propogation(&mut self) {
        let _ = &self.linear_back();

        // let c = &self.computed_;
        // self.computed_ = match self.activation_function_ {
        //     Activation::Relu => c.map(|x: f64| {Activation::relu_prime(x)}),
        //     Activation::Sigmoid => c.map(|x: f64| {Activation::sigmoid_prime(x)}),
        //     Activation::Tanx => todo!(),
        //     Activation::Input => todo!(),
        // };

        let z = (&self.weights_matrix_).mul(&self.nodes_) + &self.bias_vector_;
        self.computed_ = match self.activation_function_ {
            Activation::Relu => z.map(|x: f64| {Activation::relu_prime(x)}),
            Activation::Sigmoid => z.map(|x: f64| {Activation::sigmoid_prime(x)}),
            Activation::Tanx => todo!(),
            Activation::Input => todo!(),
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
