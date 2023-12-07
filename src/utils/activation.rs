pub enum Activation {
    Relu, Tanx, Sigmoid, Input // etc
}

impl Activation {
    pub fn relu(x: f64) -> f64{
        x.max(0.0)
    }

    pub fn sigmoid(x: f64) -> f64{
        1.0 / (1.0 + (-x).exp())
    }

    pub fn relu_prime(x: f64) -> f64{
        match x > 0.0 {
            true => 1.0,
            false => 0.0
        }
    }

    pub fn sigmoid_prime(x: f64) -> f64{
        Self::sigmoid(x) * (1.0 - Self::sigmoid(x))
    }
}
