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
}
