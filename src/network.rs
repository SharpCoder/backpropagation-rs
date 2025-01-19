
use rand::prelude::*;
use rand_pcg::Pcg32;

fn sigmoid(val: f32) -> f32 {
    let neg = -val;
    return 1.0f32 / (1.0f32 + neg.exp());
}

pub struct Network {
    rng: Pcg32,
    learn_rate: f32,
    weights: [f32;10],
    outputs: [f32;5]
}

impl Network {
    pub fn new() -> Self {
        // Create and seed a simple random number generator
        // so our experiments are consistent.
        let rng: Pcg32 = rand::SeedableRng::seed_from_u64(0x1337);
        
        // Populate the network with empty weights and outputs.
        // The topology of our network is 2 -> 2 -> 1.
        let mut network = Network {
            rng: rng,
            learn_rate: 0.05,
            weights: [0.0;10],
            outputs: [0.0;5],
        };

        // Generate random weights using the pre-seeded rng.
        for i in 0 .. 10 {
            network.weights[i] = network.rng.gen::<f32>() - network.rng.gen::<f32>();
        }

        return network;
    }

    pub fn evaluate(&mut self, input: [f32;2]) -> f32 {			
            self.outputs[0] = sigmoid(input[0] * self.weights[0] + input[1] * self.weights[1]);
			self.outputs[1] = sigmoid(input[0] * self.weights[2] + input[1] * self.weights[3]);
			self.outputs[2] = sigmoid(self.outputs[0] * self.weights[4] + self.outputs[1] * self.weights[5]);
			self.outputs[3] = sigmoid(self.outputs[0] * self.weights[6] + self.outputs[1] * self.weights[7]);
			self.outputs[4] = sigmoid(self.outputs[2] * self.weights[8] + self.outputs[3] * self.weights[9]);
			return self.outputs[4];
    }

    pub fn train(&mut self, input: [f32;2], target: f32) {
        let mut errors: [f32;5] = [0.0; 5];
        let output = self.evaluate(input);

        // Compute the error matrix. This operation is really just the derivative of ech
        // node output multiplied by the error of the nodes below it.
        errors[4] = output * (1.0 - output) * (target - output);
        errors[3] = self.outputs[3] * (1.0 - self.outputs[3]) * (errors[4] * self.weights[9]);
        errors[2] = self.outputs[2] * (1.0 - self.outputs[2]) * (errors[4] * self.weights[8]);
        errors[1] = self.outputs[1] * (1.0 - self.outputs[1]) * ((errors[3] * self.weights[7]) + (errors[2] * self.weights[5]));
        errors[0] = self.outputs[0] * (1.0 - self.outputs[0]) * ((errors[3] * self.weights[6]) + (errors[2] * self.weights[4]));

        // Now we compute the adjustment.
        // Each weight gets adjusted equal to the learn_rate times the 
        // value that weight was multiplied against times the error of the node which "houses" the weight.
        self.weights[9] += self.learn_rate * self.outputs[3] * errors[4];
        self.weights[8] += self.learn_rate * self.outputs[2] * errors[4];
        self.weights[7] += self.learn_rate * self.outputs[1] * errors[3];
        self.weights[6] += self.learn_rate * self.outputs[0] * errors[3];
        self.weights[5] += self.learn_rate * self.outputs[1] * errors[2];
        self.weights[4] += self.learn_rate * self.outputs[0] * errors[2];
        self.weights[3] += self.learn_rate * input[1] * errors[1];
        self.weights[2] += self.learn_rate * input[0] * errors[1];
        self.weights[1] += self.learn_rate * input[1] * errors[0];
        self.weights[0] += self.learn_rate * input[0] * errors[0];
    }
}