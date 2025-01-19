mod network;
use network::*;

fn main() {
    let mut network = Network::new();
    let epoch = 100000;

    for _ in 0 .. epoch {
        network.train([0.0, 0.0], 0.0);
        network.train([1.0, 0.0], 1.0);
        network.train([0.0, 1.0], 1.0);
        network.train([1.0, 1.0], 0.0);
    }

    println!("Trained {} epochs on XOR challenge", epoch);
    println!("<0, 0> = {}", network.evaluate([0.0, 0.0]).round());
    println!("<1, 0> = {}", network.evaluate([1.0, 0.0]).round());
    println!("<0, 1> = {}", network.evaluate([0.0, 1.0]).round());
    println!("<1, 1> = {}", network.evaluate([1.0, 1.0]).round());

}
