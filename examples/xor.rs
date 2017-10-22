#![allow(non_snake_case)]
#![allow(unused_assignments)]
#![allow(unused_variables)]

extern crate dynet;

use std::error::Error;
use std::result::Result;
use std::process::exit;

use dynet::*;

fn main() {
    exit(match run() {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}

fn run() -> Result<(), Box<Error>> {
    dynet::initialize();

    const ITERATIONS: u32 = 30;

    let mut m = ParameterCollection::new();
    let mut sgd = SimpleSGDTrainer::new(&mut m, 0.1);

    const HIDDEN_SIZE: u64 = 30;
    let mut p_W = m.add_parameters(&Dim::from_slice(&[HIDDEN_SIZE, 2]));
    let mut p_b = m.add_parameters(&Dim::from_slice(&[HIDDEN_SIZE]));
    let mut p_V = m.add_parameters(&Dim::from_slice(&[1, HIDDEN_SIZE]));
    let mut p_a = m.add_parameters(&Dim::from_slice(&[1u64]));

    let mut cg = ComputationGraph::new();
    let W = parameter(&mut cg, &mut p_W);
    let b = parameter(&mut cg, &mut p_b);
    let V = parameter(&mut cg, &mut p_V);
    let a = parameter(&mut cg, &mut p_a);

    let mut x_values: Vec<f32> = vec![0f32, 0f32];
    let x = input_vector(&mut cg, &Dim::from_slice(&[2u64]), &x_values);
    let mut y_value: f32 = 0f32;
    let y = input_scalar(&mut cg, y_value);

    let h = tanh(&(W * x + b));
    let y_pred = V * h + a;
    let loss_expr = squared_distance(&y_pred, &y);

    for iter in 0..ITERATIONS {
        let mut loss: f64 = 0f64;
        for mi in 0..4 {
            let x1 = mi % 2;
            let x2 = (mi / 2) % 2;
            x_values[0] = if x1 == 1 { 1f32 } else { -1f32 };
            x_values[1] = if x2 == 1 { 1f32 } else { -1f32 };
            y_value = if x1 != x2 { 1f32 } else { -1f32 };
            loss += as_scalar(&cg.forward(&loss_expr)) as f64;
            cg.backward(&loss_expr);
            sgd.update();
        }
        loss /= 4f64;
        println!("E = {}", loss);
    }
    Ok(())
}
