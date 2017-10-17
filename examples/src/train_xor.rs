extern crate dynet;

use std::error::Error;
use std::result::Result;
// use std::path::Path;
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

fn run() -> Result<(), Box(Error)> {
    let m = ParameterCollection::new();

    const ITERATIONS: u32 = 30;

    const HIDDEN_SIZE: u32 = 30;
    let p_W = m.add_parameters({HIDDEN_SIZE, 2});
    let p_b = m.add_parameters({HIDDEN_SIZE});
    let p_V = m.add_parameters({1, HIDDEN_SIZE});
    let p_a = m.add_parameters({1});

    let cg = ComputationGraph::new();
    let W = parameter(cg, p_W);
    let b = parameter(cg, p_b);
    let V = parameter(cg, p_V);
    let a = parameter(cg, p_a);

    let x_values: Vec<f32> = vec![2];
    let x = input(cg, {2}, &x_values);
    let y_value: f32 = 0;
    let y = input(cg, &y_value);

    let h = tanh(W*x + b);
    let y_pred = V*h + a;
    let loss_expr = squared_distance(y_pred, y);

    for iter in 0..ITERATIONS {
        let mut loss: f64 = 0;
        for mi in 0..4 {
            let x1 = mi % 2;
            let x2 = (mi / 2) % 2;
            x_values[0] = if x1 { 1 } else { -1 };
            x_values[1] = if x2 { 1 } else { -1 };
            y_value = if x1 != x2 { 1 } else { -1 };
            loss += as_scalar(cg.forward(loss_expr));
            cg.backward(loss_expr);
            // sgd.update();
        }
        loss /= 4;
        println!("E = {}", loss);
    }
    Ok(())
}

/*

#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

#include <iostream>

using namespace std;
using namespace dynet;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  const unsigned ITERATIONS = 30;

  // ParameterCollection (all the model parameters).
  ParameterCollection m;
  SimpleSGDTrainer sgd(m);

  const unsigned HIDDEN_SIZE = 8;
  Parameter p_W = m.add_parameters({HIDDEN_SIZE, 2});
  Parameter p_b = m.add_parameters({HIDDEN_SIZE});
  Parameter p_V = m.add_parameters({1, HIDDEN_SIZE});
  Parameter p_a = m.add_parameters({1});
  if (argc == 2) {
    // Load the model and parameters from file if given.
    TextFileLoader loader(argv[1]);
    loader.populate(m);
  }

  // Static declaration of the computation graph.
  ComputationGraph cg;
  Expression W = parameter(cg, p_W);
  Expression b = parameter(cg, p_b);
  Expression V = parameter(cg, p_V);
  Expression a = parameter(cg, p_a);

  // Set x_values to change the inputs to the network.
  vector<dynet::real> x_values(2);
  Expression x = input(cg, {2}, &x_values);
  dynet::real y_value;  // Set y_value to change the target output.
  Expression y = input(cg, &y_value);

  Expression h = tanh(W*x + b);
  Expression y_pred = V*h + a;
  Expression loss_expr = squared_distance(y_pred, y);

  // Show the computation graph, just for fun.
  cg.print_graphviz();

  // Train the parameters.
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    double loss = 0;
    for (unsigned mi = 0; mi < 4; ++mi) {
      bool x1 = mi % 2;
      bool x2 = (mi / 2) % 2;
      x_values[0] = x1 ? 1 : -1;
      x_values[1] = x2 ? 1 : -1;
      y_value = (x1 != x2) ? 1 : -1;
      loss += as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      sgd.update();
    }
    loss /= 4;
    cerr << "E = " << loss << endl;
  }

  // Output the model and parameter objects to a file.
  TextFileSaver saver("xor.model");
  saver.save(m);
}

*/
