#![allow(non_snake_case)]
#![allow(unused_assignments)]
#![allow(unused_variables)]

extern crate dynet;

use std::collections::HashMap;
use std::error::Error;
use std::result::Result;
use std::process::exit;

use dynet::*;

struct BiRNNBuilder<T: RNNBuilder> {
    builders: Vec<(T, T)>,
}

impl<T: RNNBuilder> BiRNNBuilder<T> {
    pub fn new(m: &mut ParameterCollection, layers: u32, input_dim: u32, hidden_dim: u32) -> Self {
        assert!(layers > 0);
        let mut builder_layers = Vec::<(T, T)>::with_capacity(layers as usize);

        let f = T::new(1, input_dim, hidden_dim / 2, m);
        let b = T::new(1, input_dim, hidden_dim / 2, m);
        builder_layers.push((f, b));

        for _ in 0..layers - 1 {
            let f = T::new(1, hidden_dim, hidden_dim / 2, m);
            let b = T::new(1, hidden_dim, hidden_dim / 2, m);
            builder_layers.push((f, b));
        }
        BiRNNBuilder { builders: builder_layers }
    }

    pub fn new_graph(&mut self, cg: &mut ComputationGraph, update: bool) {
        for &mut (ref mut f, ref mut b) in self.builders.iter_mut() {
            f.new_graph(cg, update);
            b.new_graph(cg, update);
        }
    }

    pub fn add_inputs(&mut self, xs: &Vec<Expression>) -> Vec<Expression> {
        let len = xs.len();
        let mut buf = Vec::<Expression>::with_capacity(len);
        let mut out = Vec::<Expression>::with_capacity(len);
        for (layer, &mut (ref mut f, ref mut b)) in self.builders.iter_mut().enumerate() {
            f.start_new_sequence(None);
            b.start_new_sequence(None);
            {
                let seq = if layer > 0 { &out } else { &xs };
                let mut f_hs = Vec::<Expression>::with_capacity(len);
                let mut b_hs = Vec::<Expression>::with_capacity(len);
                for i in 0..len {
                    f_hs.push(f.add_input(&seq[i]));
                    b_hs.push(b.add_input(&seq[len - i - 1]));
                }
                b_hs.reverse();
                buf.extend(f_hs.into_iter().zip(b_hs.into_iter()).map(|(f_h, b_h)| {
                    concatenate(&vec![f_h, b_h], 0)
                }));
            }
            out.clear();
            out.append(&mut buf);
        }
        out
    }
}


struct Tagger {
    p_lookup_W: LookupParameter,
    bilstm_builder: BiRNNBuilder<VanillaLSTMBuilder>,
    p_W1: Parameter,
    p_b1: Parameter,
    p_W2: Parameter,
    p_b2: Parameter,
    param_vars: Vec<Expression>,
}

impl Tagger {
    pub fn new(
        pc: &mut ParameterCollection,
        vocab_size: u32,
        embed_size: u32,
        n_layers: u32,
        hidden_size: u32,
        out_size: u32,
        dropout: f32,
    ) -> Self {
        Tagger {
            p_lookup_W: pc.add_lookup_parameters(
                vocab_size,
                &Dim::from_slice(&[embed_size as u64]),
            ),
            bilstm_builder: BiRNNBuilder::new(pc, n_layers, embed_size, hidden_size),
            p_W1: pc.add_parameters(&Dim::from_slice(&[100u64, hidden_size as u64])),
            p_b1: pc.add_parameters(&Dim::from_slice(&[100u64])),
            p_W2: pc.add_parameters(&Dim::from_slice(&[out_size as u64, 100u64])),
            p_b2: pc.add_parameters(&Dim::from_slice(&[out_size as u64])),
            param_vars: Vec::with_capacity(4),
        }
    }

    pub fn new_graph(&mut self, cg: &mut ComputationGraph, update: bool) {
        self.param_vars.clear();
        self.param_vars.push(parameter(cg, &mut self.p_W1));
        self.param_vars.push(parameter(cg, &mut self.p_b1));
        self.param_vars.push(parameter(cg, &mut self.p_W2));
        self.param_vars.push(parameter(cg, &mut self.p_b2));
        self.bilstm_builder.new_graph(cg, update);
    }

    pub fn run(&mut self, cg: &mut ComputationGraph, words: &Vec<u32>) -> Vec<Expression> {
        let xs = words
            .iter()
            .map(|&x| lookup(cg, &mut self.p_lookup_W, x))
            .collect();
        let hs = self.bilstm_builder.add_inputs(&xs);

        let W1 = &self.param_vars[0];
        let b1 = &self.param_vars[1];
        let W2 = &self.param_vars[2];
        let b2 = &self.param_vars[3];
        hs.iter().map(|h| W2 * tanh(&(W1 * h + b1)) + b2).collect()
    }
}

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

    const ITERATIONS: u32 = 50;

    let mut m = ParameterCollection::new();
    let mut sgd = SimpleSGDTrainer::new(&mut m, 0.1);

    let mut tagger = Tagger::new(&mut m, 512, 100, 3, 400, 48, 0.5);

    let sentence = "Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .";
    let tags = vec![
        "NNP",
        "NNP",
        ",",
        "CD",
        "NNS",
        "JJ",
        ",",
        "MD",
        "VB",
        "DT",
        "NN",
        "IN",
        "DT",
        "JJ",
        "NN",
        "NNP",
        "CD",
        ".",
    ];

    let mut w2i: HashMap<String, u32> = HashMap::new();
    let mut t2i: HashMap<String, u32> = HashMap::new();

    let mut counter = 0u32;
    let word_ids: Vec<u32> = sentence
        .split(" ")
        .map(|w| {
            *w2i.entry(w.to_lowercase()).or_insert_with(|| {
                let id = counter;
                counter += 1;
                id
            })
        })
        .collect();
    let mut counter = 0u32;
    let tag_ids: Vec<u32> = tags.iter()
        .map(|t| {
            *t2i.entry(t.to_lowercase()).or_insert_with(|| {
                let id = counter;
                counter += 1;
                id
            })
        })
        .collect();

    for i in 0..ITERATIONS {
        let mut cg = ComputationGraph::new();
        tagger.new_graph(&mut cg, true);
        let ys = tagger.run(&mut cg, &word_ids);
        let loss = ys.iter()
            .zip(tag_ids.iter())
            .map(|(y, t)| pickneglogsoftmax(y, *t))
            .sum();
        println!("loss: {}", as_scalar(&cg.forward(&loss)));
        cg.backward(&loss);
        sgd.update();
    }
    Ok(())
}
