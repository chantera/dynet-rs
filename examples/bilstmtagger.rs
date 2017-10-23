#![allow(non_snake_case)]
#![allow(unused_assignments)]
#![allow(unused_variables)]

extern crate dynet;

use std::error::Error;
use std::result::Result;
use std::process::exit;

use dynet::*;

struct BiRNNBuilder<T: RNNBuilder> {
    builders: Vec<(T, T)>,
}

impl<T: RNNBuilder> BiRNNBuilder<T> {
    pub fn new(m: &mut ParameterCollection, layers: u32, input_dim: u32, hidden_dim: u32) -> Self {
        // assert layers > 0
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
            let res = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| f.start_new_sequence(None)),
            );
            // match res {
            //     Err(e) => {
            //         eprintln!("hoge");
            //         eprintln!("ERROR: {:?}", e);
            //         std::panic::resume_unwind(e);
            //     }
            //     Ok(t) => t,
            // };
            f.start_new_sequence(None);
            b.start_new_sequence(None);
            {
                let seq = if layer > 0 { &out } else { &xs };
                for i in 0..len {
                    buf.push(concatenate(
                        &vec![f.add_input(&seq[i]), b.add_input(&seq[len - i - 1])],
                        0,
                    ));
                }
            }
            out.clear();
            out.append(&mut buf);
        }
        out
    }

    /*
    fwR.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) fw[i] = fwR.add_input(seqE[i]);
    bwR.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) bw[i] = bwR.add_input(seqE[seq.size()-i-1]);

    for(size_t i = 0; i < seq.size(); ++i) zs[i] = T * concatenate({fw[i], bw[seq.size()-i-1]});
    fwR2.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) fw[i] = fwR2.add_input(zs[i]);
    bwR2.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) bw[i] = bwR2.add_input(zs[seq.size()-i-1]);
    }


    vector<Expression> seqE(seq.size()), fw(seq.size()), bw(seq.size()), zs(seq.size()), losses(seq.size());
    for(size_t i = 0; i < seq.size(); ++i) seqE[i] = lookup(cg, E, seq[i]);

    for(size_t i = 0; i < seq.size(); ++i) zs[i] = W * concatenate({fw[i], bw[seq.size()-i-1]});
    for(size_t i = 0; i < seq.size(); ++i) losses[i] = pickneglogsoftmax(zs[i], Y[i]);
    return sum(losses);
    */
}


struct Tagger {
    p_lookup_W: LookupParameter,
    bilstm_builder: BiRNNBuilder<VanillaLSTMBuilder>,
    p_W1: Parameter,
    p_b1: Parameter,
    p_W2: Parameter,
    p_b2: Parameter,
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
        // let mut bilstm_builders: Vec<(VanillaLSTMBuilder, VanillaLSTMBuilder)> =
        //     Vec::with_capacity(n_layers);
        // for i in 0..n_layers {
        //     let mut f_bilstm
        //     bilstm_builders.push((
        //             ))
        // }
        Tagger {
            p_lookup_W: pc.add_lookup_parameters(
                vocab_size,
                &Dim::from_slice(&[embed_size as u64]),
            ),
            bilstm_builder: BiRNNBuilder::new(pc, n_layers, embed_size, hidden_size),
            p_W1: pc.add_parameters(&Dim::from_slice(&[100u64, (hidden_size * 2) as u64])),
            p_b1: pc.add_parameters(&Dim::from_slice(&[100u64])),
            p_W2: pc.add_parameters(&Dim::from_slice(&[out_size as u64, 100u64])),
            p_b2: pc.add_parameters(&Dim::from_slice(&[out_size as u64])),
        }
    }

    pub fn new_graph(&mut self, cg: &mut ComputationGraph, update: bool) {
        self.bilstm_builder.new_graph(cg, update);
    }

    pub fn run(&mut self, cg: &mut ComputationGraph, words: &Vec<u32>) -> Vec<Expression> {
        let W1 = parameter(cg, &mut self.p_W1);
        let b1 = parameter(cg, &mut self.p_b1);
        let W2 = parameter(cg, &mut self.p_W2);
        let b2 = parameter(cg, &mut self.p_b2);

        let xs = words
            .iter()
            .map(|&x| lookup(cg, &mut self.p_lookup_W, x))
            .collect();
        let hs = self.bilstm_builder.add_inputs(&xs);
        hs.iter()
            .map(|h| &W2 * tanh(&(&W1 * h + &b1)) + &b2)
            .collect()
    }
    //
    //

    /*
    builder = dy.VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc)
	for layer, params in enumerate(builder.get_parameters()):
		W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + (lstm_hiddens if layer >0 else input_dims))
		W_h, W_x = W[:,:lstm_hiddens], W[:,lstm_hiddens:]
		params[0].set_value(np.concatenate([W_x]*4, 0))
		params[1].set_value(np.concatenate([W_h]*4, 0))
		b = np.zeros(4*lstm_hiddens, dtype=np.float32)
		b[lstm_hiddens:2*lstm_hiddens] = -1.0
		params[2].set_value(b)
	return builder
*/
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

    const ITERATIONS: u32 = 30;

    let mut m = ParameterCollection::new();

    let mut tagger = Tagger::new(&mut m, 512, 100, 3, 200, 48, 0.5);

    let sentence = vec![1u32, 2u32, 1u32, 5u32, 3u32, 2u32];
    for i in 0..ITERATIONS {
        let mut cg = ComputationGraph::new();
        tagger.new_graph(&mut cg, true);
        tagger.run(&mut cg, &sentence);
    }
    Ok(())
}
