//! Makes the evaluation copy of a trained net in lower precision (see src/eval_net.rs).
//!
//!   cargo run --release --bin make_eval_net -- [--precision f16|f16acc32|i8|f32] nets/td.bgnet nets/td-eval.bgnet
//!
//! i8 uses the fixed hidden-layer input scale of quantization-aware training
//! (`td_training::I8_HIDDEN_SCALE`).
use std::path::Path;

use backgammon_rust::eval_net::{EvalNet, Kind, INPUT_SCALE};
use backgammon_rust::td_training::I8_HIDDEN_SCALE;
use backgammon_rust::NeuralNet;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut kind = Kind::F16;
    if let Some(i) = args.iter().position(|a| a == "--precision") {
        let name = args.get(i + 1).expect("--precision needs a value").clone();
        kind = Kind::from_name(&name).unwrap_or_else(|| panic!("precision f32, f16, f16acc32 or i8, not {}", name));
        args.drain(i..i + 2);
    }
    let [input, output] = args.as_slice() else {
        panic!("usage: make_eval_net [--precision f16] INPUT OUTPUT");
    };
    let net = NeuralNet::load(Path::new(input)).unwrap_or_else(|e| panic!("{}", e));
    let n_hidden = net.sizes().len() - 2;
    let mut scales = vec![I8_HIDDEN_SCALE; n_hidden];
    scales[0] = INPUT_SCALE;
    let kinds = vec![kind; n_hidden];
    EvalNet::save_net(&net, &kinds, &scales, Path::new(output)).unwrap_or_else(|e| panic!("{}", e));
    let eval = EvalNet::load(Path::new(output)).unwrap_or_else(|e| panic!("{}", e));
    println!("{} -> {}: {}", input, output, eval.description());
}
