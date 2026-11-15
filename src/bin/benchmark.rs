use std::time::Instant;
use backgammon_rust::{build_position, position::Player, RustAnalyzer};

fn main() {
    let mut me_checkers = vec![0u8; 26];
    let mut opp_checkers = vec![0u8; 26];
    me_checkers[6] = 2;
    opp_checkers[6] = 2;

    let pos = build_position(me_checkers, opp_checkers, 1, 1, None).unwrap();

    let analyzer = RustAnalyzer::new();
    let max_depth = 8;
    let runs = 10;

    let mut total = std::time::Duration::ZERO;

    for i in 0..runs {
        analyzer.clear_cache();
        let start = Instant::now();
        let equity = analyzer.equity(&pos, Player::Me, max_depth);
        let elapsed = start.elapsed();
        total += elapsed;
        println!("run {:2}: equity = {:.4}  time = {:?}", i + 1, equity, elapsed);
    }

    println!("\naverage over {} runs: {:?}", runs, total / runs);
}
