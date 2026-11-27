//! Stage 1 of the mixture of experts (src/moe.rs): collect positions per expert from self-play of
//! the reference net.
//!
//!     moe_collect --types nets/moe/types.txt --out nets/moe/positions --per-type 1000000 --threads 12
//!
//! Plays greedy 1-ply self-play games with the game types' reference net (backgammon and
//! nackgammon starts alternately, until the bearoff database or the end of the game), classifies
//! every position and appends it to the set of each of its experts (its game types, or generic),
//! skipping positions the set already has, until every set has --per-type positions (sets that
//! are full take no more). `--sample P` keeps each position only with probability P (spreads a
//! small set over many games); `--exclude DIR` skips every position in DIR/*.pos (e.g. keeps the
//! measuring set out of the training set). Files: OUT/<expert>.pos, 52 bytes per position: the checker counts of
//! the player on roll (points 0-25 in its own numbering: 0 borne off, 25 the bar), then the
//! other player's; OUT/counts.txt.
use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use backgammon_rust::bearoff::BearoffDb;
use backgammon_rust::encoding::MAX_INPUTS;
use backgammon_rust::game::{opening_roll, Dice, Variant};
use backgammon_rust::moe::{experts_in, GameTypes};
use backgammon_rust::move_generator::MoveGenerator;
use backgammon_rust::{EnginePosition, Player, Position};

fn checkers(position: &EnginePosition) -> [u8; 52] {
    let on_roll = position.get_turn();
    let mut c = [0u8; 52];
    for (block, player) in [on_roll, on_roll.other_player()].into_iter().enumerate() {
        for point in 0..26 {
            c[block * 26 + point] = position.count(player, point);
        }
    }
    c
}

fn key(c: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    c.hash(&mut h);
    h.finish()
}

fn finished(bearoff: Option<&BearoffDb>, p: &EnginePosition) -> bool {
    p.count(Player::Me, 0) == 15 || p.count(Player::Opponent, 0) == 15
        || bearoff.is_some_and(|db| db.win_probability(p).is_some())
}

struct Sets {
    writers: Vec<BufWriter<File>>,
    seen: Vec<HashSet<u64>>,
    counts: Vec<usize>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut types_path, mut out) = (PathBuf::from("nets/moe/types.txt"), PathBuf::from("nets/moe/positions"));
    let (mut per_type, mut threads, mut seed, mut bearoff_path) = (1_000_000usize, 12usize, 31u64, None::<PathBuf>);
    let (mut sample, mut exclude) = (1.0f64, None::<PathBuf>);
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--types" => types_path = v.into(),
            "--out" => out = v.into(),
            "--per-type" => per_type = v.parse().expect("--per-type N"),
            "--threads" => threads = v.parse().expect("--threads N"),
            "--seed" => seed = v.parse().expect("--seed N"),
            "--bearoff" => bearoff_path = if v == "none" { None } else { Some(v.into()) },
            "--sample" => sample = v.parse().expect("--sample P"),
            "--exclude" => exclude = Some(v.into()),
            other => panic!("unknown option {other}"),
        }
        i += 2;
    }
    let types = GameTypes::load(&types_path).unwrap_or_else(|e| panic!("{e}"));
    let names = types.names();
    let n = names.len();
    let bearoff = bearoff_path.as_ref().map(|p| BearoffDb::open(p).unwrap_or_else(|e| panic!("{e}")));
    // Positions of another collection (e.g. the measuring set) that this one must not contain.
    let mut excluded = HashSet::new();
    if let Some(dir) = &exclude {
        for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|x| x == "pos") {
                for c in std::fs::read(&path).unwrap().chunks_exact(52) {
                    excluded.insert(key(c));
                }
            }
        }
        eprintln!("excluding {} positions of {}", excluded.len(), dir.display());
    }
    std::fs::create_dir_all(&out).unwrap();
    let sets = Mutex::new(Sets {
        writers: names.iter().map(|nm| BufWriter::new(File::create(out.join(format!("{nm}.pos"))).unwrap())).collect(),
        seen: vec![HashSet::new(); n],
        counts: vec![0; n],
    });
    let (next_game, done) = (AtomicU64::new(0), AtomicBool::new(false));
    let started = Instant::now();

    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (types, sets, next_game, done, bearoff, names, excluded) =
                (&types, &sets, &next_game, &done, bearoff.as_ref(), &names, &excluded);
            scope.spawn(move || {
                let mut acts = types.reference.new_activations();
                let mut generator = MoveGenerator::<EnginePosition>::new();
                let mut kept: Vec<(u16, [u8; 52])> = Vec::new();
                while !done.load(Ordering::Relaxed) {
                    let game = next_game.fetch_add(1, Ordering::Relaxed);
                    let mut dice = Dice::new(seed ^ game.wrapping_mul(0xd1b54a32d192ed03));
                    let mut keep = Dice::new(seed ^ 0x6b65_6570 ^ game.wrapping_mul(0x9e3779b97f4a7c15));
                    let variant = if game % 2 == 0 { Variant::Backgammon } else { Variant::Nackgammon };
                    let mut position = variant.starting_position();
                    let (starter, mut d1, mut d2) = opening_roll(&mut dice);
                    position.set_turn(starter);
                    kept.clear();
                    loop {
                        if sample >= 1.0 || ((keep.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < sample {
                            kept.push((types.classify(&position, &mut acts), checkers(&position)));
                        }
                        let next = generator
                            .best_position(&position, d1, d2, |child| {
                                let mut input = [0.0f32; MAX_INPUTS];
                                types.reference.encode(child, &mut input);
                                1.0 - types.reference.forward(&input) as f64
                            })
                            .map(|(p, _)| p)
                            .unwrap_or_else(|| { let mut p = position.clone(); p.switch_turn(); p });
                        if finished(bearoff, &next) {
                            break;
                        }
                        position = next;
                        d1 = dice.roll();
                        d2 = dice.roll();
                    }
                    let mut s = sets.lock().unwrap();
                    let Sets { writers, seen, counts } = &mut *s;
                    for (mask, c) in &kept {
                        let key = key(c);
                        if excluded.contains(&key) {
                            continue;
                        }
                        for e in experts_in(*mask) {
                            if counts[e] < per_type && seen[e].insert(key) {
                                writers[e].write_all(c).unwrap();
                                counts[e] += 1;
                            }
                        }
                    }
                    if (game + 1) % 20000 == 0 {
                        eprintln!("{} games, {:.0} s: {}", game + 1, started.elapsed().as_secs_f64(),
                                  names.iter().zip(counts.iter()).map(|(nm, c)| format!("{nm} {c}")).collect::<Vec<_>>().join(", "));
                    }
                    if counts.iter().all(|&c| c >= per_type) {
                        done.store(true, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    let s = sets.into_inner().unwrap();
    let games = next_game.load(Ordering::Relaxed);
    let mut summary = format!("{} games of {} (reference net), {:.0} s\n", games, types.reference.description(),
                              started.elapsed().as_secs_f64());
    for (nm, (mut w, c)) in names.iter().zip(s.writers.into_iter().zip(s.counts)) {
        w.flush().unwrap();
        summary += &format!("{nm} {c}\n");
    }
    std::fs::write(out.join("counts.txt"), &summary).unwrap();
    eprint!("{summary}");
}
