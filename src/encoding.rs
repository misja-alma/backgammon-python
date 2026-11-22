//! Input encodings for the neural net.
//!
//! The net always sees the position from the side to move: the first block of inputs belongs to
//! the player on roll, the second to the other player. So which player's checkers fill which
//! inputs depends on whose turn it is, and the net's output is always the winning chance of the
//! player on roll. Points are in that player's own numbering: 0 borne off, 1-24 the points, 25 the
//! bar.
//!
//! A net only works with the encoding it was trained on, so the encoding's name is stored in the
//! weight file and the net encodes its own inputs (`NeuralNet::encode`). Two encodings:
//!
//! `CountsPips` ("counts-pips-v1", 27 inputs per player):
//! - 26 checker counts (borne off, points 1-24, bar), each divided by 15;
//! - the pip count, divided by 167 (the pip count at the start).
//!
//! `TdCounts` ("td-counts-v1", 127 inputs per player): TD-Gammon's threshold units plus the plain
//! count, so the net sees both "blot / made point / spare" and the real number of checkers:
//! - for each of points 1-24 and the bar, 5 inputs for its n checkers:
//!   n ≥ 1, n ≥ 2, n ≥ 3, max(n - 3, 0) / 2, n / 15;
//! - borne off / 15;
//! - the pip count / 167.
//!
//! `TdFlags` ("td-flags-v1", 257 inputs): `TdCounts` followed by 3 flags (0 or 1):
//! - race: no contact left, i.e. every checker of each player has passed every checker of the
//!   other (a checker on the bar is always contact);
//! - nackgammon flag of the player on roll, then of the other player: not a race, and that player
//!   has 4 or more checkers in the opponent's home board (own points 19-24; the bar doesn't count).
//! The first 254 inputs are exactly `TdCounts`, so a `TdCounts` net can be extended to it
//! (`NeuralNet::extend_inputs`) with zero weights for the flags.
//!
//! `TdNack` ("td-nack-v1", 256 inputs): `TdFlags` without the race flag, i.e. `TdCounts` followed
//! by the two nackgammon flags (player on roll, other player). Also extends `TdCounts`.
use crate::position::{Player, Position};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    CountsPips,
    TdCounts,
    TdFlags,
    TdNack,
}

/// The largest input count of any encoding: input buffers are this long.
pub const MAX_INPUTS: usize = 2 * 127 + FLAGS;

/// The most inputs after the two player blocks (`TdFlags`: race, nackgammon flag ×2).
const FLAGS: usize = 3;

const CHECKER_SCALE: f32 = 1.0 / 15.0;
const PIP_SCALE: f32 = 1.0 / 167.0;
/// Inputs per point (and the bar) in `TdCounts`.
const UNITS_PER_POINT: usize = 5;

impl Encoding {
    pub const ALL: [Encoding; 4] = [Encoding::CountsPips, Encoding::TdCounts, Encoding::TdFlags, Encoding::TdNack];

    /// The name stored in weight files.
    pub fn name(self) -> &'static str {
        match self {
            Encoding::CountsPips => "counts-pips-v1",
            Encoding::TdCounts => "td-counts-v1",
            Encoding::TdFlags => "td-flags-v1",
            Encoding::TdNack => "td-nack-v1",
        }
    }

    pub fn from_name(name: &str) -> Option<Encoding> {
        Self::ALL.into_iter().find(|e| e.name() == name)
    }

    pub fn inputs_per_player(self) -> usize {
        match self {
            Encoding::CountsPips => 27,
            Encoding::TdCounts | Encoding::TdFlags | Encoding::TdNack => 25 * UNITS_PER_POINT + 2,
        }
    }

    /// Inputs after the two player blocks.
    fn n_flags(self) -> usize {
        match self {
            Encoding::CountsPips | Encoding::TdCounts => 0,
            Encoding::TdFlags => 3,
            Encoding::TdNack => 2,
        }
    }

    pub fn n_inputs(self) -> usize {
        2 * self.inputs_per_player() + self.n_flags()
    }

    /// Whether this encoding's inputs start with all of `other`'s, in the same order.
    pub fn extends(self, other: Encoding) -> bool {
        self == other || (matches!(self, Encoding::TdFlags | Encoding::TdNack) && other == Encoding::TdCounts)
    }

    /// Writes the inputs for `position` into `out[..self.n_inputs()]`.
    pub fn encode<P: Position>(self, position: &P, out: &mut [f32]) {
        let on_roll = position.get_turn();
        let per_player = self.inputs_per_player();
        for (block, player) in [on_roll, on_roll.other_player()].into_iter().enumerate() {
            let out = &mut out[block * per_player..(block + 1) * per_player];
            let mut pips = 0u32;
            match self {
                Encoding::CountsPips => {
                    for point in 0..26 {
                        let count = position.count(player, point);
                        out[point] = count as f32 * CHECKER_SCALE;
                        pips += point as u32 * count as u32;
                    }
                }
                Encoding::TdCounts | Encoding::TdFlags | Encoding::TdNack => {
                    for point in 1..26 {
                        let n = position.count(player, point);
                        pips += point as u32 * n as u32;
                        let units = &mut out[(point - 1) * UNITS_PER_POINT..point * UNITS_PER_POINT];
                        units[0] = (n >= 1) as u8 as f32;
                        units[1] = (n >= 2) as u8 as f32;
                        units[2] = (n >= 3) as u8 as f32;
                        units[3] = n.saturating_sub(3) as f32 * 0.5;
                        units[4] = n as f32 * CHECKER_SCALE;
                    }
                    out[25 * UNITS_PER_POINT] = position.count(player, 0) as f32 * CHECKER_SCALE;
                }
            }
            out[per_player - 1] = pips as f32 * PIP_SCALE;
        }
        let n_flags = self.n_flags();
        if n_flags > 0 {
            let flags = &mut out[2 * per_player..2 * per_player + n_flags];
            let race = is_race(position);
            let nack = |player| (!race && checkers_in_opponent_home(position, player) >= 4) as u8 as f32;
            if self == Encoding::TdFlags {
                flags[0] = race as u8 as f32;
            }
            flags[n_flags - 2] = nack(on_roll);
            flags[n_flags - 1] = nack(on_roll.other_player());
        }
    }
}

/// The player's rearmost occupied point in their own numbering (25 = bar), 0 if all are off.
fn rearmost<P: Position>(position: &P, player: Player) -> usize {
    (1..26).rev().find(|&p| position.count(player, p) > 0).unwrap_or(0)
}

/// No contact left: point p of one player is point 25 - p of the other, so the rearmost checkers
/// have passed each other when their points add up to 25 or less (a bar checker, point 25, is
/// always contact while the other player has a checker on the board).
pub fn is_race<P: Position>(position: &P) -> bool {
    rearmost(position, Player::Me) + rearmost(position, Player::Opponent) <= 25
}

/// The player's checkers on points 19-24 of their own numbering (the opponent's home board).
fn checkers_in_opponent_home<P: Position>(position: &P, player: Player) -> u8 {
    (19..25).map(|p| position.count(player, p)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Player;
    use crate::simple_position::SimplePosition;

    fn sample() -> SimplePosition {
        let mut p = SimplePosition::new();
        p.set_checkers(Player::Me, 6, 3).unwrap();
        p.set_checkers(Player::Me, 25, 1).unwrap();
        p.set_checkers(Player::Me, 8, 5).unwrap();
        p.set_checkers(Player::Me, 0, 6).unwrap();
        p.set_checkers(Player::Opponent, 13, 2).unwrap();
        p
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_counts_pips() {
        let mut p = sample();
        p.set_turn(Player::Me);
        let mut x = [0.0; MAX_INPUTS];
        Encoding::CountsPips.encode(&p, &mut x);
        assert!(close(x[6], 3.0 / 15.0));
        assert!(close(x[25], 1.0 / 15.0));
        assert!(close(x[26], (18.0 + 25.0 + 40.0) / 167.0));
        assert!(close(x[27 + 13], 2.0 / 15.0));
        assert!(close(x[27 + 26], 26.0 / 167.0));
    }

    #[test]
    fn test_td_counts() {
        let mut p = sample();
        p.set_turn(Player::Me);
        let mut x = [0.0; MAX_INPUTS];
        Encoding::TdCounts.encode(&p, &mut x);
        let point = |base: usize, point: usize| x[base + (point - 1) * 5..base + point * 5].to_vec();
        assert_eq!(point(0, 6), vec![1.0, 1.0, 1.0, 0.0, 3.0 * CHECKER_SCALE]);
        assert_eq!(point(0, 8), vec![1.0, 1.0, 1.0, 1.0, 5.0 * CHECKER_SCALE]);
        assert_eq!(point(0, 25), vec![1.0, 0.0, 0.0, 0.0, 1.0 * CHECKER_SCALE]);
        assert_eq!(point(0, 7), vec![0.0; 5]);
        assert!(close(x[125], 6.0 / 15.0));
        assert!(close(x[126], (18.0 + 25.0 + 40.0) / 167.0));
        assert_eq!(point(127, 13), vec![1.0, 1.0, 0.0, 0.0, 2.0 * CHECKER_SCALE]);
        assert!(close(x[127 + 126], 26.0 / 167.0));
    }

    fn position(me: &[(usize, u8)], opponent: &[(usize, u8)]) -> SimplePosition {
        let mut p = SimplePosition::new();
        for (player, layout) in [(Player::Me, me), (Player::Opponent, opponent)] {
            let mut on_board = 0;
            for &(point, count) in layout {
                p.set_checkers(player, point, count).unwrap();
                on_board += count;
            }
            p.set_checkers(player, 0, 15 - on_board).unwrap();
        }
        p
    }

    fn flags(p: &SimplePosition) -> Vec<f32> {
        let mut x = [0.0; MAX_INPUTS];
        Encoding::TdFlags.encode(p, &mut x);
        x[254..257].to_vec()
    }

    #[test]
    fn test_race_flag() {
        // Starting position: contact.
        let mut start = SimplePosition::new();
        start.setup_starting_position();
        assert!(!is_race(&start));
        // My rearmost on my 12; the opponent's 14 is my 11, still in front of me: contact.
        assert!(!is_race(&position(&[(12, 2)], &[(14, 2)])));
        // The opponent's 12 is my 13, behind my rearmost checker: passed.
        assert!(is_race(&position(&[(12, 2)], &[(12, 2)])));
        assert!(is_race(&position(&[(6, 5)], &[(6, 5)])));
        // A checker on the bar is contact.
        assert!(!is_race(&position(&[(25, 1), (6, 2)], &[(3, 2)])));
        // All borne off on one side.
        assert!(is_race(&position(&[], &[(24, 1)])));
    }

    #[test]
    fn test_nackgammon_flags() {
        // Nackgammon start, Me on roll: both have 4 checkers on 23 + 24, contact.
        let mut p = SimplePosition::new();
        p.setup_starting_position();
        for player in [Player::Me, Player::Opponent] {
            for (point, count) in [(23, 2), (13, 4), (6, 4)] {
                p.set_checkers(player, point, count).unwrap();
            }
        }
        p.set_turn(Player::Me);
        assert_eq!(flags(&p), vec![0.0, 1.0, 1.0]);
        // Backgammon start: only 2 back checkers each.
        let mut b = SimplePosition::new();
        b.setup_starting_position();
        assert_eq!(flags(&b), vec![0.0, 0.0, 0.0]);
        // 3 back checkers plus one on the bar: the bar doesn't count. On roll comes first.
        let mut q = position(&[(25, 1), (20, 3), (6, 5)], &[(19, 4), (8, 3)]);
        q.set_turn(Player::Opponent);
        assert_eq!(flags(&q), vec![0.0, 1.0, 0.0]);
        // In a race (here the opponent has borne off everything) the nackgammon flags are off.
        let r = position(&[(20, 4)], &[]);
        assert_eq!(flags(&r), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_flags_extend_td_counts() {
        let mut p = sample();
        p.set_turn(Player::Opponent);
        let (mut a, mut b) = ([0.0; MAX_INPUTS], [0.0; MAX_INPUTS]);
        Encoding::TdCounts.encode(&p, &mut a);
        Encoding::TdFlags.encode(&p, &mut b);
        assert_eq!(a[..254], b[..254]);
        assert!(Encoding::TdFlags.extends(Encoding::TdCounts));
        assert!(!Encoding::TdCounts.extends(Encoding::TdFlags));
        assert!(!Encoding::TdFlags.extends(Encoding::CountsPips));
    }

    #[test]
    fn test_td_nack_is_td_flags_without_race() {
        let mut q = position(&[(25, 1), (20, 3), (6, 5)], &[(19, 4), (8, 3)]);
        let r = position(&[(20, 4)], &[]);
        for p in [sample(), q.clone(), r] {
            let (mut f, mut n) = ([0.0; MAX_INPUTS], [0.0; MAX_INPUTS]);
            Encoding::TdFlags.encode(&p, &mut f);
            Encoding::TdNack.encode(&p, &mut n);
            assert_eq!(Encoding::TdNack.n_inputs(), 256);
            assert_eq!(f[..254], n[..254]);
            assert_eq!(f[255..257], n[254..256]);
        }
        q.set_turn(Player::Opponent);
        let mut n = [0.0; MAX_INPUTS];
        Encoding::TdNack.encode(&q, &mut n);
        assert_eq!(n[254..256], [1.0, 0.0]);
        assert!(Encoding::TdNack.extends(Encoding::TdCounts));
    }

    #[test]
    fn test_side_to_move_comes_first() {
        for encoding in Encoding::ALL {
            let half = encoding.inputs_per_player();
            let mut p = sample();
            p.set_turn(Player::Me);
            let mut me_first = [0.0; MAX_INPUTS];
            encoding.encode(&p, &mut me_first);
            p.set_turn(Player::Opponent);
            let mut opp_first = [0.0; MAX_INPUTS];
            encoding.encode(&p, &mut opp_first);
            assert_eq!(opp_first[..half], me_first[half..2 * half]);
            assert_eq!(opp_first[half..2 * half], me_first[..half]);
        }
    }

    #[test]
    fn test_names_round_trip() {
        for encoding in Encoding::ALL {
            assert_eq!(Encoding::from_name(encoding.name()), Some(encoding));
        }
        assert_eq!(Encoding::from_name("nope"), None);
    }
}
