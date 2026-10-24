#![allow(warnings)]
//! HyperVerITAS v2 crop, one binary for every supported PCS.
//!
//!   hv2_crop <brakedown|basefold|basefold_fri|zeromorph> <size> [last_size]
//!
//! Environment: HV2_HASH=tensor|dense (default tensor), HV2_TAMPER=range|crop|hash (negative tests).

use std::env;

use hyperveritas_impl::{
    types::{BasefoldFri, Twenty},
    v2,
};
use plonkish_backend::{
    pcs::{
        multilinear::{Basefold, MultilinearBrakedown, ZeromorphFri},
        univariate::Fri,
    },
    util::{code::BrakedownSpec6, ff_255::ft127::Ft127, hash::Blake2s, new_fields::Mersenne127},
};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: hv2_crop <brakedown|basefold|basefold_fri|zeromorph> <size> [last_size]");
        std::process::exit(2);
    }
    let first = args[2].parse::<usize>().unwrap();
    let last = if args.len() > 3 { args[3].parse::<usize>().unwrap() } else { first };
    let ok = match args[1].as_str() {
        "brakedown" => v2::run_range::<Mersenne127, MultilinearBrakedown<Mersenne127, Blake2s, BrakedownSpec6>>(first, last),
        "basefold" => v2::run_range::<Mersenne127, Basefold<Mersenne127, Blake2s, Twenty>>(first, last),
        "basefold_fri" => v2::run_range::<Mersenne127, Basefold<Mersenne127, Blake2s, BasefoldFri>>(first, last),
        "zeromorph" => v2::run_range::<Ft127, ZeromorphFri<Fri<Ft127, Blake2s>>>(first, last),
        other => {
            eprintln!("unknown pcs {other}");
            std::process::exit(2);
        }
    };
    if !ok {
        std::process::exit(1);
    }
}
