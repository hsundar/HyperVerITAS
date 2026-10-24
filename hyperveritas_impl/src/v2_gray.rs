#![allow(warnings)]
//! HyperVerITAS grayscale, redesigned protocol ("v2"), sharing the hash, LogUp-GKR range check and
//! opening machinery of `v2` (crop).
//!
//! Statement: the public grayscale image `gray` is the rounded luma of a secret RGB image whose
//! camera digest is public, i.e. for every pixel x
//!     e(x) := 50 + 100*gray(x) - (30 R(x) + 59 G(x) + 11 B(x))  lies in [0, 100].
//! Changes relative to the original `hv_gray_*` examples (same statement):
//!   * The error polynomial e is NOT committed. It is a linear combination of the three committed
//!     channels and the public gray image, so the GKR range check treats it as a fourth channel
//!     (table [0,100]) and the verifier reconstructs e~(rho) from the three channel openings and
//!     its own O(N) evaluation of gray~(rho).
//!   * No extra sumcheck for the transformation: the GKR's final point is reused.
//!   * 6 openings in total (3 channels x {hash point, GKR point}) instead of 4 committed
//!     polynomials opened at 2N-sized points plus range-check commitments.
//!
//! Negative tests: HV2_TAMPER=range|mults|gray|leaf|hash.

use std::time::Instant;

use ark_std::test_rng;
use rayon::prelude::*;

use crate::image::*;
use crate::v2::{
    absorb_meta, bytes_to_field, check_seekable, check_tensor_hash, dense_digest, dense_rta, hash_mode, lap,
    logup_gkr_prove, logup_gkr_verify, pack_bytes, tamper, HvPcs, TensorParams, Tr, HASH_ROWS, TABLE,
};

use plonkish_backend::{
    pcs::{Evaluation, PolynomialCommitmentScheme},
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{Field as myField, PrimeField},
        expression::{Expression, Query, Rotation},
        transcript::{FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite, InMemoryTranscript, TranscriptRead, TranscriptWrite},
    },
};

/// Luma weights (x100), offset and error table size: e = OFFSET + SCALE*gray - sum_c W[c]*img_c in [0, ERR_TABLE).
pub const W: [u64; 3] = [30, 59, 11];
pub const SCALE: u64 = 100;
pub const OFFSET: u64 = 50;
pub const ERR_TABLE: usize = 101;

pub struct Public<F> {
    pub nv: usize,
    pub digest: Vec<Vec<F>>,
    pub gray: Vec<u8>,
    pub tensor: Option<TensorParams<F>>,
}

pub fn setup<F: PrimeField, Pcs: HvPcs<F>>(input_size: usize) -> (Pcs::ProverParam, Pcs::VerifierParam, Public<F>) {
    println!("\nstarting setup");
    let mut t = Instant::now();
    let tensor_mode = hash_mode() == "tensor";
    if tensor_mode {
        check_tensor_hash::<F>();
    } else {
        check_seekable::<F>();
    }
    let tensor = if tensor_mode { Some(TensorParams::<F>::new(input_size)) } else { None };
    lap(&format!("setup: hash params (mode={})", hash_mode()), &mut t);

    let mut rng = test_rng();
    let nv = input_size;
    let (pp, vp) = {
        let poly_size = 1 << nv;
        let param = Pcs::setup(poly_size, 4, &mut rng).unwrap();
        Pcs::trim(&param, poly_size, 4).unwrap()
    };
    lap(&format!("setup: {} params", Pcs::NAME), &mut t);

    let orig = load_image(format!("images/Timings{}.json", input_size));
    let gray_img = load_image(format!("images/Gray{}.json", input_size));
    assert_eq!(gray_img.R.len(), 1 << nv);
    let img = [bytes_to_field::<F>(&orig.R), bytes_to_field::<F>(&orig.G), bytes_to_field::<F>(&orig.B)];
    lap("setup: load images", &mut t);

    let digest = match &tensor {
        Some(tp) => tp.digest(&img),
        None => dense_digest(&img),
    };
    lap("setup: camera digest A*img (parallel)", &mut t);
    println!("setup done!\n");
    (pp, vp, Public { nv, digest, gray: gray_img.R, tensor })
}

/// Fiat-Shamir binding of the public statement: camera digest, gray image, size.
fn absorb_public<F: PrimeField>(pb: &Public<F>, tr: &mut Tr) {
    for c in 0..3 {
        <Tr as FieldTranscript<F>>::common_field_elements(tr, &pb.digest[c]).unwrap();
    }
    <Tr as FieldTranscript<F>>::common_field_elements(tr, &pack_bytes::<F>(&pb.gray)).unwrap();
    absorb_meta::<F>(&[pb.nv, W[0] as usize, W[1] as usize, W[2] as usize, SCALE as usize, OFFSET as usize, ERR_TABLE], tr);
}

/// Error values e(x) as field elements plus their table multiplicities (out-of-range values,
/// which only occur under tampering, are left uncounted so the verifier's sum check fails).
fn error_channel<F: PrimeField>(img: [&Vec<u8>; 3], gray: &[u8]) -> (Vec<F>, Vec<u64>) {
    let n = gray.len();
    let e_int: Vec<i64> = (0..n)
        .into_par_iter()
        .map(|x| {
            let lin = W[0] * img[0][x] as u64 + W[1] * img[1][x] as u64 + W[2] * img[2][x] as u64;
            (OFFSET + SCALE * gray[x] as u64) as i64 - lin as i64
        })
        .collect();
    let e: Vec<F> = e_int.par_iter().map(|&v| if v >= 0 { F::from(v as u64) } else { -F::from((-v) as u64) }).collect();
    let mut m = vec![0u64; ERR_TABLE];
    for &v in &e_int {
        if (0..ERR_TABLE as i64).contains(&v) {
            m[v as usize] += 1;
        }
    }
    (e, m)
}

pub fn prove<F: PrimeField, Pcs: HvPcs<F>>(pp: &Pcs::ProverParam, input_size: usize, pb: &Public<F>) -> Vec<u8>
where
    Tr: TranscriptWrite<Pcs::CommitmentChunk, F>,
{
    println!("starting prover");
    let mut t = Instant::now();
    let nv = pb.nv;
    let n = 1usize << nv;
    let mut transcript = Tr::new(());
    absorb_public(pb, &mut transcript);

    // ---- load image, commit the three channel polynomials (unpadded) ----
    let orig = load_image(format!("images/Timings{}.json", input_size));
    let bytes = [&orig.R, &orig.G, &orig.B];
    let img: Vec<MultilinearPolynomial<F>> = bytes
        .iter()
        .enumerate()
        .map(|(c, b)| {
            let mut v = bytes_to_field::<F>(b);
            if c == 0 && tamper() == "range" {
                v[7] = F::from(300u64);
            }
            MultilinearPolynomial::new(v)
        })
        .collect();
    lap("prove: parse JSON + to field", &mut t);

    let img_comms = Pcs::batch_commit_and_write(pp, &img, &mut transcript).unwrap();
    lap("prove: commit 3 image polys (N each)", &mut t);

    // ---- hash preimage: Freivalds + sumcheck  sum_x rTA(x) * (img_R + a1 img_G + a2 img_B)(x) ----
    let r = transcript.squeeze_challenges(HASH_ROWS);
    let a1 = transcript.squeeze_challenge();
    let a2 = transcript.squeeze_challenge();
    let rta = MultilinearPolynomial::new(match &pb.tensor {
        Some(tp) => tp.rta_vector(&r),
        None => dense_rta(&r, n),
    });
    lap("hash: rTA (parallel)", &mut t);
    let hash_sum = {
        let dot = |c: usize| r.iter().zip(&pb.digest[c]).map(|(a, b)| *a * *b).sum::<F>();
        dot(0) + a1 * dot(1) + a2 * dot(2)
    };
    let p = |i: usize| Expression::<F>::Polynomial(Query::new(i, Rotation::cur()));
    let cst = |v: F| Expression::<F>::Constant(v);
    let hash_expr = p(0) * (p(1) + cst(a1) * p(2) + cst(a2) * p(3));
    let (p_h, _) = <ClassicSumCheck<EvaluationsProver<F>>>::prove(
        &(),
        nv,
        VirtualPolynomial::new(&hash_expr, [&rta, &img[0], &img[1], &img[2]], &[], &[]),
        hash_sum,
        &mut transcript,
    )
    .unwrap();
    drop(rta);
    lap("hash: sumcheck deg-2", &mut t);

    // ---- grayscale error channel e = 50 + 100 gray - (30 R + 59 G + 11 B), range [0, 100] ----
    let (mut e_vals, mut e_mults) = error_channel::<F>(bytes, &pb.gray);
    if tamper() == "leaf" {
        // internally consistent GKR over a fake in-range error channel; only the leaf identity
        // e~(rho) = 50 + 100 gray~(rho) - lin~(rho) can catch it
        e_vals = vec![F::ZERO; n];
        e_mults = vec![0u64; ERR_TABLE];
        e_mults[0] = n as u64;
    }
    let e_poly = MultilinearPolynomial::new(e_vals);
    lap("gray: error channel e (parallel)", &mut t);

    // ---- LogUp-GKR range check over 4 channels: img_c in [0,256), e in [0,101) ----
    let mut mults: Vec<Vec<F>> = (0..3)
        .map(|c| {
            let mut m = vec![0u64; TABLE];
            for &b in bytes[c].iter() {
                m[b as usize] += 1;
            }
            m.iter().map(|&x| F::from(x)).collect()
        })
        .collect();
    mults.push(e_mults.iter().map(|&x| F::from(x)).collect());
    if tamper() == "mults" {
        mults[3][10] += F::ONE;
        mults[3][20] -= F::ONE;
    }
    for c in 0..4 {
        transcript.write_field_elements(mults[c].iter()).unwrap();
    }
    let alpha: Vec<F> = transcript.squeeze_challenges(4);
    let chans: Vec<MultilinearPolynomial<F>> = img.iter().cloned().chain(std::iter::once(e_poly)).collect();
    let p_r = logup_gkr_prove(&chans, &alpha, &mut transcript);
    drop(chans);
    lap("range: LogUp-GKR, 4 channels (layers + n-1 sumchecks, nothing committed)", &mut t);

    // ---- openings: img_c at {p_h, p_r} ----
    let points = vec![p_h.clone(), p_r.clone()];
    let polys: Vec<&MultilinearPolynomial<F>> = img.iter().collect();
    let comms: Vec<&Pcs::Commitment> = img_comms.iter().collect();
    let jobs: Vec<(usize, usize)> = (0..3).flat_map(|c| [(c, 0), (c, 1)]).collect();
    let evals: Vec<Evaluation<F>> = jobs
        .par_iter()
        .map(|&(pid, k)| Evaluation::new(pid, k, polys[pid].evaluate(&points[k])))
        .collect();
    lap(&format!("open: {} MLE evaluations (parallel)", jobs.len()), &mut t);
    transcript.write_field_elements(evals.iter().map(Evaluation::value)).unwrap();
    Pcs::hv_open(pp, nv, &polys, &comms, &points, &evals, &mut transcript).unwrap();
    lap(&format!("open: {} openings ({} polys, {} evals)", Pcs::NAME, polys.len(), jobs.len()), &mut t);
    println!("prover done!");
    transcript.into_proof()
}

pub fn verify<F: PrimeField, Pcs: HvPcs<F>>(vp: &Pcs::VerifierParam, pb: &Public<F>, proof: &[u8]) -> bool
where
    Tr: TranscriptRead<Pcs::CommitmentChunk, F>,
{
    println!("\nstarting verifier");
    let mut t = Instant::now();
    let nv = pb.nv;
    let n = 1usize << nv;
    let mut tr = Tr::from_proof((), proof);
    absorb_public(pb, &mut tr);
    let mut ok = true;

    // ---- hash ----
    let img_comms: Vec<Pcs::Commitment> = Pcs::read_commitments(vp, 3, &mut tr).unwrap();
    let r = <Tr as FieldTranscript<F>>::squeeze_challenges(&mut tr, HASH_ROWS);
    let a1 = <Tr as FieldTranscript<F>>::squeeze_challenge(&mut tr);
    let a2 = <Tr as FieldTranscript<F>>::squeeze_challenge(&mut tr);
    let hash_sum = {
        let dot = |c: usize| r.iter().zip(&pb.digest[c]).map(|(a, b)| *a * *b).sum::<F>();
        dot(0) + a1 * dot(1) + a2 * dot(2)
    };
    let (e_h, p_h) = ClassicSumCheck::<EvaluationsProver<F>>::verify(&(), nv, 2, hash_sum, &mut tr).unwrap();

    // ---- range (4 channels) ----
    let tables = [TABLE, TABLE, TABLE, ERR_TABLE];
    let mut mults = Vec::new();
    for c in 0..4 {
        mults.push(<Tr as FieldTranscriptRead<F>>::read_field_elements(&mut tr, tables[c]).unwrap());
    }
    let alpha = <Tr as FieldTranscript<F>>::squeeze_challenges(&mut tr, 4);
    let n_f = F::from(n as u64);
    for c in 0..4 {
        ok &= mults[c].iter().sum::<F>() == n_f;
    }
    let s: Vec<F> = (0..4)
        .map(|c| (0..tables[c]).map(|j| mults[c][j] * (alpha[c] + F::from(j as u64)).invert().unwrap()).sum::<F>())
        .collect();
    let (p_r, q_claims) = match logup_gkr_verify::<F>(nv, &s, &mut tr) {
        Some(x) => x,
        None => {
            println!("LogUp-GKR range check failed");
            println!("Verifier passed!: false");
            return false;
        }
    };
    lap("verify: transcript replay + hash sumcheck + GKR", &mut t);

    // ---- openings ----
    let points = vec![p_h.clone(), p_r.clone()];
    let jobs: Vec<(usize, usize)> = (0..3).flat_map(|c| [(c, 0), (c, 1)]).collect();
    let vals = <Tr as FieldTranscriptRead<F>>::read_field_elements(&mut tr, jobs.len()).unwrap();
    let evals: Vec<Evaluation<F>> = jobs.iter().zip(&vals).map(|(&(pid, k), &v)| Evaluation::new(pid, k, v)).collect();
    let comms: Vec<&Pcs::Commitment> = img_comms.iter().collect();
    let open_ok = Pcs::hv_verify(vp, nv, &comms, &points, &evals, &mut tr);
    if let Err(e) = &open_ok {
        println!("opening verification failed: {:?}", e);
    }
    ok &= open_ok.is_ok();
    lap("verify: PCS openings", &mut t);
    let img_at = |c: usize, k: usize| vals[2 * c + k];

    // ---- hash identity ----
    let rta_at = match &pb.tensor {
        Some(tp) => tp.rta_eval(&r, &p_h),
        None => MultilinearPolynomial::new(dense_rta(&r, n)).evaluate(&p_h),
    };
    ok &= e_h == rta_at * (img_at(0, 0) + a1 * img_at(1, 0) + a2 * img_at(2, 0));
    lap(
        if pb.tensor.is_some() { "verify: rTA~(p) closed form  [O(128 log N)]" } else { "verify: rTA (parallel) + MLE eval  [linear-time part]" },
        &mut t,
    );

    // ---- range leaf claims: q_n(rho) = alpha_c + chan_c~(rho); e~(rho) from the openings + gray~(rho) ----
    for c in 0..3 {
        ok &= img_at(c, 1) + alpha[c] == q_claims[c];
    }
    let gray_at = MultilinearPolynomial::new(bytes_to_field::<F>(&pb.gray)).evaluate(&p_r);
    let lin_at = F::from(W[0]) * img_at(0, 1) + F::from(W[1]) * img_at(1, 1) + F::from(W[2]) * img_at(2, 1);
    let e_at = F::from(OFFSET) + F::from(SCALE) * gray_at - lin_at;
    ok &= e_at + alpha[3] == q_claims[3];
    lap("verify: gray~(rho)  [linear in N, the verifier reads its output]", &mut t);

    println!("Verifier passed!: {:?}", ok);
    println!("verifier done!\n");
    ok
}

pub fn run<F: PrimeField, Pcs: HvPcs<F>>(input_size: usize) -> bool
where
    Tr: TranscriptWrite<Pcs::CommitmentChunk, F> + TranscriptRead<Pcs::CommitmentChunk, F>,
{
    let (pp, vp, mut pb) = setup::<F, Pcs>(input_size);
    match tamper().as_str() {
        "gray" => pb.gray[3] ^= 2, // shifts e(3) by +-200: out of [0,100] whatever e(3) was
        "hash" => pb.digest[1][5] += F::ONE,
        _ => {}
    }
    let t0 = Instant::now();
    let proof = prove::<F, Pcs>(&pp, input_size, &pb);
    println!("PROVER TIME: {:?} seconds", t0.elapsed().as_millis() as f64 / 1000.0);
    println!("PROOF SIZE: {:?} bytes", proof.len());
    let t1 = Instant::now();
    let ok = verify::<F, Pcs>(&vp, &pb, &proof);
    println!("VERIFIER TIME: {:?} seconds", t1.elapsed().as_millis() as f64 / 1000.0);
    ok
}

pub fn run_range<F: PrimeField, Pcs: HvPcs<F>>(first: usize, last: usize) -> bool
where
    Tr: TranscriptWrite<Pcs::CommitmentChunk, F> + TranscriptRead<Pcs::CommitmentChunk, F>,
{
    let expect_ok = tamper().is_empty();
    let mut all = true;
    for i in first..=last {
        println!("-----------------------------------------------------------------------");
        println!("Full System Grayscale, HyperVerITAS v2 (range=gkr, hash={}, pcs={}). Size: 2^{:?}\n", hash_mode(), Pcs::NAME, i);
        let ok = run::<F, Pcs>(i);
        if !expect_ok {
            println!("TAMPER MODE '{}': verifier rejected = {}", tamper(), !ok);
        }
        all &= ok == expect_ok;
        println!("-----------------------------------------------------------------------");
    }
    all
}
