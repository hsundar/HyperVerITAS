#![allow(warnings)]
//! HyperVerITAS crop, redesigned protocol ("v2"), generic over the field and the multilinear PCS.
//!
//! Changes relative to the original `hv_crop_*` examples (same statement):
//!   * Range check via LogUp: per channel the prover commits ONE polynomial w = 1/(alpha + img)
//!     of size N (instead of h, frac, prod of size 2N) and sends the 256 table multiplicities in
//!     the clear; the verifier sums the table side itself.
//!   * Image polynomials are committed unpadded (N instead of 2N).
//!   * Freivalds vectors are expanded from one transcript challenge with ChaCha8 instead of
//!     2^(n-1) transcript squeezes.
//!   * Hash matrix: tensor-structured by default (HV2_HASH=tensor) so the verifier evaluates the
//!     hash row MLEs in O(128 log N); HV2_HASH=dense keeps the original pseudo-random ChaCha matrix
//!     (verifier linear in N).
//!   * Hash-matrix products, batch inversion, crop vector, MLE evaluations are rayon-parallel.
//!
//! Negative tests: HV2_TAMPER=range|crop|hash corrupts the prover's image, the public crop, or the
//! public digest; the verifier must reject.

use std::{env, time::Instant};

use ark_std::rand::{RngCore, SeedableRng};
use ark_std::test_rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::image::*;

use plonkish_backend::{
    pcs::{
        multilinear::{
            additive::{batch_open_one, batch_verify_one},
            Basefold, BasefoldExtParams, MultilinearBrakedown, ZeromorphFri,
        },
        univariate::Fri,
        Evaluation, PolynomialCommitmentScheme,
    },
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{Field as myField, PrimeField},
        code::BrakedownSpec,
        expression::{Expression, Query, Rotation},
        hash::Blake2s,
        transcript::{
            FiatShamirTranscript, FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite,
            InMemoryTranscript, TranscriptRead, TranscriptWrite,
        },
    },
    Error,
};

/// Both prover and verifier transcripts (the in-memory impl exists only for Cursor<Vec<u8>>).
pub type Tr = FiatShamirTranscript<Blake2s, std::io::Cursor<Vec<u8>>>;

pub const HASH_ROWS: usize = 128;
pub const TABLE: usize = 256;
/// Column chunk for parallel generation / folding.
const CHUNK: usize = 1 << 12;

pub fn lap(label: &str, t: &mut Instant) {
    println!("[PROF] {:<58} {:>9.3} s", label, t.elapsed().as_secs_f64());
    *t = Instant::now();
}

pub fn hash_mode() -> String {
    env::var("HV2_HASH").unwrap_or_else(|_| "tensor".to_string())
}

pub fn tamper() -> String {
    env::var("HV2_TAMPER").unwrap_or_default()
}

// ------------------------------------------------------------------------------------
// PCS opening strategy. The trait's batch_open is a real multi-poly/multi-point reduction for
// Basefold and ZeromorphFri, but for Brakedown it just opens each evaluation separately; there the
// fork's `batch_open_one` (one poly, many points, reduced by a sumcheck) gives fewer openings.
// ------------------------------------------------------------------------------------

pub trait HvPcs<F: PrimeField>: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>> {
    const NAME: &'static str;

    fn hv_open(
        pp: &Self::ProverParam,
        nv: usize,
        polys: &[&MultilinearPolynomial<F>],
        comms: &[&Self::Commitment],
        points: &[Vec<F>],
        evals: &[Evaluation<F>],
        tr: &mut Tr,
    ) -> Result<(), Error>
    where
        Tr: TranscriptWrite<Self::CommitmentChunk, F>,
    {
        Self::batch_open(pp, polys.iter().copied(), comms.iter().copied(), points, evals, tr)
    }

    fn hv_verify(
        vp: &Self::VerifierParam,
        nv: usize,
        comms: &[&Self::Commitment],
        points: &[Vec<F>],
        evals: &[Evaluation<F>],
        tr: &mut Tr,
    ) -> Result<(), Error>
    where
        Tr: TranscriptRead<Self::CommitmentChunk, F>,
    {
        Self::batch_verify(vp, comms.iter().copied(), points, evals, tr)
    }
}

impl<F, V> HvPcs<F> for Basefold<F, Blake2s, V>
where
    F: PrimeField,
    V: BasefoldExtParams,
    Self: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    const NAME: &'static str = "basefold";
}

impl<F> HvPcs<F> for ZeromorphFri<Fri<F, Blake2s>>
where
    F: PrimeField,
    Self: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    const NAME: &'static str = "zeromorph_fri";
}

impl<F, S> HvPcs<F> for MultilinearBrakedown<F, Blake2s, S>
where
    F: PrimeField,
    S: BrakedownSpec,
    Self: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    const NAME: &'static str = "brakedown";

    fn hv_open(
        pp: &Self::ProverParam,
        nv: usize,
        polys: &[&MultilinearPolynomial<F>],
        comms: &[&Self::Commitment],
        points: &[Vec<F>],
        evals: &[Evaluation<F>],
        tr: &mut Tr,
    ) -> Result<(), Error>
    where
        Tr: TranscriptWrite<Self::CommitmentChunk, F>,
    {
        for (pi, list) in group_by_poly(polys.len(), evals) {
            if list.len() == 1 {
                Self::open(pp, polys[pi], comms[pi], &points[list[0].0], &list[0].1, tr)?;
            } else {
                let pts: Vec<Vec<F>> = list.iter().map(|(k, _)| points[*k].clone()).collect();
                let evs: Vec<Evaluation<F>> =
                    list.iter().enumerate().map(|(j, (_, v))| Evaluation::new(j, j, *v)).collect();
                batch_open_one::<F, Self>(pp, nv, vec![polys[pi]; list.len()], vec![comms[pi]; list.len()], &pts, &evs, tr)?;
            }
        }
        Ok(())
    }

    fn hv_verify(
        vp: &Self::VerifierParam,
        nv: usize,
        comms: &[&Self::Commitment],
        points: &[Vec<F>],
        evals: &[Evaluation<F>],
        tr: &mut Tr,
    ) -> Result<(), Error>
    where
        Tr: TranscriptRead<Self::CommitmentChunk, F>,
    {
        for (pi, list) in group_by_poly(comms.len(), evals) {
            if list.len() == 1 {
                Self::verify(vp, comms[pi], &points[list[0].0], &list[0].1, tr)?;
            } else {
                let pts: Vec<Vec<F>> = list.iter().map(|(k, _)| points[*k].clone()).collect();
                let evs: Vec<Evaluation<F>> =
                    list.iter().enumerate().map(|(j, (_, v))| Evaluation::new(j, j, *v)).collect();
                batch_verify_one::<F, Self>(vp, nv, comms[pi].clone(), &pts, &evs, tr)?;
            }
        }
        Ok(())
    }
}

fn group_by_poly<F: PrimeField>(num_polys: usize, evals: &[Evaluation<F>]) -> Vec<(usize, Vec<(usize, F)>)> {
    let mut by: Vec<Vec<(usize, F)>> = vec![Vec::new(); num_polys];
    for e in evals {
        by[e.poly()].push((e.point(), *e.value()));
    }
    by.into_iter().enumerate().filter(|(_, l)| !l.is_empty()).collect()
}

// ------------------------------------------------------------------------------------
// Field helpers
// ------------------------------------------------------------------------------------

pub fn eq_eval<F: PrimeField>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).fold(F::ONE, |acc, (&a, &b)| acc * (a * b + (F::ONE - a) * (F::ONE - b)))
}

fn batch_invert_chunk<F: PrimeField>(v: &mut [F]) {
    let n = v.len();
    let mut prefix = Vec::with_capacity(n);
    let mut acc = F::ONE;
    for x in v.iter() {
        prefix.push(acc);
        acc *= x;
    }
    let mut inv = acc.invert().unwrap();
    for i in (0..n).rev() {
        let tmp = inv * v[i];
        v[i] = inv * prefix[i];
        inv = tmp;
    }
}

fn par_batch_invert<F: PrimeField>(v: &mut [F]) {
    v.par_chunks_mut(CHUNK).for_each(batch_invert_chunk);
}

pub(crate) fn bytes_to_field<F: PrimeField>(v: &[u8]) -> Vec<F> {
    v.par_iter().map(|&x| F::from(x as u64)).collect()
}

pub(crate) fn powers<F: PrimeField>(base: F, n: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(n);
    let mut p = F::ONE;
    for _ in 0..n {
        out.push(p);
        p *= base;
    }
    out
}

fn seed_from_field<F: PrimeField>(x: F) -> [u8; 32] {
    let mut s = [0u8; 32];
    let repr = x.to_repr();
    let b = repr.as_ref();
    let k = b.len().min(32);
    s[..k].copy_from_slice(&b[..k]);
    s
}

/// ChaCha words consumed by one F::random (constant per field; rejection sampling is negligible).
fn words_per_elem<F: PrimeField>() -> u128 {
    let mut r = ChaCha8Rng::seed_from_u64(1);
    let a = r.get_word_pos();
    let _ = F::random(&mut r);
    r.get_word_pos() - a
}

// ------------------------------------------------------------------------------------
// Dense hash matrix (HV2_HASH=dense): row i = ChaCha8 stream seeded with i, as in the original.
// ------------------------------------------------------------------------------------

fn row_rng_at<F: PrimeField>(row: u64, col: usize) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(row);
    rng.set_word_pos((col as u128) * words_per_elem::<F>());
    rng
}

pub(crate) fn check_seekable<F: PrimeField>() {
    let mut seq = ChaCha8Rng::seed_from_u64(3);
    let expected: Vec<F> = (0..1500).map(|_| F::random(&mut seq)).collect();
    let mut rng = row_rng_at::<F>(3, 1000);
    for k in 0..500 {
        assert_eq!(F::random(&mut rng), expected[1000 + k], "ChaCha seek mismatch");
    }
}

pub(crate) fn dense_digest<F: PrimeField>(img: &[Vec<F>; 3]) -> Vec<Vec<F>> {
    let n = img[0].len();
    let rows: Vec<[F; 3]> = (0..HASH_ROWS as u64)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(i);
            let mut s = [F::ZERO; 3];
            for x in 0..n {
                let a = F::random(&mut rng);
                s[0] += a * img[0][x];
                s[1] += a * img[1][x];
                s[2] += a * img[2][x];
            }
            s
        })
        .collect();
    (0..3).map(|c| rows.iter().map(|s| s[c]).collect()).collect()
}

pub(crate) fn dense_rta<F: PrimeField>(r: &[F], n: usize) -> Vec<F> {
    let mut out = vec![F::ZERO; n];
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, chunk)| {
        let col0 = ci * CHUNK;
        for i in 0..HASH_ROWS {
            let mut rng = row_rng_at::<F>(i as u64, col0);
            let ri = r[i];
            for o in chunk.iter_mut() {
                *o += F::random(&mut rng) * ri;
            }
        }
    });
    out
}

// ------------------------------------------------------------------------------------
// Tensor-structured hash matrix (HV2_HASH=tensor, the default).
//   A[i][x] = prod_k a[i][k][bit_k(x)]   (LSB-first bit convention, matching the backend's MLE)
// so the MLE of row i at a point p is prod_k ((1-p_k) a[i][k][0] + p_k a[i][k][1]):
// the verifier evaluates rTA~(p) in 128*log N field ops instead of 128*N.
// SECURITY NOTE: this replaces SIS with a pseudo-random matrix by SIS with Kronecker-structured
// rows (equivalently, the hash is 128 multilinear evaluations of the image at public random
// points). No reduction is known to us for this structure; it is a research assumption.
// ------------------------------------------------------------------------------------

const TENSOR_SEED: u64 = 0xA7;

pub struct TensorParams<F> {
    nv: usize,
    c: usize,
    a: Vec<Vec<[F; 2]>>, // [row][var]
    w: Vec<Vec<F>>,      // [row][lo] = prod_{k<c} a[row][k][bit_k(lo)]
}

impl<F: PrimeField> TensorParams<F> {
    pub fn new(nv: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(TENSOR_SEED);
        let a: Vec<Vec<[F; 2]>> = (0..HASH_ROWS)
            .map(|_| (0..nv).map(|_| [F::random(&mut rng), F::random(&mut rng)]).collect())
            .collect();
        let c = nv.min(12);
        let w = (0..HASH_ROWS).into_par_iter().map(|i| Self::tensor_expand(&a[i][..c])).collect();
        Self { nv, c, a, w }
    }

    fn tensor_expand(a: &[[F; 2]]) -> Vec<F> {
        let mut v = vec![F::ONE];
        for k in 0..a.len() {
            let mut next = Vec::with_capacity(v.len() * 2);
            for b in 0..2 {
                for x in 0..v.len() {
                    next.push(v[x] * a[k][b]);
                }
            }
            v = next;
        }
        v
    }

    fn hi_factor(&self, i: usize, hi: usize) -> F {
        let mut f = F::ONE;
        for k in self.c..self.nv {
            f *= self.a[i][k][(hi >> (k - self.c)) & 1];
        }
        f
    }

    fn row_mle(&self, i: usize, p: &[F]) -> F {
        (0..self.nv).fold(F::ONE, |acc, k| acc * ((F::ONE - p[k]) * self.a[i][k][0] + p[k] * self.a[i][k][1]))
    }

    pub fn digest(&self, img: &[Vec<F>; 3]) -> Vec<Vec<F>> {
        let cs = 1usize << self.c;
        let nchunks = (1usize << self.nv) / cs;
        let acc = (0..nchunks)
            .into_par_iter()
            .map(|hi| {
                let mut out = vec![[F::ZERO; 3]; HASH_ROWS];
                let base = hi * cs;
                for i in 0..HASH_ROWS {
                    let w = &self.w[i];
                    let mut s = [F::ZERO; 3];
                    for lo in 0..cs {
                        let wl = w[lo];
                        let x = base + lo;
                        s[0] += wl * img[0][x];
                        s[1] += wl * img[1][x];
                        s[2] += wl * img[2][x];
                    }
                    let f = self.hi_factor(i, hi);
                    out[i] = [s[0] * f, s[1] * f, s[2] * f];
                }
                out
            })
            .reduce(
                || vec![[F::ZERO; 3]; HASH_ROWS],
                |mut a, b| {
                    for i in 0..HASH_ROWS {
                        for c in 0..3 {
                            a[i][c] += b[i][c];
                        }
                    }
                    a
                },
            );
        (0..3).map(|c| acc.iter().map(|s| s[c]).collect()).collect()
    }

    pub fn rta_vector(&self, r: &[F]) -> Vec<F> {
        let cs = 1usize << self.c;
        let mut out = vec![F::ZERO; 1 << self.nv];
        out.par_chunks_mut(cs).enumerate().for_each(|(hi, chunk)| {
            for i in 0..HASH_ROWS {
                let coef = r[i] * self.hi_factor(i, hi);
                let w = &self.w[i];
                for lo in 0..cs {
                    chunk[lo] += coef * w[lo];
                }
            }
        });
        out
    }

    pub fn rta_eval(&self, r: &[F], p: &[F]) -> F {
        (0..HASH_ROWS).map(|i| r[i] * self.row_mle(i, p)).sum()
    }
}

pub(crate) fn check_tensor_hash<F: PrimeField>() {
    let nv = 14;
    let tp = TensorParams::<F>::new(nv);
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let p: Vec<F> = (0..nv).map(|_| F::random(&mut rng)).collect();
    let row5 = TensorParams::tensor_expand(&tp.a[5]);
    assert_eq!(MultilinearPolynomial::new(row5.clone()).evaluate(&p), tp.row_mle(5, &p), "row MLE mismatch");
    let mut r = vec![F::ZERO; HASH_ROWS];
    r[5] = F::from(3u64);
    let rta = tp.rta_vector(&r);
    assert!(rta.iter().zip(&row5).all(|(x, y)| *x == *y * F::from(3u64)), "rTA mismatch");
    assert_eq!(MultilinearPolynomial::new(rta).evaluate(&p), tp.rta_eval(&r, &p), "rTA eval mismatch");
    let img: [Vec<F>; 3] = [
        (0..1 << nv).map(|_| F::from(rng.next_u64() & 255)).collect(),
        (0..1 << nv).map(|_| F::from(rng.next_u64() & 255)).collect(),
        (0..1 << nv).map(|_| F::from(rng.next_u64() & 255)).collect(),
    ];
    let d = tp.digest(&img);
    let explicit: F = row5.iter().zip(&img[1]).map(|(a, b)| *a * *b).sum();
    assert_eq!(d[1][5], explicit, "digest mismatch");
}

/// Expand one transcript challenge into n pseudo-random field elements (parallel).
pub(crate) fn expand_challenge<F: PrimeField>(seed: F, n: usize) -> Vec<F> {
    let s = seed_from_field(seed);
    let wpe = words_per_elem::<F>();
    let mut out = vec![F::ZERO; n];
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, chunk)| {
        let mut rng = ChaCha8Rng::from_seed(s);
        rng.set_word_pos(((ci * CHUNK) as u128) * wpe);
        for o in chunk.iter_mut() {
            *o = F::random(&mut rng);
        }
    });
    out
}

// ------------------------------------------------------------------------------------
// LogUp range check via GKR (HV2_RANGE=gkr, the default; HV2_RANGE=logup commits w = 1/(alpha+img)).
//
// Fraction-sum circuit per channel: leaves p_n(x) = 1, q_n(x) = alpha + img(x); layer k has 2^k
// entries built by pairing on the LSB of layer k+1,
//     p_k(y) = p_{k+1}(0,y) q_{k+1}(1,y) + p_{k+1}(1,y) q_{k+1}(0,y),   q_k(y) = q_{k+1}(0,y) q_{k+1}(1,y),
// so p_0/q_0 = sum_x 1/(alpha+img(x)), which must equal the table side S = sum_j m_j/(alpha+j).
// Each layer is verified by one degree-3 sumcheck (three channels batched with gamma, p/q claims
// combined with lambda); the four half-evaluations are then folded with a fresh t into claims at
// (t, sigma) on the next layer. At the leaves the only claim left is q_n(rho) = alpha + img~(rho),
// i.e. one opening of the committed image. Nothing besides the image is committed.
// ------------------------------------------------------------------------------------

pub fn range_mode() -> String {
    env::var("HV2_RANGE").unwrap_or_else(|_| "gkr".to_string())
}

fn split_halves<F: PrimeField>(v: &[F]) -> (Vec<F>, Vec<F>) {
    let m = v.len() / 2;
    let even: Vec<F> = (0..m).into_par_iter().map(|y| v[2 * y]).collect();
    let odd: Vec<F> = (0..m).into_par_iter().map(|y| v[2 * y + 1]).collect();
    (even, odd)
}

/// halves[k] = [p0, p1, q0, q1] of layer k (each of length 2^(k-1)), for k = 1..=n-1.
struct GkrLayers<F: PrimeField> {
    halves: Vec<[MultilinearPolynomial<F>; 4]>,
}

fn build_gkr_layers<F: PrimeField>(q_leaf: &[F]) -> GkrLayers<F> {
    let n = q_leaf.len().trailing_zeros() as usize;
    assert!(n >= 2);
    let empty = || MultilinearPolynomial::<F>::new(vec![F::ZERO]);
    let mut halves: Vec<[MultilinearPolynomial<F>; 4]> = (0..n).map(|_| [empty(), empty(), empty(), empty()]).collect();
    let m = 1usize << (n - 1);
    let mut p: Vec<F> = (0..m).into_par_iter().map(|y| q_leaf[2 * y] + q_leaf[2 * y + 1]).collect();
    let mut q: Vec<F> = (0..m).into_par_iter().map(|y| q_leaf[2 * y] * q_leaf[2 * y + 1]).collect();
    for k in (1..=n - 1).rev() {
        let (p0, p1) = split_halves(&p);
        let (q0, q1) = split_halves(&q);
        if k > 1 {
            let m = 1usize << (k - 1);
            p = (0..m).into_par_iter().map(|y| p0[y] * q1[y] + p1[y] * q0[y]).collect();
            q = (0..m).into_par_iter().map(|y| q0[y] * q1[y]).collect();
        }
        halves[k] = [
            MultilinearPolynomial::new(p0),
            MultilinearPolynomial::new(p1),
            MultilinearPolynomial::new(q0),
            MultilinearPolynomial::new(q1),
        ];
    }
    GkrLayers { halves }
}

/// Layer-k expression over the halves of layer k+1 (12 polys, 4 per channel), or over the
/// leaf q-halves (6 polys, 2 per channel; p halves are identically 1) when `leaf` is set.
fn gkr_layer_expr<F: PrimeField>(g: &[F], lambda: F, leaf: bool) -> Expression<F> {
    let p = |i: usize| Expression::<F>::Polynomial(Query::new(i, Rotation::cur()));
    let cst = |v: F| Expression::<F>::Constant(v);
    let mut sum = cst(F::ZERO);
    for c in 0..g.len() {
        let term = if leaf {
            p(2 * c) + p(2 * c + 1) + cst(lambda) * p(2 * c) * p(2 * c + 1)
        } else {
            p(4 * c) * p(4 * c + 3) + p(4 * c + 1) * p(4 * c + 2) + cst(lambda) * p(4 * c + 2) * p(4 * c + 3)
        };
        sum = sum + cst(g[c]) * term;
    }
    Expression::<F>::eq_xy(0) * sum
}

/// Returns rho in F^n: the point at which img_c must be opened (q_n(rho) = alpha_c + img_c~(rho)).
pub(crate) fn logup_gkr_prove<F: PrimeField>(img: &[MultilinearPolynomial<F>], alpha: &[F], tr: &mut Tr) -> Vec<F> {
    let n = img[0].num_vars();
    let nc = img.len();
    assert_eq!(alpha.len(), nc);
    let q_leaf: Vec<Vec<F>> = (0..nc).map(|c| img[c].evals().par_iter().map(|&x| x + alpha[c]).collect()).collect();
    let layers: Vec<GkrLayers<F>> = (0..nc).map(|c| build_gkr_layers(&q_leaf[c])).collect();

    // layer 1 in the clear: p_1(0), p_1(1), q_1(0), q_1(1) per channel
    for c in 0..nc {
        for h in 0..4 {
            tr.write_field_element(&layers[c].halves[1][h].evals()[0]).unwrap();
        }
    }
    let t = tr.squeeze_challenge();
    let mut rho = vec![t];
    let mut claims: Vec<(F, F)> = (0..nc)
        .map(|c| {
            let h = &layers[c].halves[1];
            let v = |i: usize| h[i].evals()[0];
            ((F::ONE - t) * v(0) + t * v(1), (F::ONE - t) * v(2) + t * v(3))
        })
        .collect();

    for k in 1..=n - 1 {
        let gamma = tr.squeeze_challenge();
        let lambda = tr.squeeze_challenge();
        let g = powers(gamma, nc);
        let claimed: F = (0..nc).map(|c| g[c] * (claims[c].0 + lambda * claims[c].1)).sum();
        let ys = [rho.clone()];
        let leaf = k == n - 1;
        let expr = gkr_layer_expr(&g, lambda, leaf);
        let leaf_halves: Vec<MultilinearPolynomial<F>> = if leaf {
            (0..nc)
                .flat_map(|c| {
                    let (e, o) = split_halves(&q_leaf[c]);
                    [MultilinearPolynomial::new(e), MultilinearPolynomial::new(o)]
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut refs: Vec<&MultilinearPolynomial<F>> = Vec::new();
        if leaf {
            refs.extend(leaf_halves.iter());
        } else {
            for c in 0..nc {
                for h in 0..4 {
                    refs.push(&layers[c].halves[k + 1][h]);
                }
            }
        }
        let (sigma, _) = <ClassicSumCheck<EvaluationsProver<F>>>::prove(
            &(),
            k,
            VirtualPolynomial::new(&expr, refs.iter().copied(), &[], &ys),
            claimed,
            tr,
        )
        .unwrap();
        let vals: Vec<F> = refs.par_iter().map(|m| m.evaluate(&sigma)).collect();
        tr.write_field_elements(vals.iter()).unwrap();
        let t = tr.squeeze_challenge();
        for c in 0..nc {
            let (p0, p1, q0, q1) = if leaf {
                (F::ONE, F::ONE, vals[2 * c], vals[2 * c + 1])
            } else {
                (vals[4 * c], vals[4 * c + 1], vals[4 * c + 2], vals[4 * c + 3])
            };
            claims[c] = ((F::ONE - t) * p0 + t * p1, (F::ONE - t) * q0 + t * q1);
        }
        rho = std::iter::once(t).chain(sigma.into_iter()).collect();
    }
    rho
}

/// Returns (rho, q-claims per channel) on success; the caller checks q_c == alpha_c + img_c~(rho).
pub(crate) fn logup_gkr_verify<F: PrimeField>(n: usize, s: &[F], tr: &mut Tr) -> Option<(Vec<F>, Vec<F>)> {
    let nc = s.len();
    let v = <Tr as FieldTranscriptRead<F>>::read_field_elements(tr, 4 * nc).unwrap();
    let t = <Tr as FieldTranscript<F>>::squeeze_challenge(tr);
    let mut rho = vec![t];
    let mut claims: Vec<(F, F)> = Vec::new();
    for c in 0..nc {
        let (p0, p1, q0, q1) = (v[4 * c], v[4 * c + 1], v[4 * c + 2], v[4 * c + 3]);
        let (p_top, q_top) = (p0 * q1 + p1 * q0, q0 * q1);
        if q_top == F::ZERO || p_top != s[c] * q_top {
            return None;
        }
        claims.push(((F::ONE - t) * p0 + t * p1, (F::ONE - t) * q0 + t * q1));
    }
    for k in 1..=n - 1 {
        let gamma = <Tr as FieldTranscript<F>>::squeeze_challenge(tr);
        let lambda = <Tr as FieldTranscript<F>>::squeeze_challenge(tr);
        let g = powers(gamma, nc);
        let claimed: F = (0..nc).map(|c| g[c] * (claims[c].0 + lambda * claims[c].1)).sum();
        let leaf = k == n - 1;
        let (e, sigma) = ClassicSumCheck::<EvaluationsProver<F>>::verify(&(), k, 3, claimed, tr).ok()?;
        let vals = <Tr as FieldTranscriptRead<F>>::read_field_elements(tr, if leaf { 2 * nc } else { 4 * nc }).unwrap();
        let eq = eq_eval(&rho, &sigma);
        let mut rhs = F::ZERO;
        let mut quad = Vec::new();
        for c in 0..nc {
            let (p0, p1, q0, q1) = if leaf {
                (F::ONE, F::ONE, vals[2 * c], vals[2 * c + 1])
            } else {
                (vals[4 * c], vals[4 * c + 1], vals[4 * c + 2], vals[4 * c + 3])
            };
            rhs += g[c] * (p0 * q1 + p1 * q0 + lambda * q0 * q1);
            quad.push((p0, p1, q0, q1));
        }
        if e != eq * rhs {
            return None;
        }
        let t = <Tr as FieldTranscript<F>>::squeeze_challenge(tr);
        for c in 0..nc {
            let (p0, p1, q0, q1) = quad[c];
            claims[c] = ((F::ONE - t) * p0 + t * p1, (F::ONE - t) * q0 + t * q1);
        }
        rho = std::iter::once(t).chain(sigma.into_iter()).collect();
    }
    if claims.iter().any(|(p, _)| *p != F::ONE) {
        return None;
    }
    Some((rho, claims.iter().map(|(_, q)| *q).collect()))
}

// ------------------------------------------------------------------------------------
// Crop geometry (same index arithmetic as the original affineCropProve)
// ------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct CropGeom {
    orig_w: usize,
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
}

impl CropGeom {
    fn orig_index(&self, row: usize, col: usize) -> usize {
        self.orig_w * (self.start_y + row) + self.start_x + col
    }
}

/// permR[orig_index(k)] = r'[k]; zero elsewhere. Crop rows map to disjoint runs of the original.
fn crop_perm_times_r<F: PrimeField>(g: &CropGeom, r: &[F], n: usize) -> Vec<F> {
    let mut out = vec![F::ZERO; n];
    let w = g.width;
    let runs: Vec<(usize, usize)> = (0..g.height).map(|row| (g.orig_index(row, 0), row * w)).collect();
    let ptr = out.as_mut_ptr() as usize;
    runs.par_iter().for_each(|&(dst, src)| {
        let dst_slice = unsafe { std::slice::from_raw_parts_mut((ptr as *mut F).add(dst), w) };
        dst_slice.copy_from_slice(&r[src..src + w]);
    });
    out
}

// ------------------------------------------------------------------------------------
// Setup
// ------------------------------------------------------------------------------------

pub struct Public<F> {
    pub nv: usize,
    pub nv_crop: usize,
    pub digest: Vec<Vec<F>>,
    pub crop: Image,
    pub geom: CropGeom,
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
    let crop = load_image(format!("images/Crop{}.json", input_size));
    let img = [bytes_to_field::<F>(&orig.R), bytes_to_field::<F>(&orig.G), bytes_to_field::<F>(&orig.B)];
    lap("setup: load images", &mut t);

    let digest = match &tensor {
        Some(tp) => tp.digest(&img),
        None => dense_digest(&img),
    };
    lap("setup: camera digest A*img (parallel)", &mut t);

    let geom = CropGeom { orig_w: orig.rows, width: crop.rows, height: crop.cols, start_x: 0, start_y: 0 };
    let nv_crop = (crop.rows * crop.cols).trailing_zeros() as usize;
    assert_eq!(1 << nv_crop, crop.rows * crop.cols);
    println!("setup done!\n");
    (pp, vp, Public { nv, nv_crop, digest, crop, geom, tensor })
}

/// 7 bytes per field element so it is injective for any field with modulus > 2^56.
pub(crate) fn pack_bytes<F: PrimeField>(b: &[u8]) -> Vec<F> {
    b.par_chunks(7)
        .map(|ch| {
            let mut v = 0u64;
            for (i, &x) in ch.iter().enumerate() {
                v |= (x as u64) << (8 * i);
            }
            F::from(v)
        })
        .collect()
}

/// Absorbs small public integers (sizes, geometry, constants) into the transcript.
pub(crate) fn absorb_meta<F: PrimeField>(meta: &[usize], tr: &mut Tr) {
    let v: Vec<F> = meta.iter().map(|&x| F::from(x as u64)).collect();
    <Tr as FieldTranscript<F>>::common_field_elements(tr, &v).unwrap();
}

/// Fiat-Shamir binding of the public statement: camera digest, crop image, geometry.
fn absorb_public<F: PrimeField>(pb: &Public<F>, tr: &mut Tr) {
    for c in 0..3 {
        <Tr as FieldTranscript<F>>::common_field_elements(tr, &pb.digest[c]).unwrap();
    }
    for ch in [&pb.crop.R, &pb.crop.G, &pb.crop.B] {
        <Tr as FieldTranscript<F>>::common_field_elements(tr, &pack_bytes::<F>(ch)).unwrap();
    }
    let g = &pb.geom;
    let meta = [pb.nv, pb.nv_crop, pb.crop.rows, pb.crop.cols, g.orig_w, g.width, g.height, g.start_x, g.start_y];
    absorb_meta::<F>(&meta, tr);
}

// ------------------------------------------------------------------------------------
// Prover
// ------------------------------------------------------------------------------------

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
                v[7] = F::from(300u64); // out of range; multiplicities below still come from the honest bytes
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

    // ---- LogUp range check: sum_x 1/(alpha_c + img_c(x)) = sum_j m_c[j]/(alpha_c + j) ----
    // Multiplicities go into the transcript BEFORE alpha is sampled (otherwise a prover who knows
    // alpha can pick field-element "multiplicities" that satisfy the single checked equation).
    let mut mults: Vec<Vec<F>> = (0..3)
        .map(|c| {
            let mut m = vec![0u64; TABLE];
            for &b in bytes[c].iter() {
                m[b as usize] += 1;
            }
            m.iter().map(|&x| F::from(x)).collect()
        })
        .collect();
    if tamper() == "mults" {
        mults[0][10] += F::ONE;
        mults[0][20] -= F::ONE;
    }
    for c in 0..3 {
        transcript.write_field_elements(mults[c].iter()).unwrap();
    }
    let alpha: Vec<F> = transcript.squeeze_challenges(3);
    let gkr = range_mode() == "gkr";
    let mut w: Vec<MultilinearPolynomial<F>> = Vec::new();
    let mut w_comms: Vec<Pcs::Commitment> = Vec::new();
    if !gkr {
        w = (0..3)
            .map(|c| {
                let mut v: Vec<F> = img[c].evals().par_iter().map(|&x| x + alpha[c]).collect();
                par_batch_invert(&mut v);
                MultilinearPolynomial::new(v)
            })
            .collect();
        lap("range: w = 1/(alpha+img) (parallel batch invert)", &mut t);
        w_comms = Pcs::batch_commit_and_write(pp, &w, &mut transcript).unwrap();
        lap("range: commit 3 inverse polys (N each)", &mut t);
    }

    let p_r: Vec<F> = if gkr {
        let rho = logup_gkr_prove(&img, &alpha, &mut transcript);
        lap("range: LogUp-GKR (layers + n-1 sumchecks, nothing committed)", &mut t);
        rho
    } else {
        let gamma = transcript.squeeze_challenge();
        let lambda = transcript.squeeze_challenge();
        let z = transcript.squeeze_challenges(nv);
        let g = powers(gamma, 3);
        let table_sum: F = (0..3)
            .map(|c| g[c] * (0..TABLE).map(|j| mults[c][j] * (alpha[c] + F::from(j as u64)).invert().unwrap()).sum::<F>())
            .sum();
        let mut range_expr = cst(F::ZERO);
        for c in 0..3 {
            let zero_term = p(3 + c) * (p(c) + cst(alpha[c])) + cst(-F::ONE);
            range_expr = range_expr + cst(g[c]) * (p(3 + c) + cst(lambda) * Expression::eq_xy(0) * zero_term);
        }
        let ys = [z.clone()];
        let (p_r, _) = <ClassicSumCheck<EvaluationsProver<F>>>::prove(
            &(),
            nv,
            VirtualPolynomial::new(&range_expr, [&img[0], &img[1], &img[2], &w[0], &w[1], &w[2]], &[], &ys),
            table_sum,
            &mut transcript,
        )
        .unwrap();
        lap("range: sumcheck deg-3 (3 channels batched)", &mut t);
        p_r
    };

    // ---- crop: sum_x permR(x) * (img_R + b1 img_G + b2 img_B)(x) = <r', crop_R + b1 crop_G + b2 crop_B> ----
    let seed = transcript.squeeze_challenge();
    let rp = expand_challenge(seed, 1 << pb.nv_crop);
    let b1 = transcript.squeeze_challenge();
    let b2 = transcript.squeeze_challenge();
    let perm_r = MultilinearPolynomial::new(crop_perm_times_r(&pb.geom, &rp, n));
    let crop_sum: F = {
        let (cr, cg, cb) = (&pb.crop.R, &pb.crop.G, &pb.crop.B);
        (0..rp.len())
            .into_par_iter()
            .map(|k| rp[k] * (F::from(cr[k] as u64) + b1 * F::from(cg[k] as u64) + b2 * F::from(cb[k] as u64)))
            .sum()
    };
    let crop_expr = p(3) * (p(0) + cst(b1) * p(1) + cst(b2) * p(2));
    let (p_c, _) = <ClassicSumCheck<EvaluationsProver<F>>>::prove(
        &(),
        nv,
        VirtualPolynomial::new(&crop_expr, [&img[0], &img[1], &img[2], &perm_r], &[], &[]),
        crop_sum,
        &mut transcript,
    )
    .unwrap();
    drop(perm_r);
    lap("crop: expand r' + permR + sumcheck deg-2 (parallel)", &mut t);

    // ---- openings: img_c at {p_h, p_r, p_c}; w_c at {p_r} ----
    let points = vec![p_h.clone(), p_r.clone(), p_c.clone()];
    let polys: Vec<&MultilinearPolynomial<F>> = img.iter().chain(w.iter()).collect();
    let comms: Vec<&Pcs::Commitment> = img_comms.iter().chain(w_comms.iter()).collect();
    let jobs: Vec<(usize, usize)> = (0..3)
        .flat_map(|c| if gkr { vec![(c, 0), (c, 1), (c, 2)] } else { vec![(c, 0), (c, 1), (c, 2), (3 + c, 1)] })
        .collect();
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

// ------------------------------------------------------------------------------------
// Verifier
// ------------------------------------------------------------------------------------

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

    // ---- range ----
    let mut mults = Vec::new();
    for _ in 0..3 {
        mults.push(<Tr as FieldTranscriptRead<F>>::read_field_elements(&mut tr, TABLE).unwrap());
    }
    let alpha = <Tr as FieldTranscript<F>>::squeeze_challenges(&mut tr, 3);
    let gkr = range_mode() == "gkr";
    let w_comms: Vec<Pcs::Commitment> = if gkr { Vec::new() } else { Pcs::read_commitments(vp, 3, &mut tr).unwrap() };
    let n_f = F::from(n as u64);
    for c in 0..3 {
        ok &= mults[c].iter().sum::<F>() == n_f;
    }
    // table side per channel: S_c = sum_j m_c[j] / (alpha_c + j)
    let s: Vec<F> = (0..3)
        .map(|c| (0..TABLE).map(|j| mults[c][j] * (alpha[c] + F::from(j as u64)).invert().unwrap()).sum::<F>())
        .collect();
    // state used by the final identity checks (committed-w mode only)
    let (mut gamma, mut lambda, mut e_r) = (F::ZERO, F::ZERO, F::ZERO);
    let mut z: Vec<F> = Vec::new();
    let mut g: Vec<F> = Vec::new();
    let mut q_claims: Vec<F> = Vec::new();
    let p_r: Vec<F> = if gkr {
        match logup_gkr_verify::<F>(nv, &s, &mut tr) {
            Some((rho, q)) => {
                q_claims = q;
                rho
            }
            None => {
                println!("LogUp-GKR range check failed");
                println!("Verifier passed!: false");
                return false;
            }
        }
    } else {
        gamma = <Tr as FieldTranscript<F>>::squeeze_challenge(&mut tr);
        lambda = <Tr as FieldTranscript<F>>::squeeze_challenge(&mut tr);
        z = <Tr as FieldTranscript<F>>::squeeze_challenges(&mut tr, nv);
        g = powers(gamma, 3);
        let table_sum: F = (0..3).map(|c| g[c] * s[c]).sum();
        let (e, p) = ClassicSumCheck::<EvaluationsProver<F>>::verify(&(), nv, 3, table_sum, &mut tr).unwrap();
        e_r = e;
        p
    };

    // ---- crop ----
    let seed = <Tr as FieldTranscript<F>>::squeeze_challenge(&mut tr);
    let rp = expand_challenge(seed, 1 << pb.nv_crop);
    let b1 = <Tr as FieldTranscript<F>>::squeeze_challenge(&mut tr);
    let b2 = <Tr as FieldTranscript<F>>::squeeze_challenge(&mut tr);
    let crop_sum: F = {
        let (cr, cg, cb) = (&pb.crop.R, &pb.crop.G, &pb.crop.B);
        (0..rp.len())
            .into_par_iter()
            .map(|k| rp[k] * (F::from(cr[k] as u64) + b1 * F::from(cg[k] as u64) + b2 * F::from(cb[k] as u64)))
            .sum()
    };
    let (e_c, p_c) = ClassicSumCheck::<EvaluationsProver<F>>::verify(&(), nv, 2, crop_sum, &mut tr).unwrap();
    lap("verify: transcript replay + 3 sumchecks", &mut t);

    // ---- openings ----
    let points = vec![p_h.clone(), p_r.clone(), p_c.clone()];
    let jobs: Vec<(usize, usize)> = (0..3)
        .flat_map(|c| if gkr { vec![(c, 0), (c, 1), (c, 2)] } else { vec![(c, 0), (c, 1), (c, 2), (3 + c, 1)] })
        .collect();
    let vals = <Tr as FieldTranscriptRead<F>>::read_field_elements(&mut tr, jobs.len()).unwrap();
    let evals: Vec<Evaluation<F>> = jobs.iter().zip(&vals).map(|(&(pid, k), &v)| Evaluation::new(pid, k, v)).collect();
    let comms: Vec<&Pcs::Commitment> = img_comms.iter().chain(w_comms.iter()).collect();
    let open_ok = Pcs::hv_verify(vp, nv, &comms, &points, &evals, &mut tr);
    if let Err(e) = &open_ok {
        println!("opening verification failed: {:?}", e);
    }
    ok &= open_ok.is_ok();
    lap("verify: PCS openings", &mut t);

    // img_at[c][k], w_at[c]
    let stride = if gkr { 3 } else { 4 };
    let img_at = |c: usize, k: usize| vals[stride * c + k];
    let w_at = |c: usize| vals[stride * c + 3];

    // ---- final identity checks ----
    let rta_at = match &pb.tensor {
        Some(tp) => tp.rta_eval(&r, &p_h),
        None => MultilinearPolynomial::new(dense_rta(&r, n)).evaluate(&p_h),
    };
    ok &= e_h == rta_at * (img_at(0, 0) + a1 * img_at(1, 0) + a2 * img_at(2, 0));
    lap(
        if pb.tensor.is_some() { "verify: rTA~(p) closed form  [O(128 log N)]" } else { "verify: rTA (parallel) + MLE eval  [linear-time part]" },
        &mut t,
    );

    if gkr {
        // leaf claim of the GKR: q_n(rho) = alpha_c + img_c~(rho)
        for c in 0..3 {
            ok &= img_at(c, 1) + alpha[c] == q_claims[c];
        }
    } else {
        let eq_z = eq_eval(&z, &p_r);
        let mut rhs = F::ZERO;
        for c in 0..3 {
            rhs += g[c] * (w_at(c) + lambda * eq_z * (w_at(c) * (img_at(c, 1) + alpha[c]) - F::ONE));
        }
        ok &= e_r == rhs;
    }

    let perm_r = MultilinearPolynomial::new(crop_perm_times_r(&pb.geom, &rp, n));
    let perm_at = perm_r.evaluate(&p_c);
    ok &= e_c == perm_at * (img_at(0, 2) + b1 * img_at(1, 2) + b2 * img_at(2, 2));
    lap("verify: permR + MLE eval  [linear in crop size]", &mut t);

    println!("Verifier passed!: {:?}", ok);
    println!("verifier done!\n");
    ok
}

// ------------------------------------------------------------------------------------
// Driver
// ------------------------------------------------------------------------------------

pub fn run<F: PrimeField, Pcs: HvPcs<F>>(input_size: usize) -> bool
where
    Tr: TranscriptWrite<Pcs::CommitmentChunk, F> + TranscriptRead<Pcs::CommitmentChunk, F>,
{
    let (pp, vp, mut pb) = setup::<F, Pcs>(input_size);
    match tamper().as_str() {
        "crop" => pb.crop.R[3] ^= 1,
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

/// Runs sizes first..=last; returns true iff every run matched the expectation
/// (accept normally, reject under HV2_TAMPER).
pub fn run_range<F: PrimeField, Pcs: HvPcs<F>>(first: usize, last: usize) -> bool
where
    Tr: TranscriptWrite<Pcs::CommitmentChunk, F> + TranscriptRead<Pcs::CommitmentChunk, F>,
{
    let expect_ok = tamper().is_empty();
    let mut all = true;
    for i in first..=last {
        println!("-----------------------------------------------------------------------");
        println!("Full System Crop, HyperVerITAS v2 (range={}, hash={}, pcs={}). Size: 2^{:?}\n", range_mode(), hash_mode(), Pcs::NAME, i);
        let ok = run::<F, Pcs>(i);
        if !expect_ok {
            println!("TAMPER MODE '{}': verifier rejected = {}", tamper(), !ok);
        }
        all &= ok == expect_ok;
        println!("-----------------------------------------------------------------------");
    }
    all
}
