use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field, PrimeField64};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_keccak::Keccak256Hash;
use p3_keccak_air::{KeccakAir};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::Poseidon2;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation, SerializingHasher32, CompressionFunctionFromHasher};
use p3_uni_stark::{prove, verify, StarkConfig};
use p3_util::log2_ceil_usize;
use rand::thread_rng;

pub struct PolynomialEvaluationAir {
    coefficients: Vec<BabyBear>,
}

impl<F> BaseAir<F> for PolynomialEvaluationAir {
    fn width(&self) -> usize {
        self.coefficients.len() + 1
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for PolynomialEvaluationAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let x = pis[0];

        let local: &PolynomialEvaluationRow<AB::Var> = main.row_slice(0).borrow();
        let next: &PolynomialEvaluationRow<AB::Var> = main.row_slice(1).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.result, self.coefficients[0]);

        let mut when_transition = builder.when_transition();

        // result' <- x * result + coeff
        when_transition.assert_eq(
            next.result,
            x * local.result + self.coefficients[local.coeff_index + 1],
        );

        builder.when_last_row().assert_eq(local.result, pis[1]);
    }
}

pub fn generate_trace_rows<F: PrimeField64>(
    coefficients: &[u64],
    x: u64,
    n: usize,
) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace = RowMajorMatrix::new(
        vec![F::zero(); n * (coefficients.len() + 1)],
        coefficients.len() + 1,
    );

    let (prefix, rows, suffix) =
        unsafe { trace.values.align_to_mut::<PolynomialEvaluationRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = PolynomialEvaluationRow::new(F::from_canonical_u64(coefficients[0]), 0);

    for i in 1..n {
        rows[i].result = x * rows[i - 1].result + coefficients[rows[i - 1].coeff_index + 1];
        rows[i].coeff_index = rows[i - 1].coeff_index + 1;
    }

    trace
}

pub struct PolynomialEvaluationRow<F> {
    pub result: F,
    pub coeff_index: usize,
}

impl<F> PolynomialEvaluationRow<F> {
    fn new(result: F, coeff_index: usize) -> PolynomialEvaluationRow<F> {
        PolynomialEvaluationRow { result, coeff_index }
    }
}

impl<F> std::borrow::Borrow<PolynomialEvaluationRow<F>> for [F] {
    fn borrow(&self) -> &PolynomialEvaluationRow<F> {
        debug_assert_eq!(self.len(), NUM_POLYNOMIAL_EVAL_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<PolynomialEvaluationRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

const NUM_POLYNOMIAL_EVAL_COLS: usize = 2;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 8>;
type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher32<ByteHash>;


#[test]
fn test_polynomial_evaluation() {
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(Keccak256Hash {});
    type MyCompress = CompressionFunctionFromHasher<u8, ByteHash, 2, 32>;
    let compress = MyCompress::new(byte_hash);
    type ValMmcs = FieldMerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress);
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    type Dft = Radix2DitParallel;
    let dft = Dft {};
    let coefficients = vec![1, 2, 3, 4];
    let air = PolynomialEvaluationAir {
        coefficients: coefficients.iter().map(|&c| BabyBear::from_canonical_u64(c)).collect(),
    };
    let trace = generate_trace_rows::<Val>(&coefficients, 5, 1 << 4);
    let fri_config = FriConfig {
        log_blowup: 2,
        num_queries: 28,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);
    let mut challenger = Challenger::from_hasher(vec![], byte_hash);
    let pis = vec![
        BabyBear::from_canonical_u64(5),
        BabyBear::from_canonical_u64(1234),
    ];
    let proof = prove(&config, &air, &mut challenger, trace, &pis);
    let mut challenger = Challenger::from_hasher(vec![], byte_hash);
    verify(&config, &air, &mut challenger, &proof, &pis)
}