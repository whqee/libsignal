# Engineering review and regression fixes

## Changes

- Replace the radix-2 FFT with iterative Cooley–Tukey butterflies and correct bit-reversal indexing.
- Use direct DFT/IDFT for non-power-of-two lengths. Splitting a sequence into unrelated power-of-two blocks is not equivalent to transforming the complete sequence.
- Normalize inverse transforms by the complete transform length.
- Replace raw-pointer DCT worker access with scoped threads writing disjoint output chunks.
- Decode IEEE-754 samples from little-endian bytes rather than reinterpreting Vec<u8> as an aligned float slice.
- Generate IEEE-float WAV data with consistent format tag, bit depth, block alignment, byte rate, RIFF size, and chunk IDs.
- Fix time-series allocation to use sample count rather than byte count.
- Define empty-window behavior and fix causal convolution/correlation boundary loops.
- Add regression tests for FFT/DFT agreement, arbitrary-length round trips, parallel DCT equivalence, WAV headers/sample decoding, and convolution boundaries.

## Compatibility notes

- Wave::raw_as_float now returns Vec<Float>, not a borrowed slice. A borrowed float slice cannot safely be created from arbitrary Vec<u8> storage because alignment and aliasing guarantees are insufficient.
- Wave::new_default now describes IEEE-float WAV samples; it does not encode integer PCM.
- Non-power-of-two FFTs use direct DFT and therefore preserve correctness but have quadratic time complexity.

## Verification status

The tests have been added but have not been executed in this environment. Run cargo test, cargo fmt -- --check, and cargo clippy --all-targets --all-features -- -D warnings before merging.

The project still selects nightly and uses the unstable portable_simd feature. This branch intentionally does not silently change the toolchain requirement.
