# Engineering review and remediation notes

- Replaced unsound Vec<u8>-to-float borrowed casts with explicit little-endian decoding. raw_as_float returns an owned Vec; use set_float_sample for writes.
- Fixed RIFF/WAVE serialization: duplicate RIFF field, RIFF size, IEEE-float format tag, bits-per-sample units, block alignment and byte rate.
- Replaced invalid non-power-of-two FFT segmentation with radix-2 FFT plus direct DFT fallback; inverse normalizes once by N.
- Replaced DCT shared mutable raw-pointer slices with scoped threads and disjoint output chunks.
- Fixed make_wave_with_time allocation count and complex magnitude via hypot; added regression tests.

Compatibility: raw_as_float now returns Vec<Float>, not &[Float]. raw_as_mut_float returns a decoded copy; use set_float_sample to mutate the payload. rfft still returns the full complex spectrum.

Verification: cargo test and cargo clippy have not been run in this environment. Further review remains for convolution/correlation semantics, TrigSig merging and benchmarks.
