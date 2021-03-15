# OnnxRt

[![build status](https://github.com/fuirosa-ai/onnxrt/workflows/build/badge.svg)](https://github.com/furiosa-ai/onnxrt/actions?query=workflow%3Abuild+branch%3Amain)

OnnxRt provides mid-level bindings to the C API for Microsoft's [ONNX Runtime].

[ONNX Runtime]: https://www.onnxruntime.ai/

## Requirements

### Rust

This program targets the latest stable version of Rust 1.50.0 or later.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
onnxrt = { git = "https://github.com/furiosa-ai/onnxrt", tag = "0.1.0" }
```

## Example

See the `examples` directory for a simple example.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
