# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2021-08-23

### Added

- Support ONNX Runtime v1.8.1.

## [0.5.0] - 2021-08-22

### Added

- Support ONNX Runtime v1.8.0.

### Fixed

- Fix `Session::run_with_bytes_with_nul`, which detects the null character by
  `b'0'`, not `b'\0'`.

## [0.4.0] - 2021-07-28

### Added

- Support ONNX Runtime v1.7.

## [0.3.1] - 2021-06-03

### Fixed

- Fix `ModelMetadata::{description, domain}` and
  `ThreadingOptions::set_global_inter_op_num_threads`, which invoke wrong C
  functions.

## [0.3.0] - 2021-06-02

### Changed

- Upgrade `onnxruntime-sys` to version 0.0.12.

## [0.2.1] - 2021-04-01

### Fixed

- Clean up `clippy::upper_case_acronyms` warnings.

## [0.2.0] - 2021-03-15

### Added

- Support ONNX Runtime v1.6.

## [0.1.0] - 2021-03-15

### Added

- Support ONNX Runtime v1.5.

[Unreleased]: https://github.com/furiosa-ai/onnxrt/compare/0.6.0...HEAD
[0.6.0]: https://github.com/furiosa-ai/onnxrt/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/furiosa-ai/onnxrt/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/furiosa-ai/onnxrt/compare/0.3.1...0.4.0
[0.3.1]: https://github.com/furiosa-ai/onnxrt/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/furiosa-ai/onnxrt/compare/0.2.1...0.3.0
[0.2.1]: https://github.com/furiosa-ai/onnxrt/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/furiosa-ai/onnxrt/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/furiosa-ai/onnxrt/releases/tag/0.1.0
