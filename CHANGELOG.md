# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.21.0] - 2023-09-25

### Added

- Support ONNX Runtime v1.16.0.

## [0.20.0] - 2023-06-19

### Added

- Support ONNX Runtime v1.15.1.

## [0.19.0] - 2023-05-26

### Added

- Support ONNX Runtime v1.15.0.

## [0.18.1] - 2023-04-19

### Fixed

- Change the return type of `element_count` into `Result` because
  `GetTensorShapeElementCount` may fail.

## [0.18.0] - 2023-03-09

### Added

- Support ONNX Runtime v1.14.1.

## [0.17.0] - 2023-03-09

### Added

- Support ONNX Runtime v1.14.0.

## [0.16.1] - 2023-01-09

### Changed

- Upgrade `onnrt-sys` to version 0.13.1, which avoid redundant recompilation.

## [0.16.0] - 2022-11-01

### Added

- Support ONNX Runtime v1.13.1.

## [0.15.0] - 2022-08-14

### Added

- Support ONNX Runtime v1.12.1.

## [0.14.0] - 2022-08-14

### Added

- Support ONNX Runtime v1.12.0.

### Changed

- Rename `add*` methods to `set*`.

## [0.13.0] - 2022-05-20

### Added

- Support ONNX Runtime v1.11.1.

## [0.12.0] - 2022-05-20

### Added

- Support ONNX Runtime v1.11.0.

## [0.11.0] - 2022-05-18

### Added

- Support ONNX Runtime v1.10.0.

## [0.10.0] - 2022-03-11

### Changed

- Upgrade `onnxrt-sys` to version 0.7.0, which supports Apple M1.

## [0.9.0] - 2022-01-13

### Changed

- Upgrade `onnxrt-sys` to version 0.6.0, which supports `ORT_INCLUDE_DIR` and
  `ORT_LIB_DIR`.

## [0.8.0] - 2022-01-05

### Added

- Support ONNX Runtime v1.9.0.

## [0.7.0] - 2021-09-24

### Changed

- Upgrade `onnxrt-sys` to version 0.4.0, which supports Apple M1.

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

[Unreleased]: https://github.com/furiosa-ai/onnxrt/compare/0.21.0...HEAD
[0.21.0]: https://github.com/furiosa-ai/onnxrt/compare/0.20.0...0.21.0
[0.20.0]: https://github.com/furiosa-ai/onnxrt/compare/0.19.0...0.20.0
[0.19.0]: https://github.com/furiosa-ai/onnxrt/compare/0.18.1...0.19.0
[0.18.1]: https://github.com/furiosa-ai/onnxrt/compare/0.18.0...0.18.1
[0.18.0]: https://github.com/furiosa-ai/onnxrt/compare/0.17.0...0.18.0
[0.17.0]: https://github.com/furiosa-ai/onnxrt/compare/0.16.1...0.17.0
[0.16.1]: https://github.com/furiosa-ai/onnxrt/compare/0.16.0...0.16.1
[0.16.0]: https://github.com/furiosa-ai/onnxrt/compare/0.15.0...0.16.0
[0.15.0]: https://github.com/furiosa-ai/onnxrt/compare/0.14.0...0.15.0
[0.14.0]: https://github.com/furiosa-ai/onnxrt/compare/0.13.0...0.14.0
[0.13.0]: https://github.com/furiosa-ai/onnxrt/compare/0.12.0...0.13.0
[0.12.0]: https://github.com/furiosa-ai/onnxrt/compare/0.11.0...0.12.0
[0.11.0]: https://github.com/furiosa-ai/onnxrt/compare/0.10.0...0.11.0
[0.10.0]: https://github.com/furiosa-ai/onnxrt/compare/0.9.0...0.10.0
[0.9.0]: https://github.com/furiosa-ai/onnxrt/compare/0.8.0...0.9.0
[0.8.0]: https://github.com/furiosa-ai/onnxrt/compare/0.7.0...0.8.0
[0.7.0]: https://github.com/furiosa-ai/onnxrt/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/furiosa-ai/onnxrt/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/furiosa-ai/onnxrt/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/furiosa-ai/onnxrt/compare/0.3.1...0.4.0
[0.3.1]: https://github.com/furiosa-ai/onnxrt/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/furiosa-ai/onnxrt/compare/0.2.1...0.3.0
[0.2.1]: https://github.com/furiosa-ai/onnxrt/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/furiosa-ai/onnxrt/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/furiosa-ai/onnxrt/releases/tag/0.1.0
