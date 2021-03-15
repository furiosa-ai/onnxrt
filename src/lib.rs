//! https://github.com/microsoft/onnxruntime/blob/v1.5.2/include/onnxruntime/core/session/onnxruntime_c_api.h

#![warn(rust_2018_idioms)]

use std::{
    convert::TryInto,
    ffi::{CStr, CString},
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    mem,
    os::{
        raw::{c_char, c_int, c_void},
        unix::ffi::OsStrExt,
    },
    path::Path,
    ptr::{self, NonNull},
    slice, str,
    sync::{Arc, Mutex},
};

use once_cell::sync::Lazy;
pub use onnxruntime_sys::{
    ExecutionMode, GraphOptimizationLevel, ONNXTensorElementDataType, ONNXType, OrtAllocatorType,
    OrtErrorCode, OrtLanguageProjection, OrtLoggingLevel, OrtMemType, ORT_API_VERSION,
};
use onnxruntime_sys::{
    ONNXTensorElementDataType::{
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    },
    ONNXType::ONNX_TYPE_UNKNOWN,
    OrtAllocator, OrtApi, OrtArenaCfg, OrtEnv, OrtGetApiBase, OrtIoBinding, OrtLoggingFunction,
    OrtMemType::OrtMemTypeDefault,
    OrtMemoryInfo, OrtModelMetadata, OrtRunOptions, OrtSession, OrtSessionOptions,
    OrtTensorTypeAndShapeInfo, OrtThreadingOptions, OrtTypeInfo, OrtValue,
};

macro_rules! bail_on_error {
    ($x:expr) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let status = $x;
            if !status.is_null() {
                let code = $crate::ORT_API.GetErrorCode.unwrap()(status);
                assert_ne!(code, onnxruntime_sys::OrtErrorCode::ORT_OK);
                let message =
                    std::ffi::CStr::from_ptr($crate::ORT_API.GetErrorMessage.unwrap()(status))
                        .to_string_lossy()
                        .into_owned();
                $crate::ORT_API.ReleaseStatus.unwrap()(status);
                return std::result::Result::Err($crate::Error::OrtError { code, message });
            }
        }
    }};
}

macro_rules! panic_on_error {
    ($x:expr) => {{
        #[allow(unused_unsafe)]
        let status = unsafe { $x };
        assert_eq!(status, std::ptr::null_mut::<onnxruntime_sys::OrtStatus>());
    }};
}

pub static ORT_API: Lazy<&'static OrtApi> = Lazy::new(|| unsafe {
    OrtGetApiBase().as_ref().unwrap().GetApi.unwrap()(ORT_API_VERSION).as_ref().unwrap()
});

pub static ALLOCATOR_WITH_DEFAULT_OPTIONS: Lazy<Mutex<Allocator>> = Lazy::new(|| {
    let mut allocator = ptr::null_mut::<OrtAllocator>();
    panic_on_error!(ORT_API.GetAllocatorWithDefaultOptions.unwrap()(&mut allocator));
    Mutex::new(Allocator { raw: NonNull::new(allocator).unwrap() })
});

#[derive(Debug)]
pub enum Error {
    IoError { source: std::io::Error },
    NulError { source: std::ffi::NulError },
    OrtError { code: OrtErrorCode, message: String },
    TryFromIntError { source: std::num::TryFromIntError },
    Utf8Error { source: std::str::Utf8Error },
}

impl Display for self::Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::IoError { source } => source.fmt(f),
            Error::NulError { source } => source.fmt(f),
            Error::OrtError { message, .. } => write!(f, "{}", message),
            Error::TryFromIntError { source } => source.fmt(f),
            Error::Utf8Error { source } => source.fmt(f),
        }
    }
}

impl std::error::Error for self::Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::IoError { source } => Some(source),
            Error::NulError { source } => Some(source),
            Error::OrtError { .. } => None,
            Error::TryFromIntError { source } => Some(source),
            Error::Utf8Error { source } => Some(source),
        }
    }
}

impl From<std::io::Error> for self::Error {
    fn from(source: std::io::Error) -> Self {
        Self::IoError { source }
    }
}

impl From<std::ffi::NulError> for self::Error {
    fn from(source: std::ffi::NulError) -> Self {
        Self::NulError { source }
    }
}

impl From<std::num::TryFromIntError> for self::Error {
    fn from(source: std::num::TryFromIntError) -> Self {
        Self::TryFromIntError { source }
    }
}

impl From<std::str::Utf8Error> for self::Error {
    fn from(source: std::str::Utf8Error) -> Self {
        Self::Utf8Error { source }
    }
}

pub type Result<T, E = self::Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub struct Env {
    raw: NonNull<OrtEnv>,
}

impl Env {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L289-L292
    pub fn new(logging_level: OrtLoggingLevel, log_id: &str) -> self::Result<Self> {
        let mut env = ptr::null_mut::<OrtEnv>();
        let log_id = CString::new(log_id)?;
        bail_on_error!(ORT_API.CreateEnv.unwrap()(logging_level, log_id.as_ptr(), &mut env));
        Ok(Self { raw: NonNull::new(env).unwrap() })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L817-L823
    pub fn new_with_global_thread_pools(
        logging_level: OrtLoggingLevel,
        log_id: &str,
        threading_options: &ThreadingOptions,
    ) -> self::Result<Self> {
        let mut env = ptr::null_mut::<OrtEnv>();
        let log_id = CString::new(log_id)?;
        bail_on_error!(ORT_API.CreateEnvWithGlobalThreadPools.unwrap()(
            logging_level,
            log_id.as_ptr(),
            threading_options.raw.as_ptr(),
            &mut env,
        ));
        Ok(Self { raw: NonNull::new(env).unwrap() })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L294-L298
    pub fn new_with_custom_logger<T>(
        logging_function: OrtLoggingFunction,
        logger_param: Option<&'static mut T>,
        logging_level: OrtLoggingLevel,
        log_id: &str,
    ) -> self::Result<Self> {
        let mut env = ptr::null_mut::<OrtEnv>();
        let log_id = CString::new(log_id)?;
        bail_on_error!(ORT_API.CreateEnvWithCustomLogger.unwrap()(
            logging_function,
            logger_param
                .map(|param| param as *mut T as *mut c_void)
                .unwrap_or(ptr::null_mut::<c_void>()),
            logging_level,
            log_id.as_ptr(),
            &mut env,
        ));
        Ok(Self { raw: NonNull::new(env).unwrap() })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L300-L302
    pub fn enable_telemetry_events(&self) {
        panic_on_error!(ORT_API.EnableTelemetryEvents.unwrap()(self.raw.as_ptr()));
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L300-L302
    pub fn disable_telemetry_events(&self) {
        panic_on_error!(ORT_API.DisableTelemetryEvents.unwrap()(self.raw.as_ptr()));
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L1026-L1030
    pub fn set_language_projection(&self, projection: OrtLanguageProjection) {
        panic_on_error!(ORT_API.SetLanguageProjection.unwrap()(self.raw.as_ptr(), projection));
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L1014-L1024
    pub fn create_and_register_allocator(
        &mut self,
        memory_info: &MemoryInfo,
        arena_cfg: Option<&ArenaCfg>,
    ) -> self::Result<()> {
        bail_on_error!(ORT_API.CreateAndRegisterAllocator.unwrap()(
            self.raw.as_ptr(),
            memory_info.raw.as_ptr(),
            arena_cfg.map(|cfg| cfg.raw.as_ptr()).unwrap_or(ptr::null_mut::<OrtArenaCfg>()),
        ));
        Ok(())
    }
}

impl Drop for Env {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L707
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseEnv.unwrap()(self.raw.as_ptr());
        }
    }
}

unsafe impl Send for Env {}

unsafe impl Sync for Env {}

#[derive(Debug)]
pub struct RunOptions {
    raw: NonNull<OrtRunOptions>,
}

impl RunOptions {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L447-L450
    pub fn new() -> Self {
        let mut options = ptr::null_mut::<OrtRunOptions>();
        panic_on_error!(ORT_API.CreateRunOptions.unwrap()(&mut options));
        Self { raw: NonNull::new(options).unwrap() }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L452
    pub fn set_log_verbosity_level(&mut self, level: i32) -> &mut Self {
        panic_on_error!(ORT_API.RunOptionsSetRunLogVerbosityLevel.unwrap()(
            self.raw.as_ptr(),
            level
        ));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L456
    pub fn log_verbosity_level(&self) -> i32 {
        let mut level = 0;
        panic_on_error!(ORT_API.RunOptionsGetRunLogVerbosityLevel.unwrap()(
            self.raw.as_ptr(),
            &mut level,
        ));
        level
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L453
    pub fn set_log_severity_level(&mut self, level: i32) -> &mut Self {
        panic_on_error!(ORT_API.RunOptionsSetRunLogSeverityLevel.unwrap()(
            self.raw.as_ptr(),
            level
        ));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L457
    pub fn log_severity_level(&self) -> i32 {
        let mut level = 0;
        panic_on_error!(ORT_API.RunOptionsGetRunLogSeverityLevel.unwrap()(
            self.raw.as_ptr(),
            &mut level,
        ));
        level
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L454
    pub fn set_tag(&mut self, tag: &str) -> self::Result<&mut Self> {
        let tag = CString::new(tag)?;
        panic_on_error!(ORT_API.RunOptionsSetRunTag.unwrap()(self.raw.as_ptr(), tag.as_ptr()));
        Ok(self)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L458
    pub fn tag(&self) -> self::Result<&str> {
        let mut tag = ptr::null::<c_char>();
        panic_on_error!(ORT_API.RunOptionsGetRunTag.unwrap()(self.raw.as_ptr(), &mut tag));
        Ok(unsafe { CStr::from_ptr(tag) }.to_str()?)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L460-L462
    pub fn set_terminate(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.RunOptionsSetTerminate.unwrap()(self.raw.as_ptr()));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L463
    pub fn unset_terminate(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.RunOptionsUnsetTerminate.unwrap()(self.raw.as_ptr()));
        self
    }
}

impl Default for RunOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RunOptions {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L712
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseRunOptions.unwrap()(self.raw.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct SessionOptions {
    session_options: NonNull<OrtSessionOptions>,
}

impl SessionOptions {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L321-L324
    pub fn new() -> Self {
        let mut session_options = ptr::null_mut::<OrtSessionOptions>();
        panic_on_error!(ORT_API.CreateSessionOptions.unwrap()(&mut session_options));
        Self { session_options: NonNull::new(session_options).unwrap() }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L367-L371
    pub fn set_intra_op_num_threads(&mut self, intra_op_num_threads: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetIntraOpNumThreads.unwrap()(
            self.session_options.as_ptr(),
            intra_op_num_threads,
        ));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L373-L376
    pub fn set_inter_op_num_threads(&mut self, inter_op_num_threads: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetInterOpNumThreads.unwrap()(
            self.session_options.as_ptr(),
            inter_op_num_threads,
        ));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L364-L365
    pub fn set_graph_optimization_level(
        &mut self,
        graph_optimization_level: GraphOptimizationLevel,
    ) -> &mut Self {
        panic_on_error!(ORT_API.SetSessionGraphOptimizationLevel.unwrap()(
            self.session_options.as_ptr(),
            graph_optimization_level,
        ));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L351-L355
    pub fn enable_cpu_mem_arena(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.EnableCpuMemArena.unwrap()(self.session_options.as_ptr()));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L351-L355
    pub fn disable_cpu_mem_arena(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.DisableCpuMemArena.unwrap()(self.session_options.as_ptr()));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L326-L328
    pub fn set_optimized_model_file_path<P: AsRef<Path>>(
        &mut self,
        optimized_model_file_path: P,
    ) -> self::Result<&mut Self> {
        let optimized_model_file =
            CString::new(optimized_model_file_path.as_ref().as_os_str().as_bytes())?;
        panic_on_error!(ORT_API.SetOptimizedModelFilePath.unwrap()(
            self.session_options.as_ptr(),
            optimized_model_file.as_ptr(),
        ));
        Ok(self)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L339-L341
    pub fn enable_profiling<P: AsRef<Path>>(
        &mut self,
        profile_file_prefix: P,
    ) -> self::Result<&mut Self> {
        let profile_file_prefix =
            CString::new(profile_file_prefix.as_ref().as_os_str().as_bytes())?;
        panic_on_error!(ORT_API.EnableProfiling.unwrap()(
            self.session_options.as_ptr(),
            profile_file_prefix.as_ptr(),
        ));
        Ok(self)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L339-L341
    pub fn disable_profiling(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.DisableProfiling.unwrap()(self.session_options.as_ptr()));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L343-L349
    pub fn enable_mem_pattern(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.EnableMemPattern.unwrap()(self.session_options.as_ptr()));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L343-L349
    pub fn disable_mem_pattern(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.DisableMemPattern.unwrap()(self.session_options.as_ptr()));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L334-L337
    pub fn set_execution_mode(&mut self, execution_mode: ExecutionMode) -> &mut Self {
        panic_on_error!(ORT_API.SetSessionExecutionMode.unwrap()(
            self.session_options.as_ptr(),
            execution_mode,
        ));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L357-L358
    pub fn set_log_id(&mut self, log_id: &str) -> self::Result<&mut Self> {
        let log_id = CString::new(log_id)?;
        panic_on_error!(ORT_API.SetSessionLogId.unwrap()(
            self.session_options.as_ptr(),
            log_id.as_ptr(),
        ));
        Ok(self)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L360-L362
    pub fn set_log_verbosity_level(&mut self, level: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetSessionLogVerbosityLevel.unwrap()(
            self.session_options.as_ptr(),
            level,
        ));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L360-L362
    pub fn set_log_severity_level(&mut self, level: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetSessionLogSeverityLevel.unwrap()(
            self.session_options.as_ptr(),
            level,
        ));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L827-L831
    pub fn disable_per_session_threads(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.DisablePerSessionThreads.unwrap()(self.session_options.as_ptr()));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L894-L902
    pub fn add_session_config_entry(
        &mut self,
        config_key: &str,
        config_value: &str,
    ) -> self::Result<&mut Self> {
        let config_key = CString::new(config_key)?;
        let config_value = CString::new(config_value)?;
        bail_on_error!(ORT_API.AddSessionConfigEntry.unwrap()(
            self.session_options.as_ptr(),
            config_key.as_ptr(),
            config_value.as_ptr(),
        ));
        Ok(self)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L603-L606
    pub fn add_free_dimension_override(
        &mut self,
        dim_denotation: &str,
        dim_value: i64,
    ) -> self::Result<&mut Self> {
        let dim_notation = CString::new(dim_denotation)?;
        panic_on_error!(ORT_API.AddFreeDimensionOverride.unwrap()(
            self.session_options.as_ptr(),
            dim_notation.as_ptr(),
            dim_value,
        ));
        Ok(self)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L846-L851
    pub fn add_free_dimension_override_by_name(
        &mut self,
        dim_name: &str,
        dim_value: i64,
    ) -> self::Result<&mut Self> {
        let dim_notation = CString::new(dim_name)?;
        panic_on_error!(ORT_API.AddFreeDimensionOverrideByName.unwrap()(
            self.session_options.as_ptr(),
            dim_notation.as_ptr(),
            dim_value,
        ));
        Ok(self)
    }
}

impl Clone for SessionOptions {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L330-L332
    fn clone(&self) -> Self {
        let mut session_options = ptr::null_mut::<OrtSessionOptions>();
        panic_on_error!(ORT_API.CloneSessionOptions.unwrap()(
            self.session_options.as_ptr(),
            &mut session_options,
        ));
        Self { session_options: NonNull::new(session_options).unwrap() }
    }
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SessionOptions {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L715
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseSessionOptions.unwrap()(self.session_options.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct ModelMetadata {
    raw: NonNull<OrtModelMetadata>,
}

impl ModelMetadata {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L795-L805
    pub fn producer_name(&self, allocator: &mut Allocator) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut name_ptr = ptr::null_mut::<c_char>();
        panic_on_error!(ORT_API.ModelMetadataGetProducerName.unwrap()(
            self.raw.as_ptr(),
            allocator,
            &mut name_ptr,
        ));
        unsafe {
            let name = CStr::from_ptr(name_ptr).to_str()?.to_string();
            (*allocator).Free.unwrap()(allocator, name_ptr as *mut c_void);
            Ok(name)
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L795-L805
    pub fn graph_name(&self, allocator: &mut Allocator) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut name_ptr = ptr::null_mut::<c_char>();
        panic_on_error!(ORT_API.ModelMetadataGetGraphName.unwrap()(
            self.raw.as_ptr(),
            allocator,
            &mut name_ptr,
        ));
        unsafe {
            let name = CStr::from_ptr(name_ptr).to_str()?.to_string();
            (*allocator).Free.unwrap()(allocator, name_ptr as *mut c_void);
            Ok(name)
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L795-L805
    pub fn domain(&self, allocator: &mut Allocator) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut domain_ptr = ptr::null_mut::<c_char>();
        panic_on_error!(ORT_API.ModelMetadataGetGraphName.unwrap()(
            self.raw.as_ptr(),
            allocator,
            &mut domain_ptr,
        ));
        unsafe {
            let domain = CStr::from_ptr(domain_ptr).to_str()?.to_string();
            (*allocator).Free.unwrap()(allocator, domain_ptr as *mut c_void);
            Ok(domain)
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L795-L805
    pub fn description(&self, allocator: &mut Allocator) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut description_ptr = ptr::null_mut::<c_char>();
        panic_on_error!(ORT_API.ModelMetadataGetGraphName.unwrap()(
            self.raw.as_ptr(),
            allocator,
            &mut description_ptr,
        ));
        unsafe {
            let description = CStr::from_ptr(description_ptr).to_str()?.to_string();
            (*allocator).Free.unwrap()(allocator, description_ptr as *mut c_void);
            Ok(description)
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L837-L844
    pub fn custom_metadata_map_keys(&self, allocator: &mut Allocator) -> self::Result<Vec<String>> {
        let allocator = allocator.raw.as_ptr();
        let mut keys_ptr = ptr::null_mut::<*mut c_char>();
        let mut num_keys = 0;
        panic_on_error!(ORT_API.ModelMetadataGetCustomMetadataMapKeys.unwrap()(
            self.raw.as_ptr(),
            allocator,
            &mut keys_ptr,
            &mut num_keys,
        ));
        if !keys_ptr.is_null() {
            unsafe {
                let keys_slice = slice::from_raw_parts(keys_ptr, num_keys.try_into()?);
                let keys = (keys_slice.iter())
                    .map(|&key| CStr::from_ptr(key).to_str().map(str::to_string))
                    .collect::<Result<Vec<_>, _>>()?;
                let free = (*allocator).Free.as_ref().unwrap();
                keys_slice.iter().for_each(|&key| free(allocator, key as *mut c_void));
                free(allocator, keys_ptr as *mut c_void);
                Ok(keys)
            }
        } else {
            Ok(Vec::new())
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L806-L811
    pub fn lookup_custom_metadata_map(
        &self,
        allocator: &mut Allocator,
        key: &str,
    ) -> self::Result<Option<String>> {
        let allocator = allocator.raw.as_ptr();
        let key = CString::new(key)?;
        let mut value_ptr = ptr::null_mut::<c_char>();
        panic_on_error!(ORT_API.ModelMetadataLookupCustomMetadataMap.unwrap()(
            self.raw.as_ptr(),
            allocator,
            key.as_ptr(),
            &mut value_ptr,
        ));
        if !value_ptr.is_null() {
            unsafe {
                let value = CStr::from_ptr(value_ptr).to_str()?.to_string();
                (*allocator).Free.unwrap()(allocator, value_ptr as *mut c_void);
                Ok(Some(value))
            }
        } else {
            Ok(None)
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L813
    pub fn version(&self) -> i64 {
        let mut version = 0;
        panic_on_error!(ORT_API.ModelMetadataGetVersion.unwrap()(self.raw.as_ptr(), &mut version));
        version
    }
}

impl Drop for ModelMetadata {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L815
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseModelMetadata.unwrap()(self.raw.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct Session {
    raw: NonNull<OrtSession>,
    #[allow(dead_code)] // `env` must outlive `raw`.
    env: Arc<Mutex<Env>>,
}

impl Session {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L304-L310
    pub fn new_with_model_path<P: AsRef<Path>>(
        env: Arc<Mutex<Env>>,
        model_path: P,
        options: &SessionOptions,
    ) -> self::Result<Self> {
        let model_path = CString::new(model_path.as_ref().as_os_str().as_bytes())?;
        let mut session = ptr::null_mut::<OrtSession>();
        bail_on_error!(ORT_API.CreateSession.unwrap()(
            env.lock().unwrap().raw.as_ptr(),
            model_path.as_ptr(),
            options.session_options.as_ptr(),
            &mut session,
        ));
        Ok(Session { raw: NonNull::new(session).unwrap(), env })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L312-L313
    pub fn new_with_model_data(
        env: Arc<Mutex<Env>>,
        model_data: &[u8],
        options: &SessionOptions,
    ) -> self::Result<Self> {
        let mut session = ptr::null_mut::<OrtSession>();
        bail_on_error!(ORT_API.CreateSessionFromArray.unwrap()(
            env.lock().unwrap().raw.as_ptr(),
            model_data.as_ptr() as *const c_void,
            model_data.len().try_into()?,
            options.session_options.as_ptr(),
            &mut session,
        ));
        Ok(Session { raw: NonNull::new(session).unwrap(), env })
    }

    pub fn run<I: AsRef<str>, O: AsRef<str>>(
        &mut self,
        run_options: Option<&RunOptions>,
        input_names: &[I],
        input_values: &[Value<'_>],
        output_names: &[O],
        output_values: &mut [Value<'_>],
    ) -> self::Result<()> {
        assert_eq!(input_names.len(), input_values.len());
        assert_eq!(output_names.len(), output_values.len());
        unsafe {
            self.run_unchecked(run_options, input_names, input_values, output_names, output_values)
        }
    }

    /// # Safety
    ///
    /// The lengths of `input_names` and `output_names` must be those of `input_values` and
    /// `output_values`, respectively.
    pub unsafe fn run_unchecked<I: AsRef<str>, O: AsRef<str>>(
        &mut self,
        run_options: Option<&RunOptions>,
        input_names: &[I],
        input_values: &[Value<'_>],
        output_names: &[O],
        output_values: &mut [Value<'_>],
    ) -> self::Result<()> {
        let input_names_c_string = (input_names.iter())
            .map(AsRef::as_ref)
            .map(CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        let output_names_c_string = (output_names.iter())
            .map(AsRef::as_ref)
            .map(CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        self.run_with_c_str_unchecked(
            run_options,
            &input_names_c_string,
            input_values,
            &output_names_c_string,
            output_values,
        )
    }

    pub fn run_with_c_str<I: AsRef<CStr>, O: AsRef<CStr>>(
        &mut self,
        run_options: Option<&RunOptions>,
        input_names: &[I],
        input_values: &[Value<'_>],
        output_names: &[O],
        output_values: &mut [Value<'_>],
    ) -> self::Result<()> {
        assert_eq!(input_names.len(), input_values.len());
        assert_eq!(output_names.len(), output_values.len());
        unsafe {
            self.run_with_c_str_unchecked(
                run_options,
                input_names,
                input_values,
                output_names,
                output_values,
            )
        }
    }

    /// # Safety
    ///
    /// The lengths of `input_names` and `output_names` must be those of `input_values` and
    /// `output_values`, respectively.
    pub unsafe fn run_with_c_str_unchecked<I: AsRef<CStr>, O: AsRef<CStr>>(
        &mut self,
        run_options: Option<&RunOptions>,
        input_names: &[I],
        input_values: &[Value<'_>],
        output_names: &[O],
        output_values: &mut [Value<'_>],
    ) -> self::Result<()> {
        let input_names_c_char =
            input_names.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        let output_names_c_char =
            output_names.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        self.run_with_c_chars_with_nul(
            run_options,
            &input_names_c_char,
            input_values,
            &output_names_c_char,
            output_values,
        )
    }

    pub fn run_with_bytes_with_nul(
        &mut self,
        run_options: Option<&RunOptions>,
        input_names: &[&[u8]],
        input_values: &[Value<'_>],
        output_names: &[&[u8]],
        output_values: &mut [Value<'_>],
    ) -> self::Result<()> {
        assert_eq!(input_names.len(), input_values.len());
        assert!(input_names.iter().all(|name| name.ends_with(&[b'0'])));
        assert_eq!(output_names.len(), output_values.len());
        assert!(output_names.iter().all(|name| name.ends_with(&[b'0'])));
        unsafe {
            self.run_with_bytes_with_nul_unchecked(
                run_options,
                input_names,
                input_values,
                output_names,
                output_values,
            )
        }
    }

    /// # Safety
    ///
    /// The lengths of `input_names` and `output_names` must be those of `input_values` and
    /// `output_values`, respectively.
    pub unsafe fn run_with_bytes_with_nul_unchecked(
        &mut self,
        run_options: Option<&RunOptions>,
        input_names: &[&[u8]],
        input_values: &[Value<'_>],
        output_names: &[&[u8]],
        output_values: &mut [Value<'_>],
    ) -> self::Result<()> {
        let input_names_c_char =
            input_names.iter().map(|x| x.as_ptr() as *const c_char).collect::<Vec<_>>();
        let output_names_c_char =
            output_names.iter().map(|x| x.as_ptr() as *const c_char).collect::<Vec<_>>();
        self.run_with_c_chars_with_nul(
            run_options,
            &input_names_c_char,
            input_values,
            &output_names_c_char,
            output_values,
        )
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L315-L319
    ///
    /// # Safety
    ///
    /// The lengths of `input_names` and `output_names` must be those of `input_values` and
    /// `output_values`, respectively.
    pub unsafe fn run_with_c_chars_with_nul(
        &mut self,
        run_options: Option<&RunOptions>,
        input_names: &[*const c_char],
        input_values: &[Value<'_>],
        output_names: &[*const c_char],
        output_values: &mut [Value<'_>],
    ) -> self::Result<()> {
        debug_assert_eq!(input_names.len(), input_values.len());
        debug_assert_eq!(output_names.len(), output_values.len());
        bail_on_error!(ORT_API.Run.unwrap()(
            self.raw.as_ptr(),
            run_options.map(|x| x.raw.as_ptr() as *const _).unwrap_or(ptr::null::<OrtRunOptions>()),
            input_names.as_ptr(),
            input_values.as_ptr() as *const *const OrtValue,
            input_values.len().try_into()?,
            output_names.as_ptr(),
            output_values.len().try_into()?,
            output_values.as_mut_ptr() as *mut *mut OrtValue,
        ));
        Ok(())
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L917
    pub fn run_with_binding(
        &mut self,
        run_options: Option<&RunOptions>,
        binding: &IoBinding<'_>,
    ) -> self::Result<()> {
        bail_on_error!(ORT_API.RunWithBinding.unwrap()(
            self.raw.as_ptr(),
            run_options.map(|x| x.raw.as_ptr() as *const _).unwrap_or(ptr::null::<OrtRunOptions>()),
            binding.raw.as_ptr(),
        ));
        Ok(())
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L416
    pub fn input_count(&self) -> self::Result<usize> {
        let mut count = 0;
        bail_on_error!(ORT_API.SessionGetInputCount.unwrap()(self.raw.as_ptr(), &mut count));
        Ok(count.try_into()?)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L417
    pub fn output_count(&self) -> self::Result<usize> {
        let mut count = 0;
        bail_on_error!(ORT_API.SessionGetOutputCount.unwrap()(self.raw.as_ptr(), &mut count));
        Ok(count.try_into()?)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L418
    pub fn overridable_initializer_count(&self) -> self::Result<usize> {
        let mut count = 0;
        bail_on_error!(ORT_API.SessionGetOverridableInitializerCount.unwrap()(
            self.raw.as_ptr(),
            &mut count,
        ));
        Ok(count.try_into()?)
    }

    pub fn input_name(&self, index: usize) -> self::Result<String> {
        self.input_name_using_allocator(index, &mut ALLOCATOR_WITH_DEFAULT_OPTIONS.lock().unwrap())
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L437-L445
    pub fn input_name_using_allocator(
        &self,
        index: usize,
        allocator: &mut Allocator,
    ) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut name_ptr = ptr::null_mut::<c_char>();
        bail_on_error!(ORT_API.SessionGetInputName.unwrap()(
            self.raw.as_ptr(),
            index.try_into()?,
            allocator,
            &mut name_ptr,
        ));
        unsafe {
            let name = CStr::from_ptr(name_ptr).to_str()?.to_string();
            (*allocator).Free.unwrap()(allocator, name_ptr as *mut c_void);
            Ok(name)
        }
    }

    pub fn output_name(&self, index: usize) -> self::Result<String> {
        self.output_name_using_allocator(index, &mut ALLOCATOR_WITH_DEFAULT_OPTIONS.lock().unwrap())
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L437-L445
    pub fn output_name_using_allocator(
        &self,
        index: usize,
        allocator: &mut Allocator,
    ) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut name_ptr = ptr::null_mut::<c_char>();
        bail_on_error!(ORT_API.SessionGetOutputName.unwrap()(
            self.raw.as_ptr(),
            index.try_into()?,
            allocator,
            &mut name_ptr,
        ));
        unsafe {
            let name = CStr::from_ptr(name_ptr).to_str()?.to_string();
            (*allocator).Free.unwrap()(allocator, name_ptr as *mut c_void);
            Ok(name)
        }
    }

    pub fn overridable_initializer_name(&self, index: usize) -> self::Result<String> {
        self.overridable_initializer_name_using_allocator(
            index,
            &mut ALLOCATOR_WITH_DEFAULT_OPTIONS.lock().unwrap(),
        )
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L437-L445
    pub fn overridable_initializer_name_using_allocator(
        &self,
        index: usize,
        allocator: &mut Allocator,
    ) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut name_ptr = ptr::null_mut::<c_char>();
        bail_on_error!(ORT_API.SessionGetOverridableInitializerName.unwrap()(
            self.raw.as_ptr(),
            index.try_into()?,
            allocator,
            &mut name_ptr,
        ));
        unsafe {
            let name = CStr::from_ptr(name_ptr).to_str()?.to_string();
            (*allocator).Free.unwrap()(allocator, name_ptr as *mut c_void);
            Ok(name)
        }
    }

    pub fn input_names(&self) -> self::Result<Vec<String>> {
        self.input_names_using_allocator(&mut ALLOCATOR_WITH_DEFAULT_OPTIONS.lock().unwrap())?
            .collect::<Result<Vec<_>>>()
    }

    pub fn input_names_using_allocator<'a, 'i, 's>(
        &'s self,
        allocator: &'a mut Allocator,
    ) -> self::Result<impl Iterator<Item = self::Result<String>> + 'i>
    where
        's: 'i,
        'a: 'i,
    {
        Ok((0..self.input_count()?).map(move |i| self.input_name_using_allocator(i, allocator)))
    }

    pub fn output_names(&self) -> self::Result<Vec<String>> {
        self.output_names_using_allocator(&mut ALLOCATOR_WITH_DEFAULT_OPTIONS.lock().unwrap())?
            .collect::<Result<Vec<_>>>()
    }

    pub fn output_names_using_allocator<'a, 'i, 's>(
        &'s self,
        allocator: &'a mut Allocator,
    ) -> self::Result<impl Iterator<Item = self::Result<String>> + 'i>
    where
        's: 'i,
        'a: 'i,
    {
        Ok((0..self.output_count()?).map(move |i| self.output_name_using_allocator(i, allocator)))
    }

    pub fn overridable_initializer_names(&self) -> self::Result<Vec<String>> {
        self.overridable_initializer_names_using_allocator(
            &mut ALLOCATOR_WITH_DEFAULT_OPTIONS.lock().unwrap(),
        )?
        .collect::<Result<Vec<_>>>()
    }

    pub fn overridable_initializer_names_using_allocator<'a, 'i, 's>(
        &'s self,
        allocator: &'a mut Allocator,
    ) -> self::Result<impl Iterator<Item = self::Result<String>> + 'i>
    where
        's: 'i,
        'a: 'i,
    {
        Ok((0..self.overridable_initializer_count()?)
            .map(move |i| self.overridable_initializer_name_using_allocator(i, allocator)))
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L783-L788
    pub fn end_profiling(&mut self, allocator: &mut Allocator) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut profile_ptr = ptr::null_mut::<c_char>();
        panic_on_error!(ORT_API.SessionEndProfiling.unwrap()(
            self.raw.as_ptr(),
            allocator,
            &mut profile_ptr,
        ));
        unsafe {
            let profile = CStr::from_ptr(profile_ptr).to_str()?.to_string();
            (*allocator).Free.unwrap()(allocator, profile_ptr as *mut c_void);
            Ok(profile)
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L1032-L1035
    pub fn profiling_start_time_ns(&self) -> u64 {
        let mut time = 0;
        panic_on_error!(ORT_API.SessionGetProfilingStartTimeNs.unwrap()(
            self.raw.as_ptr(),
            &mut time,
        ));
        time
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L790-L793
    pub fn model_metadata(&self) -> self::Result<ModelMetadata> {
        let mut model_metadata = ptr::null_mut::<OrtModelMetadata>();
        bail_on_error!(ORT_API.SessionGetModelMetadata.unwrap()(
            self.raw.as_ptr(),
            &mut model_metadata,
        ));
        Ok(ModelMetadata { raw: NonNull::new(model_metadata).unwrap() })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L420-L423
    pub fn input_type_info(&self, index: usize) -> self::Result<TypeInfo> {
        let mut type_info = ptr::null_mut::<OrtTypeInfo>();
        bail_on_error!(ORT_API.SessionGetInputTypeInfo.unwrap()(
            self.raw.as_ptr(),
            index.try_into()?,
            &mut type_info,
        ));
        Ok(TypeInfo { raw: NonNull::new(type_info).unwrap() })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L425-L429
    pub fn output_type_info(&self, index: usize) -> self::Result<TypeInfo> {
        let mut type_info = ptr::null_mut::<OrtTypeInfo>();
        bail_on_error!(ORT_API.SessionGetOutputTypeInfo.unwrap()(
            self.raw.as_ptr(),
            index.try_into()?,
            &mut type_info,
        ));
        Ok(TypeInfo { raw: NonNull::new(type_info).unwrap() })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L431-L435
    pub fn overridable_initializer_type_info(&self, index: usize) -> self::Result<TypeInfo> {
        let mut type_info = ptr::null_mut::<OrtTypeInfo>();
        bail_on_error!(ORT_API.SessionGetOverridableInitializerTypeInfo.unwrap()(
            self.raw.as_ptr(),
            index.try_into()?,
            &mut type_info,
        ));
        Ok(TypeInfo { raw: NonNull::new(type_info).unwrap() })
    }
}

impl Drop for Session {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L710
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseSession.unwrap()(self.raw.as_ptr());
        }
    }
}

unsafe impl Send for Session {}

unsafe impl Sync for Session {}

#[derive(Debug)]
#[repr(transparent)]
pub struct TensorTypeAndShapeInfo {
    raw: NonNull<OrtTensorTypeAndShapeInfo>,
}

impl TensorTypeAndShapeInfo {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L524-L527
    pub fn new() -> Self {
        let mut tensor_type_info = ptr::null_mut::<OrtTensorTypeAndShapeInfo>();
        panic_on_error!(ORT_API.CreateTensorTypeAndShapeInfo.unwrap()(&mut tensor_type_info));
        Self { raw: NonNull::new(tensor_type_info).unwrap() }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L529
    pub fn set_element_type(&mut self, typ: ONNXTensorElementDataType) -> &mut Self {
        panic_on_error!(ORT_API.SetTensorElementType.unwrap()(self.raw.as_ptr(), typ));
        self
    }

    pub fn element_type(&self) -> ONNXTensorElementDataType {
        element_type(unsafe { self.raw.as_ref() })
    }

    pub fn element_count(&self) -> i64 {
        element_count(unsafe { self.raw.as_ref() })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L531-L536
    pub fn set_dimensions(&mut self, dims: &[i64]) -> &mut Self {
        panic_on_error!(ORT_API.SetDimensions.unwrap()(
            self.raw.as_ptr(),
            dims.as_ptr(),
            dims.len().try_into().unwrap(),
        ));
        self
    }

    pub fn dimensions_count(&self) -> usize {
        dimensions_count(unsafe { self.raw.as_ref() })
    }

    pub fn dimensions(&self) -> Vec<i64> {
        dimensions(unsafe { self.raw.as_ref() })
    }

    pub fn symbolic_dimensions(&self) -> self::Result<Vec<&str>> {
        symbolic_dimensions(unsafe { self.raw.as_ref() })
    }
}

impl Default for TensorTypeAndShapeInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TensorTypeAndShapeInfo {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L714
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseTensorTypeAndShapeInfo.unwrap()(self.raw.as_ptr());
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UnownedTensorTypeAndShapeInfo<'a> {
    raw: &'a OrtTensorTypeAndShapeInfo,
}

impl<'a> UnownedTensorTypeAndShapeInfo<'a> {
    pub fn element_type(&self) -> ONNXTensorElementDataType {
        element_type(self.raw)
    }

    pub fn element_count(&self) -> i64 {
        element_count(self.raw)
    }

    pub fn dimensions_count(&self) -> usize {
        dimensions_count(self.raw)
    }

    pub fn dimensions(&self) -> Vec<i64> {
        dimensions(self.raw)
    }

    pub fn symbolic_dimensions(&self) -> self::Result<Vec<&str>> {
        symbolic_dimensions(self.raw)
    }
}

/// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L538-L539
fn element_type(tensor_type_info: &OrtTensorTypeAndShapeInfo) -> ONNXTensorElementDataType {
    let mut typ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    panic_on_error!(ORT_API.GetTensorElementType.unwrap()(tensor_type_info, &mut typ));
    typ
}

/// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L546-L555
fn element_count(tensor_type_info: &OrtTensorTypeAndShapeInfo) -> i64 {
    let mut count = 0;
    panic_on_error!(ORT_API.GetTensorShapeElementCount.unwrap()(
        tensor_type_info,
        // https://github.com/microsoft/onnxruntime/issues/3132
        &mut count as *mut i64 as *mut _,
    ));
    count
}

/// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L540
fn dimensions_count(tensor_type_info: &OrtTensorTypeAndShapeInfo) -> usize {
    let mut count = 0;
    panic_on_error!(ORT_API.GetDimensionsCount.unwrap()(tensor_type_info, &mut count));
    count.try_into().unwrap()
}

/// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L541-L542
fn dimensions(tensor_type_info: &OrtTensorTypeAndShapeInfo) -> Vec<i64> {
    let mut dims = vec![0; dimensions_count(tensor_type_info)];
    panic_on_error!(ORT_API.GetDimensions.unwrap()(
        tensor_type_info,
        dims.as_mut_ptr(),
        dims.len().try_into().unwrap(),
    ));
    dims
}

/// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L543-L544
fn symbolic_dimensions(tensor_type_info: &OrtTensorTypeAndShapeInfo) -> self::Result<Vec<&str>> {
    let mut dimensions = vec![ptr::null::<c_char>(); dimensions_count(tensor_type_info)];
    panic_on_error!(ORT_API.GetSymbolicDimensions.unwrap()(
        tensor_type_info,
        dimensions.as_mut_ptr(),
        dimensions.len().try_into()?,
    ));
    Ok((dimensions.iter())
        .map(|&dimension| unsafe { CStr::from_ptr(dimension) }.to_str())
        .collect::<Result<Vec<_>, _>>()?)
}

#[derive(Debug)]
pub struct TypeInfo {
    raw: NonNull<OrtTypeInfo>,
}

impl TypeInfo {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L513-L517
    pub fn cast_to_tensor_type_info(&self) -> Option<UnownedTensorTypeAndShapeInfo<'_>> {
        let mut tensor_info = ptr::null::<OrtTensorTypeAndShapeInfo>();
        panic_on_error!(ORT_API.CastTypeInfoToTensorInfo.unwrap()(
            self.raw.as_ptr(),
            &mut tensor_info,
        ));
        if !tensor_info.is_null() {
            Some(UnownedTensorTypeAndShapeInfo { raw: unsafe { &*tensor_info } })
        } else {
            None
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L519-L522
    pub fn onnx_type(&self) -> ONNXType {
        let mut typ = ONNX_TYPE_UNKNOWN;
        panic_on_error!(ORT_API.GetOnnxTypeFromTypeInfo.unwrap()(self.raw.as_ptr(), &mut typ));
        typ
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L722-L728
    pub fn denotation(&self) -> self::Result<&str> {
        let mut denotation = ptr::null::<c_char>();
        let mut length = 0;
        panic_on_error!(ORT_API.GetDenotationFromTypeInfo.unwrap()(
            self.raw.as_ptr(),
            &mut denotation,
            &mut length,
        ));
        Ok(unsafe { CStr::from_ptr(denotation) }.to_str()?)
    }
}

impl Drop for TypeInfo {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L713
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseTypeInfo.unwrap()(self.raw.as_ptr());
        }
    }
}

pub trait AsONNXTensorElementDataType {
    fn as_onnx_tensor_element_data_type() -> ONNXTensorElementDataType;
}

#[macro_export]
macro_rules! impl_AsONNXTensorElementDataType {
    ($typ:ty, $onnx_tensor_element_data_type:expr$(,)?) => {
        impl $crate::AsONNXTensorElementDataType for $typ {
            fn as_onnx_tensor_element_data_type() -> onnxruntime_sys::ONNXTensorElementDataType {
                $onnx_tensor_element_data_type
            }
        }
    };
}

impl_AsONNXTensorElementDataType!(f32, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
impl_AsONNXTensorElementDataType!(u8, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
impl_AsONNXTensorElementDataType!(i8, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
impl_AsONNXTensorElementDataType!(u16, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
impl_AsONNXTensorElementDataType!(i16, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
impl_AsONNXTensorElementDataType!(i32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
impl_AsONNXTensorElementDataType!(i64, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
impl_AsONNXTensorElementDataType!(bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
impl_AsONNXTensorElementDataType!(f64, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
impl_AsONNXTensorElementDataType!(u32, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);
impl_AsONNXTensorElementDataType!(u64, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64);

#[derive(Debug)]
#[repr(transparent)]
pub struct Value<'d> {
    raw: *mut OrtValue,
    phantom: PhantomData<&'d ()>,
}

impl<'d> Value<'d> {
    pub fn new_tensor(
        shape: &[i64],
        element_type: ONNXTensorElementDataType,
    ) -> self::Result<Self> {
        Self::new_tensor_using_allocator(
            &mut ALLOCATOR_WITH_DEFAULT_OPTIONS.lock().unwrap(),
            shape,
            element_type,
        )
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L466-L472
    pub fn new_tensor_using_allocator(
        allocator: &mut Allocator,
        shape: &[i64],
        element_type: ONNXTensorElementDataType,
    ) -> self::Result<Self> {
        let mut value = ptr::null_mut::<OrtValue>();
        bail_on_error!(ORT_API.CreateTensorAsOrtValue.unwrap()(
            allocator.raw.as_ptr(),
            shape.as_ptr(),
            shape.len().try_into()?,
            element_type,
            &mut value,
        ));
        Ok(Value { raw: value, phantom: PhantomData })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L474-L481
    pub fn new_tensor_with_data<T: AsONNXTensorElementDataType>(
        memory_info: &MemoryInfo,
        data: &'d mut [T],
        shape: &[i64],
    ) -> self::Result<Self> {
        let mut value = ptr::null_mut::<OrtValue>();
        bail_on_error!(ORT_API.CreateTensorWithDataAsOrtValue.unwrap()(
            memory_info.raw.as_ptr(),
            data.as_mut_ptr() as *mut c_void,
            mem::size_of_val(data).try_into()?,
            shape.as_ptr(),
            shape.len().try_into()?,
            T::as_onnx_tensor_element_data_type(),
            &mut value,
        ));
        Ok(Value { raw: value, phantom: PhantomData })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L483-L486
    pub fn is_tensor(&self) -> bool {
        let mut result = 0;
        panic_on_error!(ORT_API.IsTensor.unwrap()(self.raw, &mut result));
        result != 0
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L488-L490
    pub fn tensor_data<T>(&self) -> self::Result<*const T> {
        let mut data = ptr::null_mut::<T>();
        bail_on_error!(ORT_API.GetTensorMutableData.unwrap()(
            self.raw,
            &mut data as *mut *mut T as *mut *mut c_void,
        ));
        Ok(data)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L488-L490
    pub fn tensor_data_mut<T>(&mut self) -> self::Result<*mut T> {
        let mut data = ptr::null_mut::<T>();
        bail_on_error!(ORT_API.GetTensorMutableData.unwrap()(
            self.raw,
            &mut data as *mut *mut T as *mut *mut c_void,
        ));
        Ok(data)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L1000-L1012
    ///
    /// # Safety
    ///
    /// `T` must be the same element type as specified when `self` was created.
    pub unsafe fn get<T>(&self, indices: &[i64]) -> self::Result<&T> {
        let mut element = ptr::null_mut::<T>();
        bail_on_error!(ORT_API.TensorAt.unwrap()(
            self.raw,
            indices.as_ptr(),
            indices.len().try_into()?,
            &mut element as *mut *mut T as *mut *mut c_void,
        ));
        Ok(&*element)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L1000-L1012
    ///
    /// # Safety
    ///
    /// `T` must be the same element type as specified when `self` was created.
    pub unsafe fn get_mut<T>(&mut self, indices: &[i64]) -> self::Result<&mut T> {
        let mut element = ptr::null_mut::<T>();
        bail_on_error!(ORT_API.TensorAt.unwrap()(
            self.raw,
            indices.as_ptr(),
            indices.len().try_into()?,
            &mut element as *mut *mut T as *mut *mut c_void,
        ));
        Ok(&mut *element)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L562-L567
    pub fn type_info(&self) -> self::Result<Option<TypeInfo>> {
        let mut type_info = ptr::null_mut::<OrtTypeInfo>();
        bail_on_error!(ORT_API.GetTypeInfo.unwrap()(self.raw, &mut type_info));
        // https://github.com/microsoft/onnxruntime/blob/3433576fd39bb6451fb520e208a2f34a07ba4c7b/onnxruntime/core/framework/tensor_type_and_shape.cc#L243-L249
        //
        // > // TODO: This is consistent with the previous implementation but inconsistent with
        // > // GetValueType which returns ONNX_TYPE_UNKNOWN if v->Type() is null. Should we
        // > // instead just call OrtTypeInfo::FromOrtValue and return an OrtTypeInfo value in
        // > // 'out' with type set to ONNX_TYPE_UNKNOWN? Or is the inconsistency fine?
        // > if (v->Type() == nullptr) {
        // >   *out = nullptr;
        // >   return nullptr;
        // > }
        Ok(if !type_info.is_null() {
            Some(TypeInfo { raw: NonNull::new(type_info).unwrap() })
        } else {
            None
        })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L557-L560
    pub fn tensor_type_info(&self) -> self::Result<TensorTypeAndShapeInfo> {
        let mut tensor_type_info = ptr::null_mut::<OrtTensorTypeAndShapeInfo>();
        bail_on_error!(ORT_API.GetTensorTypeAndShape.unwrap()(self.raw, &mut tensor_type_info));
        Ok(TensorTypeAndShapeInfo { raw: NonNull::new(tensor_type_info).unwrap() })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L569
    pub fn value_type(&self) -> self::Result<ONNXType> {
        let mut value_type = ONNX_TYPE_UNKNOWN;
        bail_on_error!(ORT_API.GetValueType.unwrap()(self.raw, &mut value_type));
        Ok(value_type)
    }
}

impl<'d> Default for Value<'d> {
    fn default() -> Self {
        Value { raw: ptr::null_mut::<OrtValue>(), phantom: PhantomData }
    }
}

impl<'d> Drop for Value<'d> {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L711
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseValue.unwrap()(self.raw);
        }
    }
}

#[derive(Debug)]
pub struct MemoryInfo {
    raw: NonNull<OrtMemoryInfo>,
}

impl MemoryInfo {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L571-L572
    pub fn new(
        name: &str,
        allocator_type: OrtAllocatorType,
        id: i32,
        memory_type: OrtMemType,
    ) -> self::Result<Self> {
        let name = CString::new(name)?;
        let mut memory_info = ptr::null_mut::<OrtMemoryInfo>();
        bail_on_error!(ORT_API.CreateMemoryInfo.unwrap()(
            name.as_ptr(),
            allocator_type,
            id,
            memory_type,
            &mut memory_info,
        ));
        Ok(MemoryInfo { raw: NonNull::new(memory_info).unwrap() })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L574-L578
    pub fn new_for_cpu(allocator_type: OrtAllocatorType, memory_type: OrtMemType) -> Self {
        let mut memory_info = ptr::null_mut::<OrtMemoryInfo>();
        panic_on_error!(ORT_API.CreateCpuMemoryInfo.unwrap()(
            allocator_type,
            memory_type,
            &mut memory_info,
        ));
        MemoryInfo { raw: NonNull::new(memory_info).unwrap() }
    }

    pub fn allocator_name(&self) -> self::Result<&str> {
        allocator_name(unsafe { self.raw.as_ref() })
    }

    pub fn allocator_type(&self) -> OrtAllocatorType {
        allocator_type(unsafe { self.raw.as_ref() })
    }

    pub fn device_id(&self) -> c_int {
        device_id(unsafe { self.raw.as_ref() })
    }

    pub fn memory_type(&self) -> OrtMemType {
        memory_type(unsafe { self.raw.as_ref() })
    }
}

impl Drop for MemoryInfo {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L709
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseMemoryInfo.unwrap()(self.raw.as_ptr());
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UnownedMemoryInfo<'a> {
    raw: &'a OrtMemoryInfo,
}

impl<'a> UnownedMemoryInfo<'a> {
    pub fn allocator_name(&self) -> self::Result<&str> {
        allocator_name(self.raw)
    }

    pub fn allocator_type(&self) -> OrtAllocatorType {
        allocator_type(self.raw)
    }

    pub fn device_id(&self) -> c_int {
        device_id(self.raw)
    }

    pub fn memory_type(&self) -> OrtMemType {
        memory_type(self.raw)
    }
}

/// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L586-L592
fn allocator_name(memory_info: &OrtMemoryInfo) -> self::Result<&str> {
    let mut name = ptr::null::<c_char>();
    panic_on_error!(ORT_API.MemoryInfoGetName.unwrap()(memory_info, &mut name));
    Ok(unsafe { CStr::from_ptr(name) }.to_str()?)
}

/// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L586-L592
fn allocator_type(memory_info: &OrtMemoryInfo) -> OrtAllocatorType {
    let mut allocator_type = OrtAllocatorType::Invalid;
    panic_on_error!(ORT_API.MemoryInfoGetType.unwrap()(memory_info, &mut allocator_type));
    allocator_type
}

/// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L586-L592
fn device_id(memory_info: &OrtMemoryInfo) -> c_int {
    let mut id = 0;
    panic_on_error!(ORT_API.MemoryInfoGetId.unwrap()(memory_info, &mut id));
    id
}

/// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L586-L592
fn memory_type(memory_info: &OrtMemoryInfo) -> OrtMemType {
    let mut memory_type = OrtMemTypeDefault;
    panic_on_error!(ORT_API.MemoryInfoGetMemType.unwrap()(memory_info, &mut memory_type));
    memory_type
}

#[derive(Debug)]
pub struct Allocator {
    raw: NonNull<OrtAllocator>,
}

impl Allocator {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L904-L912
    pub fn new(session: &Session, memory_info: &MemoryInfo) -> self::Result<Self> {
        let mut allocator = ptr::null_mut::<OrtAllocator>();
        bail_on_error!(ORT_API.CreateAllocator.unwrap()(
            session.raw.as_ptr(),
            memory_info.raw.as_ptr(),
            &mut allocator,
        ));
        Ok(Self { raw: NonNull::new(allocator).unwrap() })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L594
    pub fn alloc<T>(&mut self) -> *mut T {
        let mut ptr = ptr::null_mut::<T>();
        panic_on_error!(ORT_API.AllocatorAlloc.unwrap()(
            self.raw.as_ptr(),
            mem::size_of::<T>().try_into().unwrap(),
            &mut ptr as *mut *mut T as *mut *mut c_void,
        ));
        ptr
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L595
    ///
    /// # Safety
    ///
    /// The memory pointed by `ptr` must have been allocated by `Allocator::alloc`.
    pub unsafe fn free<T>(&mut self, ptr: *mut T) {
        panic_on_error!(ORT_API.AllocatorFree.unwrap()(self.raw.as_ptr(), ptr as *mut c_void));
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L596
    pub fn memory_info(&self) -> UnownedMemoryInfo<'_> {
        let mut memory_info = ptr::null::<OrtMemoryInfo>();
        panic_on_error!(ORT_API.AllocatorGetInfo.unwrap()(self.raw.as_ptr(), &mut memory_info));
        UnownedMemoryInfo { raw: unsafe { &*memory_info } }
    }
}

impl Drop for Allocator {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L914-L915
    fn drop(&mut self) {
        unsafe { ORT_API.ReleaseAllocator.unwrap()(self.raw.as_ptr()) }
    }
}

unsafe impl Send for Allocator {}

unsafe impl Sync for Allocator {}

#[derive(Debug)]
pub struct IoBinding<'s> {
    raw: NonNull<OrtIoBinding>,
    phantom: PhantomData<&'s Session>,
}

impl<'s> IoBinding<'s> {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L919-L922
    pub fn new(session: &'s mut Session) -> self::Result<Self> {
        let mut io_binding = ptr::null_mut::<OrtIoBinding>();
        bail_on_error!(ORT_API.CreateIoBinding.unwrap()(session.raw.as_ptr(), &mut io_binding));
        Ok(Self { raw: NonNull::new(io_binding).unwrap(), phantom: PhantomData })
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L927-L935
    pub fn bind_input(&mut self, name: &str, value: &Value<'_>) -> self::Result<&mut Self> {
        let name = CString::new(name)?;
        bail_on_error!(ORT_API.BindInput.unwrap()(self.raw.as_ptr(), name.as_ptr(), value.raw));
        Ok(self)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L937-L946
    pub fn bind_output(&mut self, name: &str, value: &Value<'_>) -> self::Result<&mut Self> {
        let name = CString::new(name)?;
        panic_on_error!(ORT_API.BindOutput.unwrap()(self.raw.as_ptr(), name.as_ptr(), value.raw));
        Ok(self)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L948-L959
    pub fn bind_output_to_device(
        &mut self,
        name: &str,
        memory_info: &MemoryInfo,
    ) -> self::Result<&mut Self> {
        let name = CString::new(name)?;
        bail_on_error!(ORT_API.BindOutputToDevice.unwrap()(
            self.raw.as_ptr(),
            name.as_ptr(),
            memory_info.raw.as_ptr(),
        ));
        Ok(self)
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L961-L977
    pub fn bound_output_names(&self, allocator: &mut Allocator) -> self::Result<Vec<String>> {
        let allocator = allocator.raw.as_ptr();
        let mut buffer = ptr::null_mut::<c_char>();
        let mut lengths = ptr::null_mut();
        let mut count = 0;
        bail_on_error!(ORT_API.GetBoundOutputNames.unwrap()(
            self.raw.as_ptr(),
            allocator,
            &mut buffer,
            &mut lengths,
            &mut count,
        ));
        if !buffer.is_null() {
            unsafe {
                let count = count.try_into()?;
                let mut names = Vec::with_capacity(count);
                let mut ptr = buffer;
                for &length in slice::from_raw_parts(lengths, count) {
                    let length = length.try_into()?;
                    let name = str::from_utf8(slice::from_raw_parts(ptr as *const u8, length))?
                        .to_string();
                    names.push(name);
                    ptr = ptr.add(length);
                }
                let free = (*allocator).Free.as_ref().unwrap();
                free(allocator, lengths as *mut c_void);
                free(allocator, buffer as *mut c_void);
                Ok(names)
            }
        } else {
            Ok(Vec::new())
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L979-L993
    pub fn bound_output_values(
        &self,
        allocator: &mut Allocator,
    ) -> self::Result<Vec<Value<'static>>> {
        let allocator = allocator.raw.as_ptr();
        let mut values_ptr = ptr::null_mut::<*mut OrtValue>();
        let mut count = 0;
        bail_on_error!(ORT_API.GetBoundOutputValues.unwrap()(
            self.raw.as_ptr(),
            allocator,
            &mut values_ptr,
            &mut count,
        ));
        if !values_ptr.is_null() {
            unsafe {
                let values_slice = slice::from_raw_parts(values_ptr, count.try_into()?);
                let values = (values_slice.iter())
                    .map(|&value| Value { raw: value, phantom: PhantomData })
                    .collect::<Vec<_>>();
                (*allocator).Free.as_ref().unwrap()(allocator, values_ptr as *mut c_void);
                Ok(values)
            }
        } else {
            Ok(Vec::new())
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L995-L998
    pub fn clear_bound_inputs(&mut self) {
        unsafe {
            ORT_API.ClearBoundInputs.unwrap()(self.raw.as_ptr());
        }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L995-L998
    pub fn clear_bound_outputs(&mut self) {
        unsafe {
            ORT_API.ClearBoundOutputs.unwrap()(self.raw.as_ptr());
        }
    }
}

impl<'s> Drop for IoBinding<'s> {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L924-L925
    fn drop(&mut self) {
        unsafe { ORT_API.ReleaseIoBinding.unwrap()(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct ThreadingOptions {
    raw: NonNull<OrtThreadingOptions>,
}

impl ThreadingOptions {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L833
    pub fn new() -> Self {
        let mut threading_options = ptr::null_mut::<OrtThreadingOptions>();
        panic_on_error!(ORT_API.CreateThreadingOptions.unwrap()(&mut threading_options));
        Self { raw: NonNull::new(threading_options).unwrap() }
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L1037-L1043
    pub fn set_global_intra_op_num_threads(&mut self, intra_op_num_threads: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetGlobalIntraOpNumThreads.unwrap()(
            self.raw.as_ptr(),
            intra_op_num_threads,
        ));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L1037-L1043
    pub fn set_global_inter_op_num_threads(&mut self, inter_op_num_threads: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetGlobalIntraOpNumThreads.unwrap()(
            self.raw.as_ptr(),
            inter_op_num_threads,
        ));
        self
    }

    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L1045-L1053
    pub fn set_global_spin_control(&mut self, allow_spinning: bool) -> &mut Self {
        panic_on_error!(ORT_API.SetGlobalSpinControl.unwrap()(
            self.raw.as_ptr(),
            allow_spinning as _,
        ));
        self
    }
}

impl Default for ThreadingOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ThreadingOptions {
    /// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L835
    fn drop(&mut self) {
        unsafe { ORT_API.ReleaseThreadingOptions.unwrap()(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct ArenaCfg {
    raw: NonNull<OrtArenaCfg>,
}

/// https://github.com/microsoft/onnxruntime/blob/ad7cc541ff96f32cc6dd27040f27a451b36acdef/include/onnxruntime/core/session/onnxruntime_c_api.h#L853-L862
pub fn available_providers() -> self::Result<Vec<String>> {
    let mut providers_ptr = ptr::null_mut::<*mut c_char>();
    let mut num_providers = 0;
    panic_on_error!(ORT_API.GetAvailableProviders.unwrap()(&mut providers_ptr, &mut num_providers));
    if !providers_ptr.is_null() {
        unsafe {
            let providers_slice = slice::from_raw_parts(providers_ptr, num_providers.try_into()?);
            let providers = (providers_slice.iter())
                .map(|&provider| CStr::from_ptr(provider).to_str().map(str::to_string))
                .collect::<Result<Vec<_>, _>>()?;
            panic_on_error!(ORT_API.ReleaseAvailableProviders.unwrap()(
                providers_ptr,
                num_providers,
            ));
            Ok(providers)
        }
    } else {
        Ok(Vec::new())
    }
}

// Not implemented:
//
// ORT_API2_STATUS(AddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);
// ORT_API2_STATUS(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info, _Outptr_result_maybenull_ const OrtMapTypeInfo** out);
// ORT_API2_STATUS(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info, _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out);
// ORT_API2_STATUS(CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);
// ORT_API2_STATUS(CreateOpaqueValue, _In_z_ const char* domain_name, _In_z_ const char* type_name, _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(CreateValue, _In_reads_(num_values) const OrtValue* const* in, size_t num_values, enum ONNXType value_type, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ const OrtCustomOp* op);
// ORT_API2_STATUS(FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len);
// ORT_API2_STATUS(FillStringTensorElement, _Inout_ OrtValue* value, _In_ const char* s, size_t index);
// ORT_API2_STATUS(GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);
// ORT_API2_STATUS(GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);
// ORT_API2_STATUS(GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name, _In_ const OrtValue* in, _Out_ void* data_container, size_t data_container_size);
// ORT_API2_STATUS(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info, _Outptr_ OrtTypeInfo** type_info);
// ORT_API2_STATUS(GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s, size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len);
// ORT_API2_STATUS(GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* len);
// ORT_API2_STATUS(GetStringTensorElement, _In_ const OrtValue* value, size_t s_len, size_t index, _Out_writes_bytes_all_(s_len) void* s);
// ORT_API2_STATUS(GetStringTensorElementLength, _In_ const OrtValue* value, size_t index, _Out_ size_t* out);
// ORT_API2_STATUS(GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out);
// ORT_API2_STATUS(KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index, _Out_ const OrtValue** out);
// ORT_API2_STATUS(KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
// ORT_API2_STATUS(KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
// ORT_API2_STATUS(KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out);
// ORT_API2_STATUS(KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out);
// ORT_API2_STATUS(KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out, _Inout_ size_t* size);
// ORT_API2_STATUS(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, void** library_handle);
