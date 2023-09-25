//! [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h)

#![doc(html_root_url = "https://furiosa-ai.github.io/onnxrt/0.21.0/onnxrt")]
#![warn(rust_2018_idioms)]

#[cfg(target_family = "unix")]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_family = "windows")]
use std::os::windows::ffi::OsStrExt;
use std::{
    convert::TryInto,
    ffi::{CStr, CString},
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    mem,
    os::raw::{c_char, c_int, c_void},
    path::Path,
    ptr::{self, NonNull},
    slice, str,
    sync::{Arc, Mutex},
};

use once_cell::sync::Lazy;
pub use onnxrt_sys::{
    ExecutionMode, GraphOptimizationLevel, ONNXTensorElementDataType, ONNXType, OrtAllocatorType,
    OrtErrorCode, OrtLanguageProjection, OrtLoggingLevel, OrtMemType, OrtMemoryInfoDeviceType,
    OrtPrepackedWeightsContainer, ORT_API_VERSION,
};
use onnxrt_sys::{
    ONNXTensorElementDataType::{
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    },
    ONNXType::ONNX_TYPE_UNKNOWN,
    OrtAllocator, OrtApi, OrtArenaCfg, OrtCUDAProviderOptionsV2, OrtCustomCreateThreadFn,
    OrtCustomJoinThreadFn, OrtEnv, OrtGetApiBase, OrtIoBinding, OrtLoggingFunction,
    OrtMemType::OrtMemTypeDefault,
    OrtMemoryInfo,
    OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
    OrtModelMetadata, OrtRunOptions, OrtSession, OrtSessionOptions, OrtTensorTypeAndShapeInfo,
    OrtThreadingOptions, OrtTypeInfo, OrtValue,
};

macro_rules! bail_on_error {
    ($x:expr) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let status = $x;
            if !status.is_null() {
                let code = $crate::ORT_API.GetErrorCode.unwrap()(status);
                assert_ne!(code, onnxrt_sys::OrtErrorCode::ORT_OK);
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
        assert_eq!(status, std::ptr::null_mut::<onnxrt_sys::OrtStatus>());
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
    FromVecWithNulError { source: std::ffi::FromVecWithNulError },
    IntoStringError { source: std::ffi::IntoStringError },
    IoError { source: std::io::Error },
    NulError { source: std::ffi::NulError },
    OrtError { code: OrtErrorCode, message: String },
    TryFromIntError { source: std::num::TryFromIntError },
    Utf8Error { source: std::str::Utf8Error },
}

impl Display for self::Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::FromVecWithNulError { source } => source.fmt(f),
            Error::IntoStringError { source } => source.fmt(f),
            Error::IoError { source } => source.fmt(f),
            Error::NulError { source } => source.fmt(f),
            Error::OrtError { message, .. } => write!(f, "{message}"),
            Error::TryFromIntError { source } => source.fmt(f),
            Error::Utf8Error { source } => source.fmt(f),
        }
    }
}

impl std::error::Error for self::Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::FromVecWithNulError { source } => Some(source),
            Error::IntoStringError { source } => Some(source),
            Error::IoError { source } => Some(source),
            Error::NulError { source } => Some(source),
            Error::OrtError { .. } => None,
            Error::TryFromIntError { source } => Some(source),
            Error::Utf8Error { source } => Some(source),
        }
    }
}

impl From<std::ffi::FromVecWithNulError> for self::Error {
    fn from(source: std::ffi::FromVecWithNulError) -> Self {
        Self::FromVecWithNulError { source }
    }
}

impl From<std::ffi::IntoStringError> for self::Error {
    fn from(source: std::ffi::IntoStringError) -> Self {
        Self::IntoStringError { source }
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
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L746-L754)
    pub fn new(logging_level: OrtLoggingLevel, log_id: &str) -> self::Result<Self> {
        let mut env = ptr::null_mut::<OrtEnv>();
        let log_id = CString::new(log_id)?;
        bail_on_error!(ORT_API.CreateEnv.unwrap()(logging_level, log_id.as_ptr(), &mut env));
        Ok(Self { raw: NonNull::new(env).unwrap() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2084-L2098)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L756-L768)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2527-L2543)
    pub fn new_with_custom_logger_and_global_thread_pools<T>(
        logging_function: OrtLoggingFunction,
        logger_param: Option<&'static mut T>,
        logging_level: OrtLoggingLevel,
        log_id: &str,
        threading_options: &ThreadingOptions,
    ) -> self::Result<Self> {
        let mut env = ptr::null_mut::<OrtEnv>();
        let log_id = CString::new(log_id)?;
        bail_on_error!(ORT_API.CreateEnvWithCustomLoggerAndGlobalThreadPools.unwrap()(
            logging_function,
            logger_param
                .map(|param| param as *mut T as *mut c_void)
                .unwrap_or(ptr::null_mut::<c_void>()),
            logging_level,
            log_id.as_ptr(),
            threading_options.raw.as_ptr(),
            &mut env,
        ));
        Ok(Self { raw: NonNull::new(env).unwrap() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3732-L3739)
    pub fn set_custom_log_level(&mut self, logging_level: OrtLoggingLevel) {
        panic_on_error!(ORT_API.UpdateEnvWithCustomLogLevel.unwrap()(
            self.raw.as_ptr(),
            logging_level
        ));
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L770-L777)
    pub fn enable_telemetry_events(&self) {
        panic_on_error!(ORT_API.EnableTelemetryEvents.unwrap()(self.raw.as_ptr()));
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L778-L785)
    pub fn disable_telemetry_events(&self) {
        panic_on_error!(ORT_API.DisableTelemetryEvents.unwrap()(self.raw.as_ptr()));
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2429-L2440)
    pub fn set_language_projection(&self, projection: OrtLanguageProjection) {
        panic_on_error!(ORT_API.SetLanguageProjection.unwrap()(self.raw.as_ptr(), projection));
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L4311-L4325)
    pub fn create_and_register_allocator<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        provider_type: &str,
        memory_info: &MemoryInfo,
        arena_cfg: Option<&ArenaCfg>,
        provider_options_keys: &[K],
        provider_options_values: &[V],
    ) -> self::Result<()> {
        assert_eq!(provider_options_keys.len(), provider_options_values.len());
        unsafe {
            self.create_and_register_allocator_unchecked(
                provider_type,
                memory_info,
                arena_cfg,
                provider_options_keys,
                provider_options_values,
            )
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L4311-L4325)
    ///
    /// # Safety
    ///
    /// The length of `provider_options_keys` must be equal to that of `provider_options_values`.
    pub unsafe fn create_and_register_allocator_unchecked<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        provider_type: &str,
        memory_info: &MemoryInfo,
        arena_cfg: Option<&ArenaCfg>,
        provider_options_keys: &[K],
        provider_options_values: &[V],
    ) -> self::Result<()> {
        let provider_type_c_string = CString::new(provider_type)?;
        let provider_options_keys_c_string = (provider_options_keys.iter())
            .map(AsRef::as_ref)
            .map(CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        let provider_options_values_c_string = (provider_options_values.iter())
            .map(AsRef::as_ref)
            .map(CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        self.create_and_register_allocator_with_c_str_unchecked(
            &provider_type_c_string,
            memory_info,
            arena_cfg,
            &provider_options_keys_c_string,
            &provider_options_values_c_string,
        )
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L4311-L4325)
    pub fn create_and_register_allocator_with_c_str<K: AsRef<CStr>, V: AsRef<CStr>>(
        &mut self,
        provider_type: &CStr,
        memory_info: &MemoryInfo,
        arena_cfg: Option<&ArenaCfg>,
        provider_options_keys: &[K],
        provider_options_values: &[V],
    ) -> self::Result<()> {
        assert_eq!(provider_options_keys.len(), provider_options_values.len());
        unsafe {
            self.create_and_register_allocator_with_c_str_unchecked(
                provider_type,
                memory_info,
                arena_cfg,
                provider_options_keys,
                provider_options_values,
            )
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L4311-L4325)
    ///
    /// # Safety
    ///
    /// The length of `provider_options_keys` must be equal to that of `provider_options_values`.
    pub unsafe fn create_and_register_allocator_with_c_str_unchecked<
        K: AsRef<CStr>,
        V: AsRef<CStr>,
    >(
        &mut self,
        provider_type: &CStr,
        memory_info: &MemoryInfo,
        arena_cfg: Option<&ArenaCfg>,
        provider_options_keys: &[K],
        provider_options_values: &[V],
    ) -> self::Result<()> {
        let provider_type_c_char = provider_type.as_ptr();
        let provider_options_keys_c_char =
            provider_options_keys.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        let provider_options_values_c_char =
            provider_options_values.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        self.create_and_register_allocator_with_c_chars_with_nul(
            provider_type_c_char,
            memory_info,
            arena_cfg,
            &provider_options_keys_c_char,
            &provider_options_values_c_char,
        )
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L4311-L4325)
    pub fn create_and_register_allocator_with_bytes_with_nul(
        &mut self,
        provider_type: &[u8],
        memory_info: &MemoryInfo,
        arena_cfg: Option<&ArenaCfg>,
        provider_options_keys: &[&[u8]],
        provider_options_values: &[&[u8]],
    ) -> self::Result<()> {
        assert_eq!(provider_options_keys.len(), provider_options_values.len());
        assert!(provider_type.ends_with(&[b'\0']));
        assert!(provider_options_keys.iter().all(|x| x.ends_with(&[b'\0'])));
        assert!(provider_options_values.iter().all(|x| x.ends_with(&[b'\0'])));
        unsafe {
            self.create_and_register_allocator_with_bytes_with_nul_unchecked(
                provider_type,
                memory_info,
                arena_cfg,
                provider_options_keys,
                provider_options_values,
            )
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L4311-L4325)
    ///
    /// # Safety
    ///
    /// The length of `provider_options_keys` must be equal to that of `provider_options_values`.
    /// `provider_type` and every slice in `provider_options_keys` and `provider_options_values`
    /// must be a null-terminated string.
    pub unsafe fn create_and_register_allocator_with_bytes_with_nul_unchecked(
        &mut self,
        provider_type: &[u8],
        memory_info: &MemoryInfo,
        arena_cfg: Option<&ArenaCfg>,
        provider_options_keys: &[&[u8]],
        provider_options_values: &[&[u8]],
    ) -> self::Result<()> {
        let provider_type_c_char = provider_type.as_ptr() as *const c_char;
        let provider_options_keys_c_char =
            provider_options_keys.iter().map(|x| x.as_ptr() as *const c_char).collect::<Vec<_>>();
        let provider_options_values_c_char =
            provider_options_values.iter().map(|x| x.as_ptr() as *const c_char).collect::<Vec<_>>();
        self.create_and_register_allocator_with_c_chars_with_nul(
            provider_type_c_char,
            memory_info,
            arena_cfg,
            &provider_options_keys_c_char,
            &provider_options_values_c_char,
        )
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L4311-L4325)
    ///
    /// # Safety
    ///
    /// The length of `provider_options_keys` must be equal to that of `provider_options_values`.
    /// `provider_type` and every pointer in `provider_options_keys` and `provider_options_values`
    /// must be a null-terminated string.
    pub unsafe fn create_and_register_allocator_with_c_chars_with_nul(
        &mut self,
        provider_type: *const c_char,
        memory_info: &MemoryInfo,
        arena_cfg: Option<&ArenaCfg>,
        provider_options_keys: &[*const c_char],
        provider_options_values: &[*const c_char],
    ) -> self::Result<()> {
        debug_assert_eq!(provider_options_keys.len(), provider_options_values.len());
        bail_on_error!(ORT_API.CreateAndRegisterAllocatorV2.unwrap()(
            self.raw.as_ptr(),
            provider_type,
            memory_info.raw.as_ptr(),
            arena_cfg.map(|cfg| cfg.raw.as_ptr()).unwrap_or(ptr::null_mut::<OrtArenaCfg>()),
            provider_options_keys.as_ptr(),
            provider_options_values.as_ptr(),
            provider_options_keys.len(),
        ));
        Ok(())
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2959-L2974)
    pub fn register_allocator(&mut self, allocator: &mut Allocator) -> self::Result<()> {
        bail_on_error!(ORT_API.RegisterAllocator.unwrap()(
            self.raw.as_ptr(),
            allocator.raw.as_ptr(),
        ));
        Ok(())
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2976-L2987)
    pub fn unregister_allocator(&mut self, memory_info: &MemoryInfo) -> self::Result<()> {
        bail_on_error!(ORT_API.UnregisterAllocator.unwrap()(
            self.raw.as_ptr(),
            memory_info.raw.as_ptr(),
        ));
        Ok(())
    }
}

impl Drop for Env {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1843)
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
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1194-L1200)
    pub fn new() -> Self {
        let mut options = ptr::null_mut::<OrtRunOptions>();
        panic_on_error!(ORT_API.CreateRunOptions.unwrap()(&mut options));
        Self { raw: NonNull::new(options).unwrap() }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1202-L1211)
    pub fn set_log_verbosity_level(&mut self, level: i32) -> &mut Self {
        panic_on_error!(ORT_API.RunOptionsSetRunLogVerbosityLevel.unwrap()(
            self.raw.as_ptr(),
            level
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1233-L1243)
    pub fn log_verbosity_level(&self) -> i32 {
        let mut level = 0;
        panic_on_error!(ORT_API.RunOptionsGetRunLogVerbosityLevel.unwrap()(
            self.raw.as_ptr(),
            &mut level,
        ));
        level
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1213-L1220)
    pub fn set_log_severity_level(&mut self, level: i32) -> &mut Self {
        panic_on_error!(ORT_API.RunOptionsSetRunLogSeverityLevel.unwrap()(
            self.raw.as_ptr(),
            level
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1245-L1252)
    pub fn log_severity_level(&self) -> i32 {
        let mut level = 0;
        panic_on_error!(ORT_API.RunOptionsGetRunLogSeverityLevel.unwrap()(
            self.raw.as_ptr(),
            &mut level,
        ));
        level
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1222-L1231)
    pub fn set_tag(&mut self, tag: &str) -> self::Result<&mut Self> {
        let tag = CString::new(tag)?;
        panic_on_error!(ORT_API.RunOptionsSetRunTag.unwrap()(self.raw.as_ptr(), tag.as_ptr()));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1254-L1265)
    pub fn tag(&self) -> self::Result<&str> {
        let mut tag = ptr::null::<c_char>();
        panic_on_error!(ORT_API.RunOptionsGetRunTag.unwrap()(self.raw.as_ptr(), &mut tag));
        Ok(unsafe { CStr::from_ptr(tag) }.to_str()?)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1267-L1275)
    pub fn set_terminate(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.RunOptionsSetTerminate.unwrap()(self.raw.as_ptr()));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1277-L1285)
    pub fn unset_terminate(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.RunOptionsUnsetTerminate.unwrap()(self.raw.as_ptr()));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2781-L2794)
    pub fn set<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        config_key: K,
        config_value: V,
    ) -> self::Result<&mut Self> {
        let config_key_c_string = CString::new(config_key.as_ref())?;
        let config_value_c_string = CString::new(config_value.as_ref())?;
        unsafe {
            self.set_with_c_chars_with_nul(
                config_key_c_string.as_ptr(),
                config_value_c_string.as_ptr(),
            )
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2781-L2794)
    pub fn set_with_c_str<K: AsRef<CStr>, V: AsRef<CStr>>(
        &mut self,
        config_key: K,
        config_value: V,
    ) -> self::Result<&mut Self> {
        unsafe {
            self.set_with_c_chars_with_nul(
                config_key.as_ref().as_ptr(),
                config_value.as_ref().as_ptr(),
            )
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2781-L2794)
    pub fn set_with_bytes_with_nul(
        &mut self,
        config_key: &[u8],
        config_value: &[u8],
    ) -> self::Result<&mut Self> {
        assert!(config_key.ends_with(&[b'\0']));
        assert!(config_value.ends_with(&[b'\0']));
        unsafe { self.set_with_bytes_with_nul_unchecked(config_key, config_value) }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2781-L2794)
    ///
    /// # Safety
    ///
    /// `config_key` and `config_value` must be terminated with a null character.
    pub unsafe fn set_with_bytes_with_nul_unchecked(
        &mut self,
        config_key: &[u8],
        config_value: &[u8],
    ) -> self::Result<&mut Self> {
        self.set_with_c_chars_with_nul(
            config_key.as_ptr() as *const c_char,
            config_value.as_ptr() as *const c_char,
        )
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2781-L2794)
    ///
    /// # Safety
    ///
    /// `config_key` and `config_value` must be terminated with a null character.
    pub unsafe fn set_with_c_chars_with_nul(
        &mut self,
        config_key: *const c_char,
        config_value: *const c_char,
    ) -> self::Result<&mut Self> {
        bail_on_error!(ORT_API.AddRunConfigEntry.unwrap()(
            self.raw.as_ptr(),
            config_key,
            config_value,
        ));
        Ok(self)
    }
}

impl Default for RunOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RunOptions {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1863)
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseRunOptions.unwrap()(self.raw.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct SessionOptions {
    raw: NonNull<OrtSessionOptions>,
}

impl SessionOptions {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L849-L864)
    pub fn new() -> Self {
        let mut session_options = ptr::null_mut::<OrtSessionOptions>();
        panic_on_error!(ORT_API.CreateSessionOptions.unwrap()(&mut session_options));
        Self { raw: NonNull::new(session_options).unwrap() }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L999-L1012)
    pub fn set_intra_op_num_threads(&mut self, intra_op_num_threads: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetIntraOpNumThreads.unwrap()(
            self.raw.as_ptr(),
            intra_op_num_threads,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1014-L1026)
    pub fn set_inter_op_num_threads(&mut self, inter_op_num_threads: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetInterOpNumThreads.unwrap()(
            self.raw.as_ptr(),
            inter_op_num_threads,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L988-L997)
    pub fn set_graph_optimization_level(
        &mut self,
        graph_optimization_level: GraphOptimizationLevel,
    ) -> &mut Self {
        panic_on_error!(ORT_API.SetSessionGraphOptimizationLevel.unwrap()(
            self.raw.as_ptr(),
            graph_optimization_level,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L941-L949)
    pub fn enable_cpu_mem_arena(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.EnableCpuMemArena.unwrap()(self.raw.as_ptr()));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L951-L957)
    pub fn disable_cpu_mem_arena(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.DisableCpuMemArena.unwrap()(self.raw.as_ptr()));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L866-L874)
    pub fn set_optimized_model_file_path<P: AsRef<Path>>(
        &mut self,
        optimized_model_file_path: P,
    ) -> self::Result<&mut Self> {
        #[cfg(target_family = "unix")]
        let optimized_model_file =
            CString::new(optimized_model_file_path.as_ref().as_os_str().as_bytes())?;
        #[cfg(target_family = "windows")]
        let optimized_model_file =
            optimized_model_file_path.as_ref().as_os_str().encode_wide().collect::<Vec<_>>();

        panic_on_error!(ORT_API.SetOptimizedModelFilePath.unwrap()(
            self.raw.as_ptr(),
            optimized_model_file.as_ptr(),
        ));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L899-L906)
    pub fn enable_profiling<P: AsRef<Path>>(
        &mut self,
        profile_file_prefix: P,
    ) -> self::Result<&mut Self> {
        #[cfg(target_family = "unix")]
        let profile_file_prefix =
            CString::new(profile_file_prefix.as_ref().as_os_str().as_bytes())?;
        #[cfg(target_family = "windows")]
        let profile_file_prefix =
            profile_file_prefix.as_ref().as_os_str().encode_wide().collect::<Vec<_>>();

        panic_on_error!(ORT_API.EnableProfiling.unwrap()(
            self.raw.as_ptr(),
            profile_file_prefix.as_ptr(),
        ));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L908-L914)
    pub fn disable_profiling(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.DisableProfiling.unwrap()(self.raw.as_ptr()));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L916-L929)
    pub fn enable_mem_pattern(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.EnableMemPattern.unwrap()(self.raw.as_ptr()));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L931-L939)
    pub fn disable_mem_pattern(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.DisableMemPattern.unwrap()(self.raw.as_ptr()));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L886-L897)
    pub fn set_execution_mode(&mut self, execution_mode: ExecutionMode) -> &mut Self {
        panic_on_error!(ORT_API.SetSessionExecutionMode.unwrap()(
            self.raw.as_ptr(),
            execution_mode,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L959-L966)
    pub fn set_log_id(&mut self, log_id: &str) -> self::Result<&mut Self> {
        let log_id = CString::new(log_id)?;
        panic_on_error!(ORT_API.SetSessionLogId.unwrap()(self.raw.as_ptr(), log_id.as_ptr(),));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L968-L977)
    pub fn set_log_verbosity_level(&mut self, level: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetSessionLogVerbosityLevel.unwrap()(self.raw.as_ptr(), level,));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L979-L986)
    pub fn set_log_severity_level(&mut self, level: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetSessionLogSeverityLevel.unwrap()(self.raw.as_ptr(), level,));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2104-L2113)
    pub fn disable_per_session_threads(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.DisablePerSessionThreads.unwrap()(self.raw.as_ptr()));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2947-L2953)
    pub fn enable_ort_custom_ops(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.EnableOrtCustomOps.unwrap()(self.raw.as_ptr()));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2228-L2241)
    pub fn set_session_config_entry(
        &mut self,
        config_key: &str,
        config_value: &str,
    ) -> self::Result<&mut Self> {
        let config_key = CString::new(config_key)?;
        let config_value = CString::new(config_value)?;
        bail_on_error!(ORT_API.AddSessionConfigEntry.unwrap()(
            self.raw.as_ptr(),
            config_key.as_ptr(),
            config_value.as_ptr(),
        ));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3962-L3991)
    pub fn session_config_entry(&self, config_key: &str) -> self::Result<String> {
        let config_key = CString::new(config_key)?;
        let mut size = 0;
        bail_on_error!(ORT_API.GetSessionConfigEntry.unwrap()(
            self.raw.as_ptr(),
            config_key.as_ptr(),
            ptr::null_mut(),
            &mut size
        ));
        let mut config_value = Vec::<u8>::with_capacity(size);
        bail_on_error!(ORT_API.GetSessionConfigEntry.unwrap()(
            self.raw.as_ptr(),
            config_key.as_ptr(),
            config_value.as_mut_ptr() as *mut c_char,
            &mut size
        ));
        unsafe {
            config_value.set_len(size);
        }
        Ok(CString::from_vec_with_nul(config_value)?.into_string()?)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3944-L3960)
    pub fn has_session_config_entry(&self, config_key: &str) -> self::Result<bool> {
        let config_key = CString::new(config_key)?;
        let mut has_session_config_entry = 0;
        panic_on_error!(ORT_API.HasSessionConfigEntry.unwrap()(
            self.raw.as_ptr(),
            config_key.as_ptr(),
            &mut has_session_config_entry
        ));
        Ok(has_session_config_entry != 0)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1626-L1638)
    pub fn set_free_dimension_by_denotation(
        &mut self,
        dim_denotation: &str,
        dim_value: i64,
    ) -> self::Result<&mut Self> {
        let dim_notation = CString::new(dim_denotation)?;
        panic_on_error!(ORT_API.AddFreeDimensionOverride.unwrap()(
            self.raw.as_ptr(),
            dim_notation.as_ptr(),
            dim_value,
        ));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2150-L2159)
    pub fn set_free_dimension_by_name(
        &mut self,
        dim_name: &str,
        dim_value: i64,
    ) -> self::Result<&mut Self> {
        let dim_notation = CString::new(dim_name)?;
        panic_on_error!(ORT_API.AddFreeDimensionOverrideByName.unwrap()(
            self.raw.as_ptr(),
            dim_notation.as_ptr(),
            dim_value,
        ));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2506-L2521)
    pub fn set_initializer<'s, 'v: 's>(
        &'s mut self,
        name: &str,
        value: &'v Value<'_>,
    ) -> self::Result<&mut Self> {
        let name = CString::new(name)?;
        bail_on_error!(ORT_API.AddInitializer.unwrap()(
            self.raw.as_ptr(),
            name.as_ptr(),
            value.raw,
        ));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3462-L3485)
    pub fn set_external_initializers<'s, 'v: 's, I: AsRef<str>>(
        &'s mut self,
        initializer_names: &[I],
        initializers: &'v [Value<'_>],
    ) -> self::Result<&mut Self> {
        assert_eq!(initializer_names.len(), initializers.len());
        unsafe { self.set_external_initializers_unchecked(initializer_names, initializers) }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3462-L3485)
    ///
    /// # Safety
    ///
    /// The length of `initializer_names` must be equal to that of `initializers`.
    pub unsafe fn set_external_initializers_unchecked<'s, 'v: 's, I: AsRef<str>>(
        &'s mut self,
        initializer_names: &[I],
        initializers: &'v [Value<'_>],
    ) -> self::Result<&mut Self> {
        let initializer_names_c_string = (initializer_names.iter())
            .map(AsRef::as_ref)
            .map(CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        self.set_external_initializers_with_c_str_unchecked(
            &initializer_names_c_string,
            initializers,
        )
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3462-L3485)
    pub fn set_external_initializers_with_c_str<'s, 'v: 's, I: AsRef<CStr>>(
        &'s mut self,
        initializer_names: &[I],
        initializers: &'v [Value<'_>],
    ) -> self::Result<&mut Self> {
        assert_eq!(initializer_names.len(), initializers.len());
        unsafe {
            self.set_external_initializers_with_c_str_unchecked(initializer_names, initializers)
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3462-L3485)
    ///
    /// # Safety
    ///
    /// The length of `initializer_names` must be equal to that of `initializers`.
    pub unsafe fn set_external_initializers_with_c_str_unchecked<'s, 'v: 's, I: AsRef<CStr>>(
        &'s mut self,
        initializer_names: &[I],
        initializers: &'v [Value<'_>],
    ) -> self::Result<&mut Self> {
        let initializer_names_c_char =
            initializer_names.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        self.set_external_initializers_with_c_chars_with_nul(
            &initializer_names_c_char,
            initializers,
        )
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3462-L3485)
    pub fn set_external_initializers_with_bytes_with_nul<'s, 'v: 's>(
        &'s mut self,
        initializer_names: &[&[u8]],
        initializers: &'v [Value<'_>],
    ) -> self::Result<&mut Self> {
        assert_eq!(initializer_names.len(), initializers.len());
        assert!(initializer_names.iter().all(|name| name.ends_with(&[b'\0'])));
        unsafe {
            self.set_external_initializers_with_bytes_with_nul_unchecked(
                initializer_names,
                initializers,
            )
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3462-L3485)
    ///
    /// # Safety
    ///
    /// The length of `initializer_names` must be equal to that of `initializers`. Every slice in
    /// `initializer_names` must be terminated with a null character.
    pub unsafe fn set_external_initializers_with_bytes_with_nul_unchecked<'s, 'v: 's>(
        &'s mut self,
        initializer_names: &[&[u8]],
        initializers: &'v [Value<'_>],
    ) -> self::Result<&mut Self> {
        let initializer_names_c_char =
            initializer_names.iter().map(|x| x.as_ptr() as *const c_char).collect::<Vec<_>>();
        self.set_external_initializers_with_c_chars_with_nul(
            &initializer_names_c_char,
            initializers,
        )
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3462-L3485)
    ///
    /// # Safety
    ///
    /// The length of `initializer_names` must be equal to that of `initializers`. Every pointer in
    /// `initializer_names` must be a null-terminated string.
    pub unsafe fn set_external_initializers_with_c_chars_with_nul<'s, 'v: 's>(
        &'s mut self,
        initializer_names: &[*const c_char],
        initializers: &'v [Value<'_>],
    ) -> self::Result<&mut Self> {
        debug_assert_eq!(initializer_names.len(), initializers.len());
        bail_on_error!(ORT_API.AddExternalInitializers.unwrap()(
            self.raw.as_ptr(),
            initializer_names.as_ptr(),
            initializers.as_ptr() as *const *const OrtValue,
            initializers.len(),
        ));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3283-L3290)
    pub fn set_custom_create_thread_fn(
        &mut self,
        custom_create_thread_fn: OrtCustomCreateThreadFn,
    ) -> &mut Self {
        panic_on_error!(ORT_API.SessionOptionsSetCustomCreateThreadFn.unwrap()(
            self.raw.as_ptr(),
            custom_create_thread_fn,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3292-L3299)
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn set_custom_thread_creation_options(
        &mut self,
        custom_thread_creation_options: *mut c_void,
    ) -> &mut Self {
        panic_on_error!(ORT_API.SessionOptionsSetCustomThreadCreationOptions.unwrap()(
            self.raw.as_ptr(),
            custom_thread_creation_options,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3301-L3308)
    pub fn set_custom_join_thread_fn(
        &mut self,
        custom_join_thread_fn: OrtCustomJoinThreadFn,
    ) -> &mut Self {
        panic_on_error!(ORT_API.SessionOptionsSetCustomJoinThreadFn.unwrap()(
            self.raw.as_ptr(),
            custom_join_thread_fn,
        ));
        self
    }

    /// [`cpu_provider_factory.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h#L10-L15)
    pub fn append_execution_provider_cpu(&mut self, use_arena: bool) -> self::Result<&mut Self> {
        bail_on_error!(onnxrt_sys::OrtSessionOptionsAppendExecutionProvider_CPU(
            self.raw.as_ptr(),
            use_arena as c_int
        ));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3364-L3384)
    pub fn append_execution_provider_cuda(
        &mut self,
        cuda_options: &CudaProviderOptions,
    ) -> self::Result<&mut Self> {
        bail_on_error!(ORT_API.SessionOptionsAppendExecutionProvider_CUDA_V2.unwrap()(
            self.raw.as_ptr(),
            cuda_options.raw.as_ptr()
        ));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3571-L3621)
    pub fn append_execution_provider<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        provider_name: &str,
        provider_options_keys: &[K],
        provider_options_values: &[V],
    ) -> self::Result<&mut Self> {
        assert_eq!(provider_options_keys.len(), provider_options_values.len());
        unsafe {
            self.append_execution_provider_unchecked(
                provider_name,
                provider_options_keys,
                provider_options_values,
            )
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3571-L3621)
    ///
    /// # Safety
    ///
    /// The length of `provider_options_keys` must be equal to that of `provider_options_values`.
    pub unsafe fn append_execution_provider_unchecked<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        provider_name: &str,
        provider_options_keys: &[K],
        provider_options_values: &[V],
    ) -> self::Result<&mut Self> {
        let provider_name_c_string = CString::new(provider_name)?;
        let provider_options_keys_c_string = (provider_options_keys.iter())
            .map(AsRef::as_ref)
            .map(CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        let provider_options_values_c_string = (provider_options_values.iter())
            .map(AsRef::as_ref)
            .map(CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        self.append_execution_provider_with_c_str_unchecked(
            &provider_name_c_string,
            &provider_options_keys_c_string,
            &provider_options_values_c_string,
        )
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3571-L3621)
    pub fn append_execution_provider_with_c_str<K: AsRef<CStr>, V: AsRef<CStr>>(
        &mut self,
        provider_name: &CStr,
        provider_options_keys: &[K],
        provider_options_values: &[V],
    ) -> self::Result<&mut Self> {
        assert_eq!(provider_options_keys.len(), provider_options_values.len());
        unsafe {
            self.append_execution_provider_with_c_str_unchecked(
                provider_name,
                provider_options_keys,
                provider_options_values,
            )
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3571-L3621)
    ///
    /// # Safety
    ///
    /// The length of `provider_options_keys` must be equal to that of `provider_options_values`.
    pub unsafe fn append_execution_provider_with_c_str_unchecked<K: AsRef<CStr>, V: AsRef<CStr>>(
        &mut self,
        provider_name: &CStr,
        provider_options_keys: &[K],
        provider_options_values: &[V],
    ) -> self::Result<&mut Self> {
        let provider_name_c_char = provider_name.as_ptr();
        let provider_options_keys_c_char =
            provider_options_keys.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        let provider_options_values_c_char =
            provider_options_values.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        self.append_execution_provider_with_c_chars_with_nul(
            provider_name_c_char,
            &provider_options_keys_c_char,
            &provider_options_values_c_char,
        )
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3571-L3621)
    pub fn append_execution_provider_with_bytes_with_nul(
        &mut self,
        provider_name: &[u8],
        provider_options_keys: &[&[u8]],
        provider_options_values: &[&[u8]],
    ) -> self::Result<&mut Self> {
        assert_eq!(provider_options_keys.len(), provider_options_values.len());
        assert!(provider_name.ends_with(&[b'\0']));
        assert!(provider_options_keys.iter().all(|x| x.ends_with(&[b'\0'])));
        assert!(provider_options_values.iter().all(|x| x.ends_with(&[b'\0'])));
        unsafe {
            self.append_execution_provider_with_bytes_with_nul_unchecked(
                provider_name,
                provider_options_keys,
                provider_options_values,
            )
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3571-L3621)
    ///
    /// # Safety
    ///
    /// The length of `provider_options_keys` must be equal to that of `provider_options_values`.
    /// `provider_name` and every slice in `provider_options_keys` and `provider_options_values`
    ///  must be a null-terminated string.
    pub unsafe fn append_execution_provider_with_bytes_with_nul_unchecked(
        &mut self,
        provider_name: &[u8],
        provider_options_keys: &[&[u8]],
        provider_options_values: &[&[u8]],
    ) -> self::Result<&mut Self> {
        let provider_name_c_char = provider_name.as_ptr() as *const c_char;
        let provider_options_keys_c_char =
            provider_options_keys.iter().map(|x| x.as_ptr() as *const c_char).collect::<Vec<_>>();
        let provider_options_values_c_char =
            provider_options_values.iter().map(|x| x.as_ptr() as *const c_char).collect::<Vec<_>>();
        self.append_execution_provider_with_c_chars_with_nul(
            provider_name_c_char,
            &provider_options_keys_c_char,
            &provider_options_values_c_char,
        )
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3571-L3621)
    ///
    /// # Safety
    ///
    /// The length of `provider_options_keys` must be equal to that of `provider_options_values`.
    /// `provider_name` and every pointer in `provider_options_keys` and `provider_options_values`
    /// must be a null-terminated string.
    pub unsafe fn append_execution_provider_with_c_chars_with_nul(
        &mut self,
        provider_name: *const c_char,
        provider_options_keys: &[*const c_char],
        provider_options_values: &[*const c_char],
    ) -> self::Result<&mut Self> {
        debug_assert_eq!(provider_options_keys.len(), provider_options_values.len());
        bail_on_error!(ORT_API.SessionOptionsAppendExecutionProvider.unwrap()(
            self.raw.as_ptr(),
            provider_name,
            provider_options_keys.as_ptr(),
            provider_options_values.as_ptr(),
            provider_options_keys.len()
        ));
        Ok(self)
    }
}

impl Clone for SessionOptions {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L876-L884)
    fn clone(&self) -> Self {
        let mut session_options = ptr::null_mut::<OrtSessionOptions>();
        panic_on_error!(ORT_API.CloneSessionOptions.unwrap()(
            self.raw.as_ptr(),
            &mut session_options,
        ));
        Self { raw: NonNull::new(session_options).unwrap() }
    }
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SessionOptions {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1875)
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseSessionOptions.unwrap()(self.raw.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct ModelMetadata {
    raw: NonNull<OrtModelMetadata>,
}

impl ModelMetadata {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2012-L2021)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2023-L2032)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2626-L2638)
    pub fn graph_description(&self, allocator: &mut Allocator) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut description_ptr = ptr::null_mut::<c_char>();
        panic_on_error!(ORT_API.ModelMetadataGetGraphDescription.unwrap()(
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2034-L2043)
    pub fn domain(&self, allocator: &mut Allocator) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut domain_ptr = ptr::null_mut::<c_char>();
        panic_on_error!(ORT_API.ModelMetadataGetDomain.unwrap()(
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2045-L2054)
    pub fn description(&self, allocator: &mut Allocator) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut description_ptr = ptr::null_mut::<c_char>();
        panic_on_error!(ORT_API.ModelMetadataGetDescription.unwrap()(
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2132-L2144)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2056-L2067)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2069-L2076)
    pub fn version(&self) -> i64 {
        let mut version = 0;
        panic_on_error!(ORT_API.ModelMetadataGetVersion.unwrap()(self.raw.as_ptr(), &mut version));
        version
    }
}

impl Drop for ModelMetadata {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2078)
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseModelMetadata.unwrap()(self.raw.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct Session {
    raw: NonNull<OrtSession>,
    #[allow(dead_code)] // Env has to outlive Session.
    env: Arc<Mutex<Env>>,
    #[allow(dead_code)] // PrepackedWeightsContainer has to outlive Session.
    prepacked_weights_container: Option<Arc<Mutex<PrepackedWeightsContainer>>>,
}

impl Session {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L791-L806)
    pub fn new_with_model_path<P: AsRef<Path>>(
        env: Arc<Mutex<Env>>,
        model_path: P,
        options: &SessionOptions,
    ) -> self::Result<Self> {
        #[cfg(target_family = "unix")]
        let model_path = CString::new(model_path.as_ref().as_os_str().as_bytes())?;
        #[cfg(target_family = "windows")]
        let model_path = model_path.as_ref().as_os_str().encode_wide().collect::<Vec<_>>();

        let mut session = ptr::null_mut::<OrtSession>();
        bail_on_error!(ORT_API.CreateSession.unwrap()(
            env.lock().unwrap().raw.as_ptr(),
            model_path.as_ptr(),
            options.raw.as_ptr(),
            &mut session,
        ));
        Ok(Session { raw: NonNull::new(session).unwrap(), env, prepacked_weights_container: None })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2824-L2843)
    pub fn new_with_model_path_and_prepacked_weights_container<P: AsRef<Path>>(
        env: Arc<Mutex<Env>>,
        model_path: P,
        options: &SessionOptions,
        prepacked_weights_container: Arc<Mutex<PrepackedWeightsContainer>>,
    ) -> self::Result<Self> {
        #[cfg(target_family = "unix")]
        let model_path = CString::new(model_path.as_ref().as_os_str().as_bytes())?;
        #[cfg(target_family = "windows")]
        let model_path = model_path.as_ref().as_os_str().encode_wide().collect::<Vec<_>>();

        let mut session = ptr::null_mut::<OrtSession>();
        bail_on_error!(ORT_API.CreateSessionWithPrepackedWeightsContainer.unwrap()(
            env.lock().unwrap().raw.as_ptr(),
            model_path.as_ptr(),
            options.raw.as_ptr(),
            prepacked_weights_container.lock().unwrap().raw.as_ptr(),
            &mut session,
        ));
        Ok(Session {
            raw: NonNull::new(session).unwrap(),
            env,
            prepacked_weights_container: Some(prepacked_weights_container),
        })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L808-L819)
    pub fn new_with_model_data(
        env: Arc<Mutex<Env>>,
        model_data: &[u8],
        options: &SessionOptions,
    ) -> self::Result<Self> {
        let mut session = ptr::null_mut::<OrtSession>();
        bail_on_error!(ORT_API.CreateSessionFromArray.unwrap()(
            env.lock().unwrap().raw.as_ptr(),
            model_data.as_ptr() as *const c_void,
            model_data.len(),
            options.raw.as_ptr(),
            &mut session,
        ));
        Ok(Session { raw: NonNull::new(session).unwrap(), env, prepacked_weights_container: None })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2845-L2866)
    pub fn new_with_model_data_and_prepacked_weights_container(
        env: Arc<Mutex<Env>>,
        model_data: &[u8],
        options: &SessionOptions,
        prepacked_weights_container: Arc<Mutex<PrepackedWeightsContainer>>,
    ) -> self::Result<Self> {
        let mut session = ptr::null_mut::<OrtSession>();
        bail_on_error!(ORT_API.CreateSessionFromArrayWithPrepackedWeightsContainer.unwrap()(
            env.lock().unwrap().raw.as_ptr(),
            model_data.as_ptr() as *const c_void,
            model_data.len(),
            options.raw.as_ptr(),
            prepacked_weights_container.lock().unwrap().raw.as_ptr(),
            &mut session,
        ));
        Ok(Session {
            raw: NonNull::new(session).unwrap(),
            env,
            prepacked_weights_container: Some(prepacked_weights_container),
        })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L821-L843)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L821-L843)
    ///
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L821-L843)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L821-L843)
    ///
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L821-L843)
    pub fn run_with_bytes_with_nul(
        &mut self,
        run_options: Option<&RunOptions>,
        input_names: &[&[u8]],
        input_values: &[Value<'_>],
        output_names: &[&[u8]],
        output_values: &mut [Value<'_>],
    ) -> self::Result<()> {
        assert_eq!(input_names.len(), input_values.len());
        assert!(input_names.iter().all(|name| name.ends_with(&[b'\0'])));
        assert_eq!(output_names.len(), output_values.len());
        assert!(output_names.iter().all(|name| name.ends_with(&[b'\0'])));
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L821-L843)
    ///
    /// # Safety
    ///
    /// The lengths of `input_names` and `output_names` must be those of `input_values` and
    /// `output_values`, respectively. Every slice in `input_names` and `output_names` must be
    /// terminated with a null character.
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L821-L843)
    ///
    /// # Safety
    ///
    /// The lengths of `input_names` and `output_names` must be those of `input_values` and
    /// `output_values`, respectively. Every pointer in `input_names` and `output_names` must be a
    /// null-terminated string.
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
            input_names.len(),
            output_names.as_ptr(),
            output_names.len(),
            output_values.as_mut_ptr() as *mut *mut OrtValue,
        ));
        Ok(())
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2266-L2276)
    pub fn run_with_binding(
        &mut self,
        run_options: &RunOptions,
        binding: &IoBinding<'_>,
    ) -> self::Result<()> {
        bail_on_error!(ORT_API.RunWithBinding.unwrap()(
            self.raw.as_ptr(),
            run_options.raw.as_ptr(),
            binding.raw.as_ptr(),
        ));
        Ok(())
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1089-L1100)
    pub fn input_count(&self) -> self::Result<usize> {
        let mut count = 0;
        bail_on_error!(ORT_API.SessionGetInputCount.unwrap()(self.raw.as_ptr(), &mut count));
        Ok(count)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1102-L1113)
    pub fn output_count(&self) -> self::Result<usize> {
        let mut count = 0;
        bail_on_error!(ORT_API.SessionGetOutputCount.unwrap()(self.raw.as_ptr(), &mut count));
        Ok(count)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1115-L1124)
    pub fn overridable_initializer_count(&self) -> self::Result<usize> {
        let mut count = 0;
        bail_on_error!(ORT_API.SessionGetOverridableInitializerCount.unwrap()(
            self.raw.as_ptr(),
            &mut count,
        ));
        Ok(count)
    }

    pub fn input_name(&self, index: usize) -> self::Result<String> {
        self.input_name_using_allocator(index, &mut ALLOCATOR_WITH_DEFAULT_OPTIONS.lock().unwrap())
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1156-L1165)
    pub fn input_name_using_allocator(
        &self,
        index: usize,
        allocator: &mut Allocator,
    ) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut name_ptr = ptr::null_mut::<c_char>();
        bail_on_error!(ORT_API.SessionGetInputName.unwrap()(
            self.raw.as_ptr(),
            index,
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1167-L1176)
    pub fn output_name_using_allocator(
        &self,
        index: usize,
        allocator: &mut Allocator,
    ) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut name_ptr = ptr::null_mut::<c_char>();
        bail_on_error!(ORT_API.SessionGetOutputName.unwrap()(
            self.raw.as_ptr(),
            index,
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1178-L1188)
    pub fn overridable_initializer_name_using_allocator(
        &self,
        index: usize,
        allocator: &mut Allocator,
    ) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut name_ptr = ptr::null_mut::<c_char>();
        bail_on_error!(ORT_API.SessionGetOverridableInitializerName.unwrap()(
            self.raw.as_ptr(),
            index,
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1987-L1997)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2446-L2455)
    pub fn profiling_start_time_ns(&self) -> u64 {
        let mut time = 0;
        panic_on_error!(ORT_API.SessionGetProfilingStartTimeNs.unwrap()(
            self.raw.as_ptr(),
            &mut time,
        ));
        time
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1999-L2006)
    pub fn model_metadata(&self) -> self::Result<ModelMetadata> {
        let mut model_metadata = ptr::null_mut::<OrtModelMetadata>();
        bail_on_error!(ORT_API.SessionGetModelMetadata.unwrap()(
            self.raw.as_ptr(),
            &mut model_metadata,
        ));
        Ok(ModelMetadata { raw: NonNull::new(model_metadata).unwrap() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1126-L1134)
    pub fn input_type_info(&self, index: usize) -> self::Result<TypeInfo> {
        let mut type_info = ptr::null_mut::<OrtTypeInfo>();
        bail_on_error!(ORT_API.SessionGetInputTypeInfo.unwrap()(
            self.raw.as_ptr(),
            index,
            &mut type_info,
        ));
        Ok(TypeInfo { raw: NonNull::new(type_info).unwrap() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1136-L1144)
    pub fn output_type_info(&self, index: usize) -> self::Result<TypeInfo> {
        let mut type_info = ptr::null_mut::<OrtTypeInfo>();
        bail_on_error!(ORT_API.SessionGetOutputTypeInfo.unwrap()(
            self.raw.as_ptr(),
            index,
            &mut type_info,
        ));
        Ok(TypeInfo { raw: NonNull::new(type_info).unwrap() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1146-L1154)
    pub fn overridable_initializer_type_info(&self, index: usize) -> self::Result<TypeInfo> {
        let mut type_info = ptr::null_mut::<OrtTypeInfo>();
        bail_on_error!(ORT_API.SessionGetOverridableInitializerTypeInfo.unwrap()(
            self.raw.as_ptr(),
            index,
            &mut type_info,
        ));
        Ok(TypeInfo { raw: NonNull::new(type_info).unwrap() })
    }
}

impl Drop for Session {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1855)
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
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1417-L1423)
    pub fn new() -> Self {
        let mut tensor_type_info = ptr::null_mut::<OrtTensorTypeAndShapeInfo>();
        panic_on_error!(ORT_API.CreateTensorTypeAndShapeInfo.unwrap()(&mut tensor_type_info));
        Self { raw: NonNull::new(tensor_type_info).unwrap() }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1425-L1432)
    pub fn set_element_type(&mut self, typ: ONNXTensorElementDataType) -> &mut Self {
        panic_on_error!(ORT_API.SetTensorElementType.unwrap()(self.raw.as_ptr(), typ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1444-L1454)
    pub fn element_type(&self) -> ONNXTensorElementDataType {
        element_type(unsafe { self.raw.as_ref() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1489-L1505)
    pub fn element_count(&self) -> self::Result<i64> {
        element_count(unsafe { self.raw.as_ref() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1434-L1442)
    pub fn set_dimensions(&mut self, dims: &[i64]) -> &mut Self {
        panic_on_error!(ORT_API.SetDimensions.unwrap()(
            self.raw.as_ptr(),
            dims.as_ptr(),
            dims.len(),
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1456-L1465)
    pub fn dimensions_count(&self) -> usize {
        dimensions_count(unsafe { self.raw.as_ref() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1467-L1476)
    pub fn dimensions(&self) -> Vec<i64> {
        dimensions(unsafe { self.raw.as_ref() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1478-L1487)
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
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1871)
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
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1444-L1454)
    pub fn element_type(&self) -> ONNXTensorElementDataType {
        element_type(self.raw)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1489-L1505)
    pub fn element_count(&self) -> self::Result<i64> {
        element_count(self.raw)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1456-L1465)
    pub fn dimensions_count(&self) -> usize {
        dimensions_count(self.raw)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1467-L1476)
    pub fn dimensions(&self) -> Vec<i64> {
        dimensions(self.raw)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1478-L1487)
    pub fn symbolic_dimensions(&self) -> self::Result<Vec<&str>> {
        symbolic_dimensions(self.raw)
    }
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1444-L1454)
fn element_type(tensor_type_info: &OrtTensorTypeAndShapeInfo) -> ONNXTensorElementDataType {
    let mut typ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    panic_on_error!(ORT_API.GetTensorElementType.unwrap()(tensor_type_info, &mut typ));
    typ
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1489-L1505)
fn element_count(tensor_type_info: &OrtTensorTypeAndShapeInfo) -> self::Result<i64> {
    let mut count = 0;
    bail_on_error!(ORT_API.GetTensorShapeElementCount.unwrap()(
        tensor_type_info,
        // https://github.com/microsoft/onnxruntime/issues/3132
        &mut count as *mut i64 as *mut _,
    ));
    Ok(count)
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1456-L1465)
fn dimensions_count(tensor_type_info: &OrtTensorTypeAndShapeInfo) -> usize {
    let mut count = 0;
    panic_on_error!(ORT_API.GetDimensionsCount.unwrap()(tensor_type_info, &mut count));
    count
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1467-L1476)
fn dimensions(tensor_type_info: &OrtTensorTypeAndShapeInfo) -> Vec<i64> {
    let mut dims = vec![0; dimensions_count(tensor_type_info)];
    panic_on_error!(ORT_API.GetDimensions.unwrap()(
        tensor_type_info,
        dims.as_mut_ptr(),
        dims.len(),
    ));
    dims
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1478-L1487)
fn symbolic_dimensions(tensor_type_info: &OrtTensorTypeAndShapeInfo) -> self::Result<Vec<&str>> {
    let mut dimensions = vec![ptr::null::<c_char>(); dimensions_count(tensor_type_info)];
    panic_on_error!(ORT_API.GetSymbolicDimensions.unwrap()(
        tensor_type_info,
        dimensions.as_mut_ptr(),
        dimensions.len(),
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
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1393-L1402)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1404-L1411)
    pub fn onnx_type(&self) -> ONNXType {
        let mut typ = ONNX_TYPE_UNKNOWN;
        panic_on_error!(ORT_API.GetOnnxTypeFromTypeInfo.unwrap()(self.raw.as_ptr(), &mut typ));
        typ
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1885-L1898)
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
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1867)
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseTypeInfo.unwrap()(self.raw.as_ptr());
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
pub trait AsONNXTensorElementDataType {
    fn as_onnx_tensor_element_data_type() -> ONNXTensorElementDataType;
}

#[macro_export]
macro_rules! impl_AsONNXTensorElementDataType {
    ($typ:ty, $onnx_tensor_element_data_type:expr$(,)?) => {
        impl $crate::AsONNXTensorElementDataType for $typ {
            fn as_onnx_tensor_element_data_type() -> onnxrt_sys::ONNXTensorElementDataType {
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1291-L1304)
    pub fn new_tensor_using_allocator(
        allocator: &mut Allocator,
        shape: &[i64],
        element_type: ONNXTensorElementDataType,
    ) -> self::Result<Self> {
        let mut value = ptr::null_mut::<OrtValue>();
        bail_on_error!(ORT_API.CreateTensorAsOrtValue.unwrap()(
            allocator.raw.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            element_type,
            &mut value,
        ));
        Ok(Value { raw: value, phantom: PhantomData })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1306-L1323)
    pub fn new_tensor_with_data<T: AsONNXTensorElementDataType>(
        memory_info: &MemoryInfo,
        data: &'d mut [T],
        shape: &[i64],
    ) -> self::Result<Self> {
        let mut value = ptr::null_mut::<OrtValue>();
        bail_on_error!(ORT_API.CreateTensorWithDataAsOrtValue.unwrap()(
            memory_info.raw.as_ptr(),
            data.as_mut_ptr() as *mut c_void,
            mem::size_of_val(data),
            shape.as_ptr(),
            shape.len(),
            T::as_onnx_tensor_element_data_type(),
            &mut value,
        ));
        Ok(Value { raw: value, phantom: PhantomData })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1325-L1332)
    pub fn is_tensor(&self) -> bool {
        let mut result = 0;
        panic_on_error!(ORT_API.IsTensor.unwrap()(self.raw, &mut result));
        result != 0
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1334-L1344)
    ///
    /// # Safety
    ///
    /// `T` must be the same element type as specified when `self` was created.
    pub unsafe fn tensor_data<T>(&self) -> self::Result<&[T]> {
        let mut data = ptr::null_mut::<T>();
        bail_on_error!(ORT_API.GetTensorMutableData.unwrap()(
            self.raw,
            &mut data as *mut *mut T as *mut *mut c_void,
        ));
        // https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html#safety
        //
        // > pub const unsafe fn from_raw_parts<'a, T>(data: *const T, len: usize) -> &'a [T]
        // > ...
        // > `data` must be non-null and aligned even for zero-length slices. One reason for this
        // > is that enum layout optimizations may rely on references (including slices of any
        // > length) being aligned and non-null to distinguish them from other data. You can obtain
        // > a pointer that is usable as `data` for zero-length slices using `NonNull::dangling()`.
        if data.is_null() {
            data = NonNull::dangling().as_ptr();
        }

        let mut element_count = self.tensor_type_info()?.element_count()?;
        // https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1492
        //
        // > If any dimension is less than 0, the result is always -1.
        if element_count < 0 {
            element_count = 0;
        }

        Ok(unsafe { slice::from_raw_parts(data, element_count.try_into()?) })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1334-L1344)
    ///
    /// # Safety
    ///
    /// `T` must be the same element type as specified when `self` was created.
    pub unsafe fn tensor_data_mut<T>(&mut self) -> self::Result<&mut [T]> {
        let mut data = ptr::null_mut::<T>();
        bail_on_error!(ORT_API.GetTensorMutableData.unwrap()(
            self.raw,
            &mut data as *mut *mut T as *mut *mut c_void,
        ));
        // https://doc.rust-lang.org/std/slice/fn.from_raw_parts_mut.html#safety
        //
        // > pub unsafe fn from_raw_parts_mut<'a, T>(data: *mut T, len: usize) -> &'a mut [T]
        // > ...
        // > `data` must be non-null and aligned even for zero-length slices. One reason for this
        // > is that enum layout optimizations may rely on references (including slices of any
        // > length) being aligned and non-null to distinguish them from other data. You can obtain
        // > a pointer that is usable as `data` for zero-length slices using `NonNull::dangling()`.
        if data.is_null() {
            data = NonNull::dangling().as_ptr();
        }

        let mut element_count = self.tensor_type_info()?.element_count()?;
        // https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1492
        //
        // > If any dimension is less than 0, the result is always -1.
        if element_count < 0 {
            element_count = 0;
        }

        Ok(unsafe { slice::from_raw_parts_mut(data, element_count.try_into()?) })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2392-L2406)
    ///
    /// # Safety
    ///
    /// `T` must be the same element type as specified when `self` was created.
    pub unsafe fn get<T>(&self, indices: &[i64]) -> self::Result<&T> {
        let mut element = ptr::null_mut::<T>();
        bail_on_error!(ORT_API.TensorAt.unwrap()(
            self.raw,
            indices.as_ptr(),
            indices.len(),
            &mut element as *mut *mut T as *mut *mut c_void,
        ));
        Ok(&*element)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2392-L2406)
    ///
    /// # Safety
    ///
    /// `T` must be the same element type as specified when `self` was created.
    pub unsafe fn get_mut<T>(&mut self, indices: &[i64]) -> self::Result<&mut T> {
        let mut element = ptr::null_mut::<T>();
        bail_on_error!(ORT_API.TensorAt.unwrap()(
            self.raw,
            indices.as_ptr(),
            indices.len(),
            &mut element as *mut *mut T as *mut *mut c_void,
        ));
        Ok(&mut *element)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1520-L1527)
    pub fn type_info(&self) -> self::Result<Option<TypeInfo>> {
        let mut type_info = ptr::null_mut::<OrtTypeInfo>();
        bail_on_error!(ORT_API.GetTypeInfo.unwrap()(self.raw, &mut type_info));
        // https://github.com/microsoft/onnxruntime/blob/v1.16.0/onnxruntime/core/framework/tensor_type_and_shape.cc#L347-L353
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1511-L1518)
    pub fn tensor_type_info(&self) -> self::Result<TensorTypeAndShapeInfo> {
        let mut tensor_type_info = ptr::null_mut::<OrtTensorTypeAndShapeInfo>();
        bail_on_error!(ORT_API.GetTensorTypeAndShape.unwrap()(self.raw, &mut tensor_type_info));
        Ok(TensorTypeAndShapeInfo { raw: NonNull::new(tensor_type_info).unwrap() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3256-L3262)
    pub fn tensor_memory_info(&self) -> UnownedMemoryInfo<'_> {
        let mut memory_info = ptr::null::<OrtMemoryInfo>();
        panic_on_error!(ORT_API.GetTensorMemoryInfo.unwrap()(self.raw, &mut memory_info));
        UnownedMemoryInfo { raw: unsafe { &*memory_info } }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1529-L1536)
    pub fn value_type(&self) -> self::Result<ONNXType> {
        let mut value_type = ONNX_TYPE_UNKNOWN;
        bail_on_error!(ORT_API.GetValueType.unwrap()(self.raw, &mut value_type));
        Ok(value_type)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3221-L3233)
    pub fn has_value(&self) -> bool {
        let mut has_value = 0;
        panic_on_error!(ORT_API.HasValue.unwrap()(self.raw, &mut has_value));
        has_value != 0
    }
}

impl<'d> Default for Value<'d> {
    fn default() -> Self {
        Value { raw: ptr::null_mut::<OrtValue>(), phantom: PhantomData }
    }
}

impl<'d> Drop for Value<'d> {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1859)
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseValue.unwrap()(self.raw);
        }
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct MemoryInfo {
    raw: NonNull<OrtMemoryInfo>,
}

impl MemoryInfo {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1542-L1553)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1555-L1566)
    pub fn new_for_cpu(allocator_type: OrtAllocatorType, memory_type: OrtMemType) -> Self {
        let mut memory_info = ptr::null_mut::<OrtMemoryInfo>();
        panic_on_error!(ORT_API.CreateCpuMemoryInfo.unwrap()(
            allocator_type,
            memory_type,
            &mut memory_info,
        ));
        MemoryInfo { raw: NonNull::new(memory_info).unwrap() }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1580-L1587)
    pub fn allocator_name(&self) -> self::Result<&str> {
        memory_info_allocator_name(unsafe { self.raw.as_ref() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1597-L1599)
    pub fn allocator_type(&self) -> OrtAllocatorType {
        memory_info_allocator_type(unsafe { self.raw.as_ref() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1589-L1591)
    pub fn device_id(&self) -> i32 {
        memory_info_device_id(unsafe { self.raw.as_ref() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3726-L3730)
    pub fn device_type(&self) -> OrtMemoryInfoDeviceType {
        memory_info_device_type(unsafe { self.raw.as_ref() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1593-L1595)
    pub fn memory_type(&self) -> OrtMemType {
        memory_info_memory_type(unsafe { self.raw.as_ref() })
    }
}

impl Drop for MemoryInfo {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1851)
    fn drop(&mut self) {
        unsafe {
            ORT_API.ReleaseMemoryInfo.unwrap()(self.raw.as_ptr());
        }
    }
}

impl PartialEq<MemoryInfo> for MemoryInfo {
    fn eq(&self, other: &MemoryInfo) -> bool {
        memory_info_is_equal(unsafe { self.raw.as_ref() }, unsafe { other.raw.as_ref() })
    }
}

impl PartialEq<UnownedMemoryInfo<'_>> for MemoryInfo {
    fn eq(&self, other: &UnownedMemoryInfo<'_>) -> bool {
        memory_info_is_equal(unsafe { self.raw.as_ref() }, other.raw)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UnownedMemoryInfo<'a> {
    raw: &'a OrtMemoryInfo,
}

impl<'a> UnownedMemoryInfo<'a> {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1580-L1587)
    pub fn allocator_name(&self) -> self::Result<&str> {
        memory_info_allocator_name(self.raw)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1597-L1599)
    pub fn allocator_type(&self) -> OrtAllocatorType {
        memory_info_allocator_type(self.raw)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1589-L1591)
    pub fn device_id(&self) -> i32 {
        memory_info_device_id(self.raw)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3726-L3730)
    pub fn device_type(&self) -> OrtMemoryInfoDeviceType {
        memory_info_device_type(self.raw)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1593-L1595)
    pub fn memory_type(&self) -> OrtMemType {
        memory_info_memory_type(self.raw)
    }
}

impl PartialEq<UnownedMemoryInfo<'_>> for UnownedMemoryInfo<'_> {
    fn eq(&self, other: &UnownedMemoryInfo<'_>) -> bool {
        memory_info_is_equal(self.raw, other.raw)
    }
}

impl PartialEq<MemoryInfo> for UnownedMemoryInfo<'_> {
    fn eq(&self, other: &MemoryInfo) -> bool {
        memory_info_is_equal(self.raw, unsafe { other.raw.as_ref() })
    }
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1580-L1587)
fn memory_info_allocator_name(memory_info: &OrtMemoryInfo) -> self::Result<&str> {
    let mut name = ptr::null::<c_char>();
    panic_on_error!(ORT_API.MemoryInfoGetName.unwrap()(memory_info, &mut name));
    Ok(unsafe { CStr::from_ptr(name) }.to_str()?)
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1597-L1599)
fn memory_info_allocator_type(memory_info: &OrtMemoryInfo) -> OrtAllocatorType {
    let mut allocator_type = OrtAllocatorType::OrtInvalidAllocator;
    panic_on_error!(ORT_API.MemoryInfoGetType.unwrap()(memory_info, &mut allocator_type));
    allocator_type
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1589-L1591)
fn memory_info_device_id(memory_info: &OrtMemoryInfo) -> i32 {
    let mut device_id = 0;
    panic_on_error!(ORT_API.MemoryInfoGetId.unwrap()(memory_info, &mut device_id));
    device_id
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3726-L3730)
fn memory_info_device_type(memory_info: &OrtMemoryInfo) -> OrtMemoryInfoDeviceType {
    let mut device_type = OrtMemoryInfoDeviceType_CPU;
    unsafe { ORT_API.MemoryInfoGetDeviceType.unwrap()(memory_info, &mut device_type) };
    device_type
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1593-L1595)
fn memory_info_memory_type(memory_info: &OrtMemoryInfo) -> OrtMemType {
    let mut memory_type = OrtMemTypeDefault;
    panic_on_error!(ORT_API.MemoryInfoGetMemType.unwrap()(memory_info, &mut memory_type));
    memory_type
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1568-L1578)
fn memory_info_is_equal(lhs: &OrtMemoryInfo, rhs: &OrtMemoryInfo) -> bool {
    let mut is_equal = 0;
    panic_on_error!(ORT_API.CompareMemoryInfo.unwrap()(lhs, rhs, &mut is_equal));
    is_equal == 0
}

#[derive(Debug)]
pub struct Allocator {
    raw: NonNull<OrtAllocator>,
}

impl Allocator {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2247-L2256)
    pub fn new(session: &Session, memory_info: &MemoryInfo) -> self::Result<Self> {
        let mut allocator = ptr::null_mut::<OrtAllocator>();
        bail_on_error!(ORT_API.CreateAllocator.unwrap()(
            session.raw.as_ptr(),
            memory_info.raw.as_ptr(),
            &mut allocator,
        ));
        Ok(Self { raw: NonNull::new(allocator).unwrap() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1605-L1606)
    pub fn alloc<T>(&mut self) -> *mut T {
        let mut ptr = ptr::null_mut::<T>();
        panic_on_error!(ORT_API.AllocatorAlloc.unwrap()(
            self.raw.as_ptr(),
            mem::size_of::<T>(),
            &mut ptr as *mut *mut T as *mut *mut c_void,
        ));
        ptr
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1607-L1608)
    ///
    /// # Safety
    ///
    /// The memory pointed by `ptr` must have been allocated by `Allocator::alloc`.
    pub unsafe fn free<T>(&mut self, ptr: *mut T) {
        panic_on_error!(ORT_API.AllocatorFree.unwrap()(self.raw.as_ptr(), ptr as *mut c_void));
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L1609-L1610)
    pub fn memory_info(&self) -> UnownedMemoryInfo<'_> {
        let mut memory_info = ptr::null::<OrtMemoryInfo>();
        panic_on_error!(ORT_API.AllocatorGetInfo.unwrap()(self.raw.as_ptr(), &mut memory_info));
        UnownedMemoryInfo { raw: unsafe { &*memory_info } }
    }
}

impl Drop for Allocator {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2258-L2260)
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
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2278-L2289)
    pub fn new(session: &'s mut Session) -> self::Result<Self> {
        let mut io_binding = ptr::null_mut::<OrtIoBinding>();
        bail_on_error!(ORT_API.CreateIoBinding.unwrap()(session.raw.as_ptr(), &mut io_binding));
        Ok(Self { raw: NonNull::new(io_binding).unwrap(), phantom: PhantomData })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2299-L2309)
    pub fn bind_input(&mut self, name: &str, value: &Value<'_>) -> self::Result<&mut Self> {
        let name = CString::new(name)?;
        bail_on_error!(ORT_API.BindInput.unwrap()(self.raw.as_ptr(), name.as_ptr(), value.raw));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2311-L2321)
    pub fn bind_output(&mut self, name: &str, value: &Value<'_>) -> self::Result<&mut Self> {
        let name = CString::new(name)?;
        panic_on_error!(ORT_API.BindOutput.unwrap()(self.raw.as_ptr(), name.as_ptr(), value.raw));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2323-L2338)
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2340-L2358)
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
                let mut names = Vec::with_capacity(count);
                let mut ptr = buffer;
                for &length in slice::from_raw_parts(lengths, count) {
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2360-L2378)
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
                let values_slice = slice::from_raw_parts(values_ptr, count);
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

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2380-L2382)
    pub fn clear_bound_inputs(&mut self) {
        unsafe {
            ORT_API.ClearBoundInputs.unwrap()(self.raw.as_ptr());
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2384-L2386)
    pub fn clear_bound_outputs(&mut self) {
        unsafe {
            ORT_API.ClearBoundOutputs.unwrap()(self.raw.as_ptr());
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3341-L3349)
    pub fn synchronize_bound_inputs(&mut self) -> self::Result<()> {
        bail_on_error!(ORT_API.SynchronizeBoundInputs.unwrap()(self.raw.as_ptr()));
        Ok(())
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3351-L3359)
    pub fn synchronize_bound_outputs(&mut self) -> self::Result<()> {
        bail_on_error!(ORT_API.SynchronizeBoundOutputs.unwrap()(self.raw.as_ptr()));
        Ok(())
    }
}

impl<'s> Drop for IoBinding<'s> {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2295-L2297)
    fn drop(&mut self) {
        unsafe { ORT_API.ReleaseIoBinding.unwrap()(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct ThreadingOptions {
    raw: NonNull<OrtThreadingOptions>,
}

impl ThreadingOptions {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2119-L2124)
    pub fn new() -> Self {
        let mut threading_options = ptr::null_mut::<OrtThreadingOptions>();
        panic_on_error!(ORT_API.CreateThreadingOptions.unwrap()(&mut threading_options));
        Self { raw: NonNull::new(threading_options).unwrap() }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2461-L2472)
    pub fn set_global_intra_op_num_threads(&mut self, intra_op_num_threads: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetGlobalIntraOpNumThreads.unwrap()(
            self.raw.as_ptr(),
            intra_op_num_threads,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2474-L2485)
    pub fn set_global_inter_op_num_threads(&mut self, inter_op_num_threads: i32) -> &mut Self {
        panic_on_error!(ORT_API.SetGlobalInterOpNumThreads.unwrap()(
            self.raw.as_ptr(),
            inter_op_num_threads,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3741-L3760)
    pub fn set_global_intra_op_thread_affinity(
        &mut self,
        affinity: &str,
    ) -> self::Result<&mut Self> {
        let affinity = CString::new(affinity)?;
        bail_on_error!(ORT_API.SetGlobalIntraOpThreadAffinity.unwrap()(
            self.raw.as_ptr(),
            affinity.as_ptr()
        ));
        Ok(self)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2487-L2500)
    pub fn set_global_spin_control(&mut self, allow_spinning: bool) -> &mut Self {
        panic_on_error!(ORT_API.SetGlobalSpinControl.unwrap()(
            self.raw.as_ptr(),
            allow_spinning as _,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2589-L2599)
    pub fn set_global_denormal_as_zero(&mut self) -> &mut Self {
        panic_on_error!(ORT_API.SetGlobalDenormalAsZero.unwrap()(self.raw.as_ptr()));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3313-L3320)
    pub fn set_global_custom_create_thread_fn(
        &mut self,
        custom_create_thread_fn: OrtCustomCreateThreadFn,
    ) -> &mut Self {
        panic_on_error!(ORT_API.SetGlobalCustomCreateThreadFn.unwrap()(
            self.raw.as_ptr(),
            custom_create_thread_fn,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3322-L3329)
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn set_global_custom_thread_creation_options(
        &mut self,
        custom_thread_creation_options: *mut c_void,
    ) -> &mut Self {
        panic_on_error!(ORT_API.SetGlobalCustomThreadCreationOptions.unwrap()(
            self.raw.as_ptr(),
            custom_thread_creation_options,
        ));
        self
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3331-L3338)
    pub fn set_global_custom_join_thread_fn(
        &mut self,
        custom_join_thread_fn: OrtCustomJoinThreadFn,
    ) -> &mut Self {
        panic_on_error!(ORT_API.SetGlobalCustomJoinThreadFn.unwrap()(
            self.raw.as_ptr(),
            custom_join_thread_fn
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
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2126)
    fn drop(&mut self) {
        unsafe { ORT_API.ReleaseThreadingOptions.unwrap()(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct ArenaCfg {
    raw: NonNull<OrtArenaCfg>,
}

impl ArenaCfg {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2742-L2775)
    pub fn new<K: AsRef<str>>(
        arena_config_keys: &[K],
        arena_config_values: &[usize],
    ) -> self::Result<Self> {
        assert_eq!(arena_config_keys.len(), arena_config_values.len());
        unsafe { Self::new_unchecked(arena_config_keys, arena_config_values) }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2742-L2775)
    ///
    /// # Safety
    ///
    /// The length of `arena_config_keys` must be equal to that of `arena_config_values`.
    pub unsafe fn new_unchecked<K: AsRef<str>>(
        arena_config_keys: &[K],
        arena_config_values: &[usize],
    ) -> self::Result<Self> {
        debug_assert_eq!(arena_config_keys.len(), arena_config_values.len());
        let arena_config_keys_c_string = (arena_config_keys.iter())
            .map(AsRef::as_ref)
            .map(CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        Self::new_with_c_str_unchecked(&arena_config_keys_c_string, arena_config_values)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2742-L2775)
    pub fn new_with_c_str<K: AsRef<CStr>>(
        arena_config_keys: &[K],
        arena_config_values: &[usize],
    ) -> self::Result<Self> {
        assert_eq!(arena_config_keys.len(), arena_config_values.len());
        unsafe { Self::new_with_c_str_unchecked(arena_config_keys, arena_config_values) }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2742-L2775)
    ///
    /// # Safety
    ///
    /// The length of `arena_config_keys` must be equal to that of `arena_config_values`.
    pub unsafe fn new_with_c_str_unchecked<K: AsRef<CStr>>(
        arena_config_keys: &[K],
        arena_config_values: &[usize],
    ) -> self::Result<Self> {
        debug_assert_eq!(arena_config_keys.len(), arena_config_values.len());
        let arena_config_keys_c_char =
            arena_config_keys.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        Self::new_with_c_chars_with_nul(&arena_config_keys_c_char, arena_config_values)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2742-L2775)
    ///
    /// # Safety
    ///
    /// The length of `arena_config_keys` must be equal to that of `arena_config_values`.
    pub unsafe fn new_with_c_chars_with_nul(
        arena_config_keys: &[*const c_char],
        arena_config_values: &[usize],
    ) -> self::Result<Self> {
        debug_assert_eq!(arena_config_keys.len(), arena_config_values.len());
        let mut arena_cfg = ptr::null_mut::<OrtArenaCfg>();
        bail_on_error!(ORT_API.CreateArenaCfgV2.unwrap()(
            arena_config_keys.as_ptr(),
            arena_config_values.as_ptr(),
            arena_config_keys.len(),
            &mut arena_cfg
        ));
        Ok(Self { raw: NonNull::new(arena_cfg).unwrap() })
    }
}

impl Drop for ArenaCfg {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2620)
    fn drop(&mut self) {
        unsafe { ORT_API.ReleaseArenaCfg.unwrap()(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct PrepackedWeightsContainer {
    raw: NonNull<OrtPrepackedWeightsContainer>,
}

impl PrepackedWeightsContainer {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2800-L2812)
    pub fn new() -> Self {
        let mut prepacked_weights_container = ptr::null_mut();
        panic_on_error!(ORT_API.CreatePrepackedWeightsContainer.unwrap()(
            &mut prepacked_weights_container
        ));
        Self { raw: NonNull::new(prepacked_weights_container).unwrap() }
    }
}

impl Default for PrepackedWeightsContainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PrepackedWeightsContainer {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2814-L2818)
    fn drop(&mut self) {
        unsafe { ORT_API.ReleasePrepackedWeightsContainer.unwrap()(self.raw.as_ptr()) }
    }
}

unsafe impl Send for PrepackedWeightsContainer {}

unsafe impl Sync for PrepackedWeightsContainer {}

#[derive(Debug)]
pub struct CudaProviderOptions {
    raw: NonNull<OrtCUDAProviderOptionsV2>,
}

impl CudaProviderOptions {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3390-L3398)
    pub fn new() -> self::Result<Self> {
        let mut options = ptr::null_mut::<OrtCUDAProviderOptionsV2>();
        bail_on_error!(ORT_API.CreateCUDAProviderOptions.unwrap()(&mut options));
        Ok(Self { raw: NonNull::new(options).unwrap() })
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3422-L3436)
    pub fn to_string(&self, allocator: &mut Allocator) -> self::Result<String> {
        let allocator = allocator.raw.as_ptr();
        let mut options_ptr = ptr::null_mut::<c_char>();
        bail_on_error!(ORT_API.GetCUDAProviderOptionsAsString.unwrap()(
            self.raw.as_ptr(),
            allocator,
            &mut options_ptr
        ));
        unsafe {
            let options = CStr::from_ptr(options_ptr).to_str()?.to_string();
            (*allocator).Free.unwrap()(allocator, options_ptr as *mut c_void);
            Ok(options)
        }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3400-L3420)
    pub fn set<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        keys: &[K],
        values: &[V],
    ) -> self::Result<()> {
        assert_eq!(keys.len(), values.len());
        unsafe { self.set_unchecked(keys, values) }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3400-L3420)
    ///
    /// # Safety
    ///
    /// The length of `keys` must be equal to that of `values`.
    pub unsafe fn set_unchecked<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        keys: &[K],
        values: &[V],
    ) -> self::Result<()> {
        let keys_c_string =
            keys.iter().map(AsRef::as_ref).map(CString::new).collect::<Result<Vec<_>, _>>()?;
        let values_c_string =
            values.iter().map(AsRef::as_ref).map(CString::new).collect::<Result<Vec<_>, _>>()?;
        self.set_with_c_str_unchecked(&keys_c_string, &values_c_string)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3400-L3420)
    pub fn set_with_c_str<K: AsRef<CStr>, V: AsRef<CStr>>(
        &mut self,
        keys: &[K],
        values: &[V],
    ) -> self::Result<()> {
        assert_eq!(keys.len(), values.len());
        unsafe { self.set_with_c_str_unchecked(keys, values) }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3400-L3420)
    ///
    /// # Safety
    ///
    /// The length of `keys` must be equal to that of `values`.
    pub unsafe fn set_with_c_str_unchecked<K: AsRef<CStr>, V: AsRef<CStr>>(
        &mut self,
        keys: &[K],
        values: &[V],
    ) -> self::Result<()> {
        let keys_c_char = keys.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        let values_c_char = values.iter().map(|x| x.as_ref().as_ptr()).collect::<Vec<_>>();
        self.set_with_c_chars_with_nul(&keys_c_char, &values_c_char)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3400-L3420)
    pub fn set_with_bytes_with_nul(
        &mut self,
        keys: &[&[u8]],
        values: &[&[u8]],
    ) -> self::Result<()> {
        assert_eq!(keys.len(), values.len());
        assert!(keys.iter().all(|x| x.ends_with(&[b'\0'])));
        assert!(values.iter().all(|x| x.ends_with(&[b'\0'])));
        unsafe { self.set_with_bytes_with_nul_unchecked(keys, values) }
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3400-L3420)
    ///
    /// # Safety
    ///
    /// The length of `keys` must be equal to that of `values`. Every slice in `keys` and `values`
    /// must be a null-terminated string.
    pub unsafe fn set_with_bytes_with_nul_unchecked(
        &mut self,
        keys: &[&[u8]],
        values: &[&[u8]],
    ) -> self::Result<()> {
        let keys_c_char = keys.iter().map(|x| x.as_ptr() as *const c_char).collect::<Vec<_>>();
        let values_c_char = values.iter().map(|x| x.as_ptr() as *const c_char).collect::<Vec<_>>();
        self.set_with_c_chars_with_nul(&keys_c_char, &values_c_char)
    }

    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3400-L3420)
    ///
    /// # Safety
    ///
    /// The length of `keys` must be equal to that of `values`. Every slice in `keys` and `values`
    /// must be a null-terminated string.
    pub unsafe fn set_with_c_chars_with_nul(
        &mut self,
        keys: &[*const c_char],
        values: &[*const c_char],
    ) -> self::Result<()> {
        debug_assert_eq!(keys.len(), values.len());
        bail_on_error!(ORT_API.UpdateCUDAProviderOptions.unwrap()(
            self.raw.as_ptr(),
            keys.as_ptr(),
            values.as_ptr(),
            keys.len()
        ));
        Ok(())
    }
}

impl Drop for CudaProviderOptions {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3438-L3444)
    fn drop(&mut self) {
        unsafe { ORT_API.ReleaseCUDAProviderOptions.unwrap()(self.raw.as_ptr()) }
    }
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L657-L661)
pub fn version_string() -> &'static str {
    unsafe {
        CStr::from_ptr(OrtGetApiBase().as_ref().unwrap().GetVersionString.unwrap()())
            .to_str()
            .unwrap()
    }
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L4244-L4250)
pub fn build_info_string() -> &'static str {
    unsafe { CStr::from_ptr(ORT_API.GetBuildInfoString.unwrap()()).to_str().unwrap() }
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2165-L2187)
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

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L3267-L3277)
pub fn execution_provider_api(provider_name: &str) -> self::Result<*const c_void> {
    let mut provider_api = ptr::null::<c_void>();
    let provider_name_c_string = CString::new(provider_name)?;
    bail_on_error!(ORT_API.GetExecutionProviderApi.unwrap()(
        provider_name_c_string.as_ptr(),
        ORT_API_VERSION,
        &mut provider_api
    ));
    Ok(provider_api)
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2660-L2670)
pub fn set_current_gpu_device_id(device_id: i32) -> self::Result<()> {
    bail_on_error!(ORT_API.SetCurrentGpuDeviceId.unwrap()(device_id));
    Ok(())
}

/// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.16.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L2672-L2682)
pub fn current_gpu_device_id() -> self::Result<i32> {
    let mut device_id = 0;
    bail_on_error!(ORT_API.GetCurrentGpuDeviceId.unwrap()(&mut device_id));
    Ok(device_id)
}

// Not implemented:
//
// ORT_API2_STATUS(AddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);
// ORT_API2_STATUS(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info, _Outptr_result_maybenull_ const OrtMapTypeInfo** out);
// ORT_API2_STATUS(CastTypeInfoToOptionalTypeInfo, _In_ const OrtTypeInfo* type_info, _Outptr_result_maybenull_ const OrtOptionalTypeInfo** out);
// ORT_API2_STATUS(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info, _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out);
// ORT_API2_STATUS(CopyKernelInfo, _In_ const OrtKernelInfo* info, _Outptr_ OrtKernelInfo** info_copy);
// ORT_API2_STATUS(CreateCANNProviderOptions, _Outptr_ OrtCANNProviderOptions** out);
// ORT_API2_STATUS(CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);
// ORT_API2_STATUS(CreateDnnlProviderOptions, _Outptr_ OrtDnnlProviderOptions** out);
// ORT_API2_STATUS(CreateOp, _In_ const OrtKernelInfo* info, _In_ const char* op_name, _In_ const char* domain, _In_ int version, _In_opt_ const char** type_constraint_names, _In_opt_ const ONNXTensorElementDataType* type_constraint_values, _In_opt_ int type_constraint_count, _In_opt_ const OrtOpAttr* const* attr_values, _In_opt_ int attr_count, _In_ int input_count, _In_ int output_count, _Outptr_ OrtOp** ort_op);
// ORT_API2_STATUS(CreateOpAttr, _In_ const char* name, _In_ const void* data, _In_ int len, _In_ OrtOpAttrType type, _Outptr_ OrtOpAttr** op_attr);
// ORT_API2_STATUS(CreateOpaqueValue, _In_z_ const char* domain_name, _In_z_ const char* type_name, _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(CreateROCMProviderOptions, _Outptr_ OrtROCMProviderOptions** out);
// ORT_API2_STATUS(CreateSparseTensorAsOrtValue, _Inout_ OrtAllocator* allocator, _In_ const int64_t* dense_shape, size_t dense_shape_len, ONNXTensorElementDataType type, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(CreateSparseTensorWithValuesAsOrtValue, _In_ const OrtMemoryInfo* info, _Inout_ void* p_data, _In_ const int64_t* dense_shape, size_t dense_shape_len, _In_ const int64_t* values_shape, size_t values_shape_len, ONNXTensorElementDataType type, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(CreateTensorRTProviderOptions, _Outptr_ OrtTensorRTProviderOptionsV2** out);
// ORT_API2_STATUS(CreateValue, _In_reads_(num_values) const OrtValue* const* in, size_t num_values, enum ONNXType value_type, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ const OrtCustomOp* op);
// ORT_API2_STATUS(FillSparseTensorBlockSparse, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info, _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values, _In_ const int64_t* indices_shape_data, size_t indices_shape_len, _In_ const int32_t* indices_data);
// ORT_API2_STATUS(FillSparseTensorCoo, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info, _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values, _In_ const int64_t* indices_data, size_t indices_num);
// ORT_API2_STATUS(FillSparseTensorCsr, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info, _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values, _In_ const int64_t* inner_indices_data, size_t inner_indices_num, _In_ const int64_t* outer_indices_data, size_t outer_indices_num);
// ORT_API2_STATUS(FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len);
// ORT_API2_STATUS(FillStringTensorElement, _Inout_ OrtValue* value, _In_ const char* s, size_t index);
// ORT_API2_STATUS(GetCANNProviderOptionsAsString, _In_ const OrtCANNProviderOptions* cann_options, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);
// ORT_API2_STATUS(GetCUDAProviderOptionsByName, _In_ const OrtCUDAProviderOptionsV2* cuda_options, _In_ const char* key, _Outptr_ void** ptr);
// ORT_API2_STATUS(GetDnnlProviderOptionsAsString, _In_ const OrtDnnlProviderOptions* dnnl_options, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);
// ORT_API2_STATUS(GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);
// ORT_API2_STATUS(GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);
// ORT_API2_STATUS(GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name, _In_ const OrtValue* in, _Out_ void* data_container, size_t data_container_size);
// ORT_API2_STATUS(GetOptionalContainedTypeInfo, _In_ const OrtOptionalTypeInfo* optional_type_info, _Outptr_ OrtTypeInfo** out);
// ORT_API2_STATUS(GetROCMProviderOptionsAsString, _In_ const OrtROCMProviderOptions* rocm_options, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);
// ORT_API2_STATUS(GetResizedStringTensorElementBuffer, _Inout_ OrtValue* value, _In_ size_t index, _In_ size_t length_in_bytes, _Inout_ char** buffer);
// ORT_API2_STATUS(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info, _Outptr_ OrtTypeInfo** type_info);
// ORT_API2_STATUS(GetSparseTensorFormat, _In_ const OrtValue* ort_value, _Out_ enum OrtSparseFormat* out);
// ORT_API2_STATUS(GetSparseTensorIndices, _In_ const OrtValue* ort_value, enum OrtSparseIndicesFormat indices_format, _Out_ size_t* num_indices, _Outptr_ const void** indices);
// ORT_API2_STATUS(GetSparseTensorIndicesTypeShape, _In_ const OrtValue* ort_value, enum OrtSparseIndicesFormat indices_format, _Outptr_ OrtTensorTypeAndShapeInfo** out);
// ORT_API2_STATUS(GetSparseTensorValues, _In_ const OrtValue* ort_value, _Outptr_ const void** out);
// ORT_API2_STATUS(GetSparseTensorValuesTypeAndShape, _In_ const OrtValue* ort_value, _Outptr_ OrtTensorTypeAndShapeInfo** out);
// ORT_API2_STATUS(GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s, size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len);
// ORT_API2_STATUS(GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* len);
// ORT_API2_STATUS(GetStringTensorElement, _In_ const OrtValue* value, size_t s_len, size_t index, _Out_writes_bytes_all_(s_len) void* s);
// ORT_API2_STATUS(GetStringTensorElementLength, _In_ const OrtValue* value, size_t index, _Out_ size_t* out);
// ORT_API2_STATUS(GetTensorRTProviderOptionsAsString, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);
// ORT_API2_STATUS(GetTensorRTProviderOptionsByName, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options, _In_ const char* key, _Outptr_ void** ptr);
// ORT_API2_STATUS(GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out);
// ORT_API2_STATUS(InvokeOp, _In_ const OrtKernelContext* context, _In_ const OrtOp* ort_op, _In_ const OrtValue* const* input_values, _In_ int input_count, _Inout_ OrtValue* const* output_values, _In_ int output_count);
// ORT_API2_STATUS(IsSparseTensor, _In_ const OrtValue* value, _Out_ int* out);
// ORT_API2_STATUS(KernelContext_GetAllocator, _In_ const OrtKernelContext* context, _In_ const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out);
// ORT_API2_STATUS(KernelContext_GetGPUComputeStream, _In_ const OrtKernelContext* context, _Outptr_ void** out);
// ORT_API2_STATUS(KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index, _Out_ const OrtValue** out);
// ORT_API2_STATUS(KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
// ORT_API2_STATUS(KernelContext_GetLogger, _In_ const OrtKernelContext* context, _Outptr_ const OrtLogger** logger);
// ORT_API2_STATUS(KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
// ORT_API2_STATUS(KernelContext_GetResource, _In_ const OrtKernelContext* context, _In_ int resouce_version, _In_ int resource_id, _Outptr_ void** resource);
// ORT_API2_STATUS(KernelInfoGetAttributeArray_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out, _Inout_ size_t* size);
// ORT_API2_STATUS(KernelInfoGetAttributeArray_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out, _Inout_ size_t* size);
// ORT_API2_STATUS(KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out);
// ORT_API2_STATUS(KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out);
// ORT_API2_STATUS(KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out, _Inout_ size_t* size);
// ORT_API2_STATUS(KernelInfoGetAttribute_tensor, _In_ const OrtKernelInfo* info, _In_z_ const char* name, _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out);
// ORT_API2_STATUS(KernelInfoGetConstantInput_tensor, _In_ const OrtKernelInfo* info, size_t index, _Out_ int* is_constant, _Outptr_ const OrtValue** out);
// ORT_API2_STATUS(KernelInfo_GetInputCount, _In_ const OrtKernelInfo* info, _Out_ size_t* out);
// ORT_API2_STATUS(KernelInfo_GetInputName, _In_ const OrtKernelInfo* info, size_t index, _Out_ char* out, _Inout_ size_t* size);
// ORT_API2_STATUS(KernelInfo_GetInputTypeInfo, _In_ const OrtKernelInfo* info, size_t index, _Outptr_ OrtTypeInfo** type_info);
// ORT_API2_STATUS(KernelInfo_GetLogger, _In_ const OrtKernelInfo* info, _Outptr_ const OrtLogger** logger);
// ORT_API2_STATUS(KernelInfo_GetNodeName, _In_ const OrtKernelInfo* info, _Out_ char* out, _Inout_ size_t* size);
// ORT_API2_STATUS(KernelInfo_GetOutputCount, _In_ const OrtKernelInfo* info, _Out_ size_t* out);
// ORT_API2_STATUS(KernelInfo_GetOutputName, _In_ const OrtKernelInfo* info, size_t index, _Out_ char* out, _Inout_ size_t* size);
// ORT_API2_STATUS(KernelInfo_GetOutputTypeInfo, _In_ const OrtKernelInfo* info, size_t index, _Outptr_ OrtTypeInfo** type_info);
// ORT_API2_STATUS(Logger_GetLoggingSeverityLevel, _In_ const OrtLogger* logger, _Out_ OrtLoggingLevel* out);
// ORT_API2_STATUS(Logger_LogMessage, _In_ const OrtLogger* logger, OrtLoggingLevel log_severity_level, _In_z_ const char* message, _In_z_ const ORTCHAR_T* file_path, int line_number, _In_z_ const char* func_name);
// ORT_API2_STATUS(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, void** library_handle);
// ORT_API2_STATUS(RegisterCustomOpsLibrary_V2, _Inout_ OrtSessionOptions* options, _In_ const ORTCHAR_T* library_name);
// ORT_API2_STATUS(RegisterCustomOpsUsingFunction, _Inout_ OrtSessionOptions* options, _In_ const char* registration_func_name);
// ORT_API2_STATUS(RunAsync, _Inout_ OrtSession* session, _In_opt_ const OrtRunOptions* run_options, _In_reads_(input_len) const char* const* input_names, _In_reads_(input_len) const OrtValue* const* input, size_t input_len, _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len, _Inout_updates_all_(output_names_len) OrtValue** output, _In_ RunAsyncCallbackFn run_async_callback, _In_opt_ void* user_data);
// ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_CANN, _In_ OrtSessionOptions* options, _In_ const OrtCANNProviderOptions* cann_options);
// ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* cuda_options);
// ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_Dnnl, _In_ OrtSessionOptions* options, _In_ const OrtDnnlProviderOptions* dnnl_options);
// ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_MIGraphX, _In_ OrtSessionOptions* options, _In_ const OrtMIGraphXProviderOptions* migraphx_options);
// ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_OpenVINO, _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options);
// ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_ROCM, _In_ OrtSessionOptions* options, _In_ const OrtROCMProviderOptions* rocm_options);
// ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_TensorRT, _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options);
// ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_TensorRT_V2, _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options);
// ORT_API2_STATUS(UpdateCANNProviderOptions, _Inout_ OrtCANNProviderOptions* cann_options, _In_reads_(num_keys) const char* const* provider_options_keys, _In_reads_(num_keys) const char* const* provider_options_values, _In_ size_t num_keys);
// ORT_API2_STATUS(UpdateCUDAProviderOptionsWithValue, _Inout_ OrtCUDAProviderOptionsV2* cuda_options, _In_ const char* key, _In_ void* value);
// ORT_API2_STATUS(UpdateDnnlProviderOptions, _Inout_ OrtDnnlProviderOptions* dnnl_options, _In_reads_(num_keys) const char* const* provider_options_keys, _In_reads_(num_keys) const char* const* provider_options_values, _In_ size_t num_keys);
// ORT_API2_STATUS(UpdateROCMProviderOptions, _Inout_ OrtROCMProviderOptions* rocm_options, _In_reads_(num_keys) const char* const* provider_options_keys, _In_reads_(num_keys) const char* const* provider_options_values, _In_ size_t num_keys);
// ORT_API2_STATUS(UpdateTensorRTProviderOptions, _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options, _In_reads_(num_keys) const char* const* provider_options_keys, _In_reads_(num_keys) const char* const* provider_options_values, _In_ size_t num_keys);
// ORT_API2_STATUS(UpdateTensorRTProviderOptionsWithValue, _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options, _In_ const char* key, _In_ void* value);
// ORT_API2_STATUS(UseBlockSparseIndices, _Inout_ OrtValue* ort_value, const int64_t* indices_shape, size_t indices_shape_len, _Inout_ int32_t* indices_data);
// ORT_API2_STATUS(UseCooIndices, _Inout_ OrtValue* ort_value, _Inout_ int64_t* indices_data, size_t indices_num);
// ORT_API2_STATUS(UseCsrIndices, _Inout_ OrtValue* ort_value, _Inout_ int64_t* inner_data, size_t inner_num, _Inout_ int64_t* outer_data, size_t outer_num);
// ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id);
// ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Dnnl, _In_ OrtSessionOptions* options, int use_arena);
// ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_MIGraphX, _In_ OrtSessionOptions* options, int device_id);
// ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_ROCM, _In_ OrtSessionOptions* options, int device_id);
// OrtMemType(ORT_API_CALL* GetInputMemoryType)(_In_ const struct OrtCustomOp* op, _In_ size_t index);
// OrtStatusPtr(ORT_API_CALL* CreateKernelV2)(_In_ const struct OrtCustomOp* op, _In_ const OrtApi* api, _In_ const OrtKernelInfo* info, _Out_ void** kernel);
// OrtStatusPtr(ORT_API_CALL* KernelComputeV2)(_In_ void* op_kernel, _In_ OrtKernelContext* context);
// const OrtTrainingApi*(ORT_API_CALL* GetTrainingApi)(uint32_t version) NO_EXCEPTION;
// int(ORT_API_CALL* GetVariadicInputHomogeneity)(_In_ const struct OrtCustomOp* op);
// int(ORT_API_CALL* GetVariadicInputMinArity)(_In_ const struct OrtCustomOp* op);
// int(ORT_API_CALL* GetVariadicOutputHomogeneity)(_In_ const struct OrtCustomOp* op);
// int(ORT_API_CALL* GetVariadicOutputMinArity)(_In_ const struct OrtCustomOp* op);
// void(ORT_API_CALL* ReleaseCANNProviderOptions)(_Frees_ptr_opt_ OrtCANNProviderOptions* input);
// void(ORT_API_CALL* ReleaseDnnlProviderOptions)(_Frees_ptr_opt_ OrtDnnlProviderOptions* input);
// void(ORT_API_CALL* ReleaseROCMProviderOptions)(_Frees_ptr_opt_ OrtROCMProviderOptions* input);
// void(ORT_API_CALL* ReleaseTensorRTProviderOptions)(_Frees_ptr_opt_ OrtTensorRTProviderOptionsV2* input);
