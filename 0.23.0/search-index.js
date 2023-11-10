var searchIndex = JSON.parse('{\
"onnxrt":{"doc":"<code>onnxruntime_c_api.h</code>","t":"HDDIDDEENENDNDDNEENNNNNNNNNNNNNNNNNNNNNNNNNNNNHRNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNENNNENEEESNNNENNNDDGDDDDDNDDDNDLLLLLLLLLLLLLLKFLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLOLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFMMMMMMMM","n":["ALLOCATOR_WITH_DEFAULT_OPTIONS","Allocator","ArenaCfg","AsONNXTensorElementDataType","CudaProviderOptions","Env","Error","ExecutionMode","FromVecWithNulError","GraphOptimizationLevel","IntoStringError","IoBinding","IoError","MemoryInfo","ModelMetadata","NulError","ONNXTensorElementDataType","ONNXType","ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16","ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL","ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128","ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64","ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE","ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT","ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16","ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN","ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ","ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2","ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ","ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16","ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32","ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64","ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8","ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING","ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16","ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32","ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64","ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8","ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED","ONNX_TYPE_MAP","ONNX_TYPE_OPAQUE","ONNX_TYPE_OPTIONAL","ONNX_TYPE_SEQUENCE","ONNX_TYPE_SPARSETENSOR","ONNX_TYPE_TENSOR","ONNX_TYPE_UNKNOWN","ORT_API","ORT_API_VERSION","ORT_DISABLE_ALL","ORT_ENABLE_ALL","ORT_ENABLE_BASIC","ORT_ENABLE_EXTENDED","ORT_ENGINE_ERROR","ORT_EP_FAIL","ORT_FAIL","ORT_INVALID_ARGUMENT","ORT_INVALID_GRAPH","ORT_INVALID_PROTOBUF","ORT_LOGGING_LEVEL_ERROR","ORT_LOGGING_LEVEL_FATAL","ORT_LOGGING_LEVEL_INFO","ORT_LOGGING_LEVEL_VERBOSE","ORT_LOGGING_LEVEL_WARNING","ORT_MODEL_LOADED","ORT_NOT_IMPLEMENTED","ORT_NO_MODEL","ORT_NO_SUCHFILE","ORT_OK","ORT_PARALLEL","ORT_PROJECTION_C","ORT_PROJECTION_CPLUSPLUS","ORT_PROJECTION_CSHARP","ORT_PROJECTION_JAVA","ORT_PROJECTION_NODEJS","ORT_PROJECTION_PYTHON","ORT_PROJECTION_WINML","ORT_RUNTIME_EXCEPTION","ORT_SEQUENTIAL","OrtAllocatorType","OrtArenaAllocator","OrtDeviceAllocator","OrtError","OrtErrorCode","OrtInvalidAllocator","OrtLanguageProjection","OrtLoggingLevel","OrtMemType","OrtMemTypeCPU","OrtMemTypeCPUInput","OrtMemTypeCPUOutput","OrtMemTypeDefault","OrtMemoryInfoDeviceType","OrtMemoryInfoDeviceType_CPU","OrtMemoryInfoDeviceType_FPGA","OrtMemoryInfoDeviceType_GPU","OrtPrepackedWeightsContainer","PrepackedWeightsContainer","Result","RunOptions","Session","SessionOptions","TensorTypeAndShapeInfo","ThreadingOptions","TryFromIntError","TypeInfo","UnownedMemoryInfo","UnownedTensorTypeAndShapeInfo","Utf8Error","Value","alloc","allocator_name","allocator_name","allocator_type","allocator_type","append_execution_provider","append_execution_provider_cpu","append_execution_provider_cuda","append_execution_provider_unchecked","append_execution_provider_with_bytes_with_nul","append_execution_provider_with_bytes_with_nul_unchecked","append_execution_provider_with_c_chars_with_nul","append_execution_provider_with_c_str","append_execution_provider_with_c_str_unchecked","as_onnx_tensor_element_data_type","available_providers","bind_input","bind_output","bind_output_to_device","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","bound_output_names","bound_output_values","build_info_string","cast_to_tensor_type_info","clear_bound_inputs","clear_bound_outputs","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","create_and_register_allocator","create_and_register_allocator_unchecked","create_and_register_allocator_with_bytes_with_nul","create_and_register_allocator_with_bytes_with_nul_unchecked","create_and_register_allocator_with_c_chars_with_nul","create_and_register_allocator_with_c_str","create_and_register_allocator_with_c_str_unchecked","current_gpu_device_id","custom_metadata_map_keys","default","default","default","default","default","default","denotation","description","device_id","device_id","device_type","device_type","dimensions","dimensions","dimensions_count","dimensions_count","disable_cpu_mem_arena","disable_mem_pattern","disable_per_session_threads","disable_profiling","disable_telemetry_events","domain","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","element_count","element_count","element_type","element_type","enable_cpu_mem_arena","enable_mem_pattern","enable_ort_custom_ops","enable_profiling","enable_telemetry_events","end_profiling","eq","eq","eq","eq","eq","eq","eq","eq","eq","eq","eq","eq","eq","eq","equivalent","equivalent","equivalent","equivalent","equivalent","equivalent","equivalent","equivalent","equivalent","equivalent","execution_provider_api","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","free","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","get","get_mut","graph_description","graph_name","has_session_config_entry","has_value","hash","hash","hash","hash","hash","hash","hash","hash","hash","hash","impl_AsONNXTensorElementDataType","input_count","input_name","input_name_using_allocator","input_names","input_names_using_allocator","input_type_info","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","is_tensor","log_severity_level","log_verbosity_level","lookup_custom_metadata_map","memory_info","memory_type","memory_type","model_metadata","new","new","new","new","new","new","new","new","new","new","new","new_for_cpu","new_tensor","new_tensor_using_allocator","new_tensor_with_data","new_unchecked","new_with_c_chars_with_nul","new_with_c_str","new_with_c_str_unchecked","new_with_custom_logger","new_with_custom_logger_and_global_thread_pools","new_with_global_thread_pools","new_with_model_data","new_with_model_data_and_prepacked_weights_container","new_with_model_path","new_with_model_path_and_prepacked_weights_container","onnx_type","output_count","output_name","output_name_using_allocator","output_names","output_names_using_allocator","output_type_info","overridable_initializer_count","overridable_initializer_name","overridable_initializer_name_using_allocator","overridable_initializer_names","overridable_initializer_names_using_allocator","overridable_initializer_type_info","producer_name","profiling_start_time_ns","register_allocator","run","run_unchecked","run_with_binding","run_with_bytes_with_nul","run_with_bytes_with_nul_unchecked","run_with_c_chars_with_nul","run_with_c_str","run_with_c_str_unchecked","session_config_entry","set","set","set_current_gpu_device_id","set_custom_create_thread_fn","set_custom_join_thread_fn","set_custom_log_level","set_custom_thread_creation_options","set_dimensions","set_element_type","set_execution_mode","set_external_initializers","set_external_initializers_unchecked","set_external_initializers_with_bytes_with_nul","set_external_initializers_with_bytes_with_nul_unchecked","set_external_initializers_with_c_chars_with_nul","set_external_initializers_with_c_str","set_external_initializers_with_c_str_unchecked","set_free_dimension_by_denotation","set_free_dimension_by_name","set_global_custom_create_thread_fn","set_global_custom_join_thread_fn","set_global_custom_thread_creation_options","set_global_denormal_as_zero","set_global_inter_op_num_threads","set_global_intra_op_num_threads","set_global_intra_op_thread_affinity","set_global_spin_control","set_graph_optimization_level","set_initializer","set_inter_op_num_threads","set_intra_op_num_threads","set_language_projection","set_log_id","set_log_severity_level","set_log_severity_level","set_log_verbosity_level","set_log_verbosity_level","set_optimized_model_file_path","set_session_config_entry","set_tag","set_terminate","set_unchecked","set_with_bytes_with_nul","set_with_bytes_with_nul","set_with_bytes_with_nul_unchecked","set_with_bytes_with_nul_unchecked","set_with_c_chars_with_nul","set_with_c_chars_with_nul","set_with_c_str","set_with_c_str","set_with_c_str_unchecked","source","symbolic_dimensions","symbolic_dimensions","synchronize_bound_inputs","synchronize_bound_outputs","tag","tensor_data","tensor_data_mut","tensor_memory_info","tensor_type_info","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_string","to_string","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_info","unregister_allocator","unset_terminate","value_type","version","version_string","code","message","source","source","source","source","source","source"],"q":[[0,"onnxrt"],[648,"onnxrt::Error"]],"d":["","","","","","","","","","\\\\brief Graph optimization level","","","","","","","Copied from TensorProto::DataType Currently, Ort doesn’t …","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","&lt; Error messages.","&lt; Fatal error messages (most severe).","&lt; Informational messages.","&lt; Verbose informational messages (least severe).","&lt; Warning messages.","","","","","","","","","","","","","","","","","","","","","","\\\\brief Language projection identifiers /see …","\\\\brief Logging severity levels","\\\\brief Memory types for allocated memory, execution …","","&lt; Any CPU memory used by non-CPU execution provider","&lt; CPU accessible memory outputted by non-CPU execution …","&lt; The default allocator for execution provider","\\\\brief This mimics OrtDevice type constants so they can be …","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>cpu_provider_factory.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","","","","","Returns the argument unchanged.","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","","<code>onnxruntime_c_api.h</code>","","","<code>onnxruntime_c_api.h</code>","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","<code>onnxruntime_c_api.h</code>","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","<code>onnxruntime_c_api.h</code>","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","",""],"i":[0,0,0,0,0,0,0,0,45,0,45,0,45,0,0,45,0,0,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,23,23,23,23,23,23,23,0,0,27,27,27,27,25,25,25,25,25,25,24,24,24,24,24,25,25,25,25,25,28,29,29,29,29,29,29,29,25,28,0,6,6,45,0,6,0,0,0,30,30,30,30,0,31,31,31,0,0,0,0,0,0,0,0,45,0,0,0,45,0,1,2,5,2,5,7,7,7,7,7,7,7,7,7,58,0,18,18,18,45,32,36,7,35,42,37,21,20,19,2,5,1,18,38,33,39,11,15,23,24,25,26,27,28,29,6,30,31,45,32,36,7,35,42,37,21,20,19,2,5,1,18,38,33,39,11,15,23,24,25,26,27,28,29,6,30,31,18,18,0,20,18,18,7,21,5,15,23,24,25,26,27,28,29,6,30,31,7,21,5,15,23,24,25,26,27,28,29,6,30,31,32,32,32,32,32,32,32,0,35,36,7,37,19,38,39,20,35,2,5,2,5,37,21,37,21,7,7,7,7,32,35,32,36,7,35,42,37,20,19,2,1,18,38,33,39,11,37,21,37,21,7,7,7,7,32,42,2,2,5,5,15,23,24,25,27,28,29,6,30,31,15,23,24,25,27,28,29,6,30,31,0,45,45,32,36,7,35,42,37,21,20,19,2,5,1,18,38,33,39,11,15,23,24,25,26,27,28,29,6,30,31,1,45,45,45,45,45,45,45,32,36,7,35,42,37,21,20,19,2,5,1,18,38,33,39,11,15,23,24,25,26,27,28,29,6,30,31,19,19,35,35,7,19,15,23,24,25,27,28,29,6,30,31,0,42,42,42,42,42,42,45,32,36,7,35,42,37,21,20,19,2,5,1,18,38,33,39,11,15,23,24,25,26,27,28,29,6,30,31,19,36,36,35,1,2,5,42,32,36,7,37,2,1,18,38,33,39,11,2,19,19,19,33,33,33,33,32,32,32,42,42,42,42,20,42,42,42,42,42,42,42,42,42,42,42,42,35,42,32,42,42,42,42,42,42,42,42,7,36,11,0,7,7,32,7,37,37,7,7,7,7,7,7,7,7,7,7,38,38,38,38,38,38,38,38,7,7,7,7,32,7,36,7,36,7,7,7,36,36,11,36,11,36,11,36,11,36,11,11,45,37,21,18,18,36,19,19,19,19,7,21,5,15,23,24,25,26,27,28,29,6,30,31,45,11,45,32,36,7,35,42,37,21,20,19,2,5,1,18,38,33,39,11,15,23,24,25,26,27,28,29,6,30,31,45,32,36,7,35,42,37,21,20,19,2,5,1,18,38,33,39,11,15,23,24,25,26,27,28,29,6,30,31,45,32,36,7,35,42,37,21,20,19,2,5,1,18,38,33,39,11,15,23,24,25,26,27,28,29,6,30,31,19,32,36,19,35,0,67,67,68,69,70,71,72,73],"f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[1],[2,[[4,[3]]]],[5,[[4,[3]]]],[2,6],[5,6],[[7,3,[9,[[8,[3]]]],[9,[[8,[3]]]]],[[4,[7]]]],[[7,10],[[4,[7]]]],[[7,11],[[4,[7]]]],[[7,3,[9,[[8,[3]]]],[9,[[8,[3]]]]],[[4,[7]]]],[[7,[9,[12]],[9,[[9,[12]]]],[9,[[9,[12]]]]],[[4,[7]]]],[[7,[9,[12]],[9,[[9,[12]]]],[9,[[9,[12]]]]],[[4,[7]]]],[[7,13,[9,[13]],[9,[13]]],[[4,[7]]]],[[7,14,[9,[[8,[14]]]],[9,[[8,[14]]]]],[[4,[7]]]],[[7,14,[9,[[8,[14]]]],[9,[[8,[14]]]]],[[4,[7]]]],[[],15],[[],[[4,[[17,[16]]]]]],[[18,3,19],[[4,[18]]]],[[18,3,19],[[4,[18]]]],[[18,3,2],[[4,[18]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[18,1],[[4,[[17,[16]]]]]],[[18,1],[[4,[[17,[19]]]]]],[[],3],[20,[[22,[21]]]],[18],[18],[7,7],[21,21],[5,5],[15,15],[23,23],[24,24],[25,25],[26,26],[27,27],[28,28],[29,29],[6,6],[30,30],[31,31],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[32,3,2,[22,[33]],[9,[[8,[3]]]],[9,[[8,[3]]]]],4],[[32,3,2,[22,[33]],[9,[[8,[3]]]],[9,[[8,[3]]]]],4],[[32,[9,[12]],2,[22,[33]],[9,[[9,[12]]]],[9,[[9,[12]]]]],4],[[32,[9,[12]],2,[22,[33]],[9,[[9,[12]]]],[9,[[9,[12]]]]],4],[[32,13,2,[22,[33]],[9,[13]],[9,[13]]],4],[[32,14,2,[22,[33]],[9,[[8,[14]]]],[9,[[8,[14]]]]],4],[[32,14,2,[22,[33]],[9,[[8,[14]]]],[9,[[8,[14]]]]],4],[[],[[4,[34]]]],[[35,1],[[4,[[17,[16]]]]]],[[],36],[[],7],[[],37],[[],19],[[],38],[[],39],[20,[[4,[3]]]],[[35,1],[[4,[16]]]],[2,34],[5,34],[2,31],[5,31],[37,[[17,[40]]]],[21,[[17,[40]]]],[37,41],[21,41],[7,7],[7,7],[7,7],[7,7],[32],[[35,1],[[4,[16]]]],[32],[36],[7],[35],[42],[37],[20],[19],[2],[1],[18],[38],[33],[39],[11],[37,[[4,[40]]]],[21,[[4,[40]]]],[37,15],[21,15],[7,7],[7,7],[7,7],[[7,[8,[43]]],[[4,[7]]]],[32],[[42,1],[[4,[16]]]],[[2,2],10],[[2,5],10],[[5,5],10],[[5,2],10],[[15,15],10],[[23,23],10],[[24,24],10],[[25,25],10],[[27,27],10],[[28,28],10],[[29,29],10],[[6,6],10],[[30,30],10],[[31,31],10],[[],10],[[],10],[[],10],[[],10],[[],10],[[],10],[[],10],[[],10],[[],10],[[],10],[3,[[4,[44]]]],[[45,46],47],[[45,46],47],[[32,46],47],[[36,46],47],[[7,46],47],[[35,46],47],[[42,46],47],[[37,46],47],[[21,46],47],[[20,46],47],[[19,46],47],[[2,46],47],[[5,46],47],[[1,46],47],[[18,46],47],[[38,46],47],[[33,46],47],[[39,46],47],[[11,46],47],[[15,46],[[49,[48]]]],[[23,46],[[49,[48]]]],[[24,46],[[49,[48]]]],[[25,46],[[49,[48]]]],[[26,46],[[49,[48]]]],[[27,46],[[49,[48]]]],[[28,46],[[49,[48]]]],[[29,46],[[49,[48]]]],[[6,46],[[49,[48]]]],[[30,46],[[49,[48]]]],[[31,46],[[49,[48]]]],[1],[50,45],[51,45],[52,45],[53,45],[[]],[54,45],[55,45],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[19,[9,[40]]],4],[[19,[9,[40]]],4],[[35,1],[[4,[16]]]],[[35,1],[[4,[16]]]],[[7,3],[[4,[10]]]],[19,10],[[15,56]],[[23,56]],[[24,56]],[[25,56]],[[27,56]],[[28,56]],[[29,56]],[[6,56]],[[30,56]],[[31,56]],0,[42,[[4,[41]]]],[[42,41],[[4,[16]]]],[[42,41,1],[[4,[16]]]],[42,[[4,[[17,[16]]]]]],[[42,1],[[4,[57]]]],[[42,41],[[4,[20]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[19,10],[36,34],[36,34],[[35,1,3],[[4,[[22,[16]]]]]],[1,5],[2,30],[5,30],[42,[[4,[35]]]],[[24,3],[[4,[32]]]],[[],36],[[],7],[[],37],[[3,6,34,30],[[4,[2]]]],[[42,2],[[4,[1]]]],[42,[[4,[18]]]],[[],38],[[[9,[[8,[3]]]],[9,[41]]],[[4,[33]]]],[[],39],[[],[[4,[11]]]],[[6,30],2],[[[9,[40]],15],[[4,[19]]]],[[1,[9,[40]],15],[[4,[19]]]],[[2,[9,[58]],[9,[40]]],[[4,[19]]]],[[[9,[[8,[3]]]],[9,[41]]],[[4,[33]]]],[[[9,[13]],[9,[41]]],[[4,[33]]]],[[[9,[[8,[14]]]],[9,[41]]],[[4,[33]]]],[[[9,[[8,[14]]]],[9,[41]]],[[4,[33]]]],[[59,22,24,3],[[4,[32]]]],[[59,22,24,3,38],[[4,[32]]]],[[24,3,38],[[4,[32]]]],[[[61,[[60,[32]]]],[9,[12]],7],[[4,[42]]]],[[[61,[[60,[32]]]],[9,[12]],7,[61,[[60,[39]]]]],[[4,[42]]]],[[[61,[[60,[32]]]],[8,[43]],7],[[4,[42]]]],[[[61,[[60,[32]]]],[8,[43]],7,[61,[[60,[39]]]]],[[4,[42]]]],[20,23],[42,[[4,[41]]]],[[42,41],[[4,[16]]]],[[42,41,1],[[4,[16]]]],[42,[[4,[[17,[16]]]]]],[[42,1],[[4,[57]]]],[[42,41],[[4,[20]]]],[42,[[4,[41]]]],[[42,41],[[4,[16]]]],[[42,41,1],[[4,[16]]]],[42,[[4,[[17,[16]]]]]],[[42,1],[[4,[57]]]],[[42,41],[[4,[20]]]],[[35,1],[[4,[16]]]],[42,62],[[32,1],4],[[42,[22,[36]],[9,[[8,[3]]]],[9,[19]],[9,[[8,[3]]]],[9,[19]]],4],[[42,[22,[36]],[9,[[8,[3]]]],[9,[19]],[9,[[8,[3]]]],[9,[19]]],4],[[42,36,18],4],[[42,[22,[36]],[9,[[9,[12]]]],[9,[19]],[9,[[9,[12]]]],[9,[19]]],4],[[42,[22,[36]],[9,[[9,[12]]]],[9,[19]],[9,[[9,[12]]]],[9,[19]]],4],[[42,[22,[36]],[9,[13]],[9,[19]],[9,[13]],[9,[19]]],4],[[42,[22,[36]],[9,[[8,[14]]]],[9,[19]],[9,[[8,[14]]]],[9,[19]]],4],[[42,[22,[36]],[9,[[8,[14]]]],[9,[19]],[9,[[8,[14]]]],[9,[19]]],4],[[7,3],[[4,[16]]]],[[36,[8,[3]],[8,[3]]],[[4,[36]]]],[[11,[9,[[8,[3]]]],[9,[[8,[3]]]]],4],[34,4],[[7,63],7],[[7,64],7],[[32,24]],[[7,44],7],[[37,[9,[40]]],37],[[37,15],37],[[7,28],7],[[7,[9,[[8,[3]]]],[9,[19]]],[[4,[7]]]],[[7,[9,[[8,[3]]]],[9,[19]]],[[4,[7]]]],[[7,[9,[[9,[12]]]],[9,[19]]],[[4,[7]]]],[[7,[9,[[9,[12]]]],[9,[19]]],[[4,[7]]]],[[7,[9,[13]],[9,[19]]],[[4,[7]]]],[[7,[9,[[8,[14]]]],[9,[19]]],[[4,[7]]]],[[7,[9,[[8,[14]]]],[9,[19]]],[[4,[7]]]],[[7,3,40],[[4,[7]]]],[[7,3,40],[[4,[7]]]],[[38,63],38],[[38,64],38],[[38,44],38],[38,38],[[38,34],38],[[38,34],38],[[38,3],[[4,[38]]]],[[38,10],38],[[7,27],7],[[7,3,19],[[4,[7]]]],[[7,34],7],[[7,34],7],[[32,29]],[[7,3],[[4,[7]]]],[[36,34],36],[[7,34],7],[[36,34],36],[[7,34],7],[[7,[8,[43]]],[[4,[7]]]],[[7,3,3],[[4,[7]]]],[[36,3],[[4,[36]]]],[36,36],[[11,[9,[[8,[3]]]],[9,[[8,[3]]]]],4],[[36,[9,[12]],[9,[12]]],[[4,[36]]]],[[11,[9,[[9,[12]]]],[9,[[9,[12]]]]],4],[[36,[9,[12]],[9,[12]]],[[4,[36]]]],[[11,[9,[[9,[12]]]],[9,[[9,[12]]]]],4],[[36,13,13],[[4,[36]]]],[[11,[9,[13]],[9,[13]]],4],[[36,[8,[14]],[8,[14]]],[[4,[36]]]],[[11,[9,[[8,[14]]]],[9,[[8,[14]]]]],4],[[11,[9,[[8,[14]]]],[9,[[8,[14]]]]],4],[45,[[22,[65]]]],[37,[[4,[[17,[3]]]]]],[21,[[4,[[17,[3]]]]]],[18,4],[18,4],[36,[[4,[3]]]],[19,[[4,[9]]]],[19,[[4,[9]]]],[19,5],[19,[[4,[37]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],16],[[11,1],[[4,[16]]]],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],49],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[[],66],[19,[[4,[[22,[20]]]]]],[[32,2],4],[36,36],[19,[[4,[23]]]],[35,40],[[],3],0,0,0,0,0,0,0,0],"c":[],"p":[[3,"Allocator"],[3,"MemoryInfo"],[15,"str"],[6,"Result"],[3,"UnownedMemoryInfo"],[4,"OrtAllocatorType"],[3,"SessionOptions"],[8,"AsRef"],[15,"slice"],[15,"bool"],[3,"CudaProviderOptions"],[15,"u8"],[6,"c_char"],[3,"CStr"],[4,"ONNXTensorElementDataType"],[3,"String"],[3,"Vec"],[3,"IoBinding"],[3,"Value"],[3,"TypeInfo"],[3,"UnownedTensorTypeAndShapeInfo"],[4,"Option"],[4,"ONNXType"],[4,"OrtLoggingLevel"],[4,"OrtErrorCode"],[3,"OrtPrepackedWeightsContainer"],[4,"GraphOptimizationLevel"],[4,"ExecutionMode"],[4,"OrtLanguageProjection"],[4,"OrtMemType"],[4,"OrtMemoryInfoDeviceType"],[3,"Env"],[3,"ArenaCfg"],[15,"i32"],[3,"ModelMetadata"],[3,"RunOptions"],[3,"TensorTypeAndShapeInfo"],[3,"ThreadingOptions"],[3,"PrepackedWeightsContainer"],[15,"i64"],[15,"usize"],[3,"Session"],[3,"Path"],[6,"c_void"],[4,"Error"],[3,"Formatter"],[6,"Result"],[3,"Error"],[4,"Result"],[3,"NulError"],[3,"Error"],[3,"Utf8Error"],[3,"IntoStringError"],[3,"FromVecWithNulError"],[3,"TryFromIntError"],[8,"Hasher"],[8,"Iterator"],[8,"AsONNXTensorElementDataType"],[6,"OrtLoggingFunction"],[3,"Mutex"],[3,"Arc"],[15,"u64"],[6,"OrtCustomCreateThreadFn"],[6,"OrtCustomJoinThreadFn"],[8,"Error"],[3,"TypeId"],[13,"OrtError"],[13,"FromVecWithNulError"],[13,"IntoStringError"],[13,"IoError"],[13,"NulError"],[13,"TryFromIntError"],[13,"Utf8Error"]]}\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};
