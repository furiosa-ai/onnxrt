var searchIndex = JSON.parse('{\
"onnxrt":{"doc":"<code>onnxruntime_c_api.h</code>","t":[7,3,3,8,3,4,4,4,13,3,13,3,3,13,4,4,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,7,17,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,4,13,13,13,4,4,4,4,18,13,13,13,6,3,3,3,3,3,13,3,3,3,13,3,11,11,11,11,11,11,11,11,11,10,5,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,5,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,14,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,5,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12],"n":["ALLOCATOR_WITH_DEFAULT_OPTIONS","Allocator","ArenaCfg","AsONNXTensorElementDataType","Env","Error","ExecutionMode","GraphOptimizationLevel","Invalid","IoBinding","IoError","MemoryInfo","ModelMetadata","NulError","ONNXTensorElementDataType","ONNXType","ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16","ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL","ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128","ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64","ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE","ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT","ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16","ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16","ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32","ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64","ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8","ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING","ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16","ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32","ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64","ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8","ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED","ONNX_TYPE_MAP","ONNX_TYPE_OPAQUE","ONNX_TYPE_SEQUENCE","ONNX_TYPE_SPARSETENSOR","ONNX_TYPE_TENSOR","ONNX_TYPE_UNKNOWN","ORT_API","ORT_API_VERSION","ORT_DISABLE_ALL","ORT_ENABLE_ALL","ORT_ENABLE_BASIC","ORT_ENABLE_EXTENDED","ORT_ENGINE_ERROR","ORT_EP_FAIL","ORT_FAIL","ORT_INVALID_ARGUMENT","ORT_INVALID_GRAPH","ORT_INVALID_PROTOBUF","ORT_LOGGING_LEVEL_ERROR","ORT_LOGGING_LEVEL_FATAL","ORT_LOGGING_LEVEL_INFO","ORT_LOGGING_LEVEL_VERBOSE","ORT_LOGGING_LEVEL_WARNING","ORT_MODEL_LOADED","ORT_NOT_IMPLEMENTED","ORT_NO_MODEL","ORT_NO_SUCHFILE","ORT_OK","ORT_PARALLEL","ORT_PROJECTION_C","ORT_PROJECTION_CPLUSPLUS","ORT_PROJECTION_CSHARP","ORT_PROJECTION_JAVA","ORT_PROJECTION_NODEJS","ORT_PROJECTION_PYTHON","ORT_PROJECTION_WINML","ORT_RUNTIME_EXCEPTION","ORT_SEQUENTIAL","OrtAllocatorType","OrtArenaAllocator","OrtDeviceAllocator","OrtError","OrtErrorCode","OrtLanguageProjection","OrtLoggingLevel","OrtMemType","OrtMemTypeCPU","OrtMemTypeCPUInput","OrtMemTypeCPUOutput","OrtMemTypeDefault","Result","RunOptions","Session","SessionOptions","TensorTypeAndShapeInfo","ThreadingOptions","TryFromIntError","TypeInfo","UnownedMemoryInfo","UnownedTensorTypeAndShapeInfo","Utf8Error","Value","add_free_dimension_override","add_free_dimension_override_by_name","add_initializer","add_session_config_entry","alloc","allocator_name","allocator_name","allocator_type","allocator_type","as_onnx_tensor_element_data_type","available_providers","bind_input","bind_output","bind_output_to_device","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","bound_output_names","bound_output_values","cast_to_tensor_type_info","clear_bound_inputs","clear_bound_outputs","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","clone_into","create_and_register_allocator","current_gpu_device_id","custom_metadata_map_keys","default","default","default","default","default","denotation","description","device_id","device_id","dimensions","dimensions","dimensions_count","dimensions_count","disable_cpu_mem_arena","disable_mem_pattern","disable_per_session_threads","disable_profiling","disable_telemetry_events","domain","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","element_count","element_count","element_type","element_type","enable_cpu_mem_arena","enable_mem_pattern","enable_profiling","enable_telemetry_events","end_profiling","eq","eq","eq","eq","eq","eq","eq","eq","eq","eq","eq","eq","eq","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","free","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","get","get_mut","graph_description","graph_name","hash","hash","hash","hash","hash","hash","hash","hash","hash","impl_AsONNXTensorElementDataType","input_count","input_name","input_name_using_allocator","input_names","input_names_using_allocator","input_type_info","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","is_tensor","log_severity_level","log_verbosity_level","lookup_custom_metadata_map","memory_info","memory_type","memory_type","model_metadata","new","new","new","new","new","new","new","new","new","new_for_cpu","new_tensor","new_tensor_using_allocator","new_tensor_with_data","new_with_custom_logger","new_with_custom_logger_and_global_thread_pools","new_with_global_thread_pools","new_with_model_data","new_with_model_path","onnx_type","output_count","output_name","output_name_using_allocator","output_names","output_names_using_allocator","output_type_info","overridable_initializer_count","overridable_initializer_name","overridable_initializer_name_using_allocator","overridable_initializer_names","overridable_initializer_names_using_allocator","overridable_initializer_type_info","producer_name","profiling_start_time_ns","run","run_unchecked","run_with_binding","run_with_bytes_with_nul","run_with_bytes_with_nul_unchecked","run_with_c_chars_with_nul","run_with_c_str","run_with_c_str_unchecked","set_current_gpu_device_id","set_dimensions","set_element_type","set_execution_mode","set_global_denormal_as_zero","set_global_inter_op_num_threads","set_global_intra_op_num_threads","set_global_spin_control","set_graph_optimization_level","set_inter_op_num_threads","set_intra_op_num_threads","set_language_projection","set_log_id","set_log_severity_level","set_log_severity_level","set_log_verbosity_level","set_log_verbosity_level","set_optimized_model_file_path","set_tag","set_terminate","source","symbolic_dimensions","symbolic_dimensions","tag","tensor_data","tensor_data_mut","tensor_type_info","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_owned","to_string","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_info","unset_terminate","value_type","version","code","message","source","source","source","source"],"q":["onnxrt","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","onnxrt::Error","","","","",""],"d":["","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","memory types for allocator, exec provider specific types …","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","","<code>onnxruntime_c_api.h</code>","","","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","<code>onnxruntime_c_api.h</code>","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","<code>onnxruntime_c_api.h</code>","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","Safety","<code>onnxruntime_c_api.h</code>","","Safety","<code>onnxruntime_c_api.h</code>","","Safety","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","<code>onnxruntime_c_api.h</code>","","","","","",""],"i":[0,0,0,0,0,0,0,0,1,0,2,0,0,2,0,0,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,0,0,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,6,6,6,6,6,8,9,9,9,9,9,9,9,6,8,0,1,1,2,0,0,0,0,10,10,10,10,0,0,0,0,0,0,2,0,0,0,2,0,11,11,11,11,12,13,14,13,14,15,0,16,16,16,2,17,18,11,19,20,21,22,23,24,13,14,12,16,25,26,3,4,7,6,5,8,9,1,10,2,17,18,11,19,20,21,22,23,24,13,14,12,16,25,26,3,4,7,6,5,8,9,1,10,16,16,23,16,16,11,22,14,3,4,7,6,5,8,9,1,10,11,22,14,3,4,7,6,5,8,9,1,10,17,0,19,18,11,21,24,25,23,19,13,14,21,22,21,22,11,11,11,11,17,19,17,18,11,19,20,21,23,24,13,12,16,25,26,21,22,21,22,11,11,11,17,20,13,13,14,14,3,4,7,6,5,8,9,1,10,2,2,17,18,11,19,20,21,22,23,24,13,14,12,16,25,26,3,4,7,6,5,8,9,1,10,12,2,2,2,2,2,17,18,11,19,20,21,22,23,24,13,14,12,16,25,26,3,4,7,6,5,8,9,1,10,24,24,19,19,3,4,7,6,5,8,9,1,10,0,20,20,20,20,20,20,2,17,18,11,19,20,21,22,23,24,13,14,12,16,25,26,3,4,7,6,5,8,9,1,10,24,18,18,19,12,13,14,20,17,18,11,21,13,12,16,25,26,13,24,24,24,17,17,17,20,20,23,20,20,20,20,20,20,20,20,20,20,20,20,19,20,20,20,20,20,20,20,20,20,0,21,21,11,25,25,25,25,11,11,11,17,11,18,11,18,11,11,18,18,2,21,22,18,24,24,24,11,22,14,3,4,7,6,5,8,9,1,10,2,2,17,18,11,19,20,21,22,23,24,13,14,12,16,25,26,3,4,7,6,5,8,9,1,10,2,17,18,11,19,20,21,22,23,24,13,14,12,16,25,26,3,4,7,6,5,8,9,1,10,2,17,18,11,19,20,21,22,23,24,13,14,12,16,25,26,3,4,7,6,5,8,9,1,10,24,18,24,19,27,27,28,29,30,31],"f":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,[[["i64",15],["str",15]],["result",6]],[[["i64",15],["str",15]],["result",6]],[[["value",3],["str",15]],["result",6]],[[["str",15]],["result",6]],[[]],[[],[["result",6],["str",15]]],[[],[["result",6],["str",15]]],[[],["ortallocatortype",4]],[[],["ortallocatortype",4]],[[],["onnxtensorelementdatatype",4]],[[],[["vec",3],["result",6]]],[[["value",3],["str",15]],["result",6]],[[["value",3],["str",15]],["result",6]],[[["str",15],["memoryinfo",3]],["result",6]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[["allocator",3]],[["vec",3],["result",6]]],[[["allocator",3]],[["vec",3],["result",6]]],[[],[["unownedtensortypeandshapeinfo",3],["option",4]]],[[]],[[]],[[]],[[],["unownedtensortypeandshapeinfo",3]],[[],["unownedmemoryinfo",3]],[[],["onnxtensorelementdatatype",4]],[[],["onnxtype",4]],[[],["ortlogginglevel",4]],[[],["orterrorcode",4]],[[],["graphoptimizationlevel",4]],[[],["executionmode",4]],[[],["ortlanguageprojection",4]],[[],["ortallocatortype",4]],[[],["ortmemtype",4]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[["arenacfg",3],["option",4],["memoryinfo",3]],["result",6]],[[],[["i32",15],["result",6]]],[[["allocator",3]],[["vec",3],["result",6]]],[[]],[[]],[[]],[[]],[[]],[[],[["result",6],["str",15]]],[[["allocator",3]],[["result",6],["string",3]]],[[],["i32",15]],[[],["i32",15]],[[],[["vec",3],["i64",15]]],[[],[["vec",3],["i64",15]]],[[],["usize",15]],[[],["usize",15]],[[]],[[]],[[]],[[]],[[]],[[["allocator",3]],[["result",6],["string",3]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],["i64",15]],[[],["i64",15]],[[],["onnxtensorelementdatatype",4]],[[],["onnxtensorelementdatatype",4]],[[]],[[]],[[["asref",8],["path",3]],["result",6]],[[]],[[["allocator",3]],[["result",6],["string",3]]],[[["memoryinfo",3]],["bool",15]],[[["unownedmemoryinfo",3]],["bool",15]],[[["memoryinfo",3]],["bool",15]],[[["unownedmemoryinfo",3]],["bool",15]],[[["onnxtensorelementdatatype",4]],["bool",15]],[[["onnxtype",4]],["bool",15]],[[["ortlogginglevel",4]],["bool",15]],[[["orterrorcode",4]],["bool",15]],[[["graphoptimizationlevel",4]],["bool",15]],[[["executionmode",4]],["bool",15]],[[["ortlanguageprojection",4]],["bool",15]],[[["ortallocatortype",4]],["bool",15]],[[["ortmemtype",4]],["bool",15]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],[["result",4],["error",3]]],[[["formatter",3]],[["result",4],["error",3]]],[[["formatter",3]],[["result",4],["error",3]]],[[["formatter",3]],[["result",4],["error",3]]],[[["formatter",3]],[["result",4],["error",3]]],[[["formatter",3]],[["result",4],["error",3]]],[[["formatter",3]],[["result",4],["error",3]]],[[["formatter",3]],[["result",4],["error",3]]],[[["formatter",3]],[["result",4],["error",3]]],[[]],[[]],[[["error",3]]],[[["tryfrominterror",3]]],[[["utf8error",3]]],[[["nulerror",3]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],["result",6]],[[],["result",6]],[[["allocator",3]],[["result",6],["string",3]]],[[["allocator",3]],[["result",6],["string",3]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],null,[[],[["usize",15],["result",6]]],[[["usize",15]],[["result",6],["string",3]]],[[["usize",15],["allocator",3]],[["result",6],["string",3]]],[[],[["vec",3],["result",6]]],[[["allocator",3]],["result",6]],[[["usize",15]],[["result",6],["typeinfo",3]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],["bool",15]],[[],["i32",15]],[[],["i32",15]],[[["str",15],["allocator",3]],[["option",4],["result",6]]],[[],["unownedmemoryinfo",3]],[[],["ortmemtype",4]],[[],["ortmemtype",4]],[[],[["result",6],["modelmetadata",3]]],[[["str",15],["ortlogginglevel",4]],["result",6]],[[]],[[]],[[]],[[["ortmemtype",4],["ortallocatortype",4],["str",15],["i32",15]],["result",6]],[[["session",3],["memoryinfo",3]],["result",6]],[[["session",3]],["result",6]],[[]],[[["usize",15],["i32",15]],["result",6]],[[["ortmemtype",4],["ortallocatortype",4]]],[[["onnxtensorelementdatatype",4]],["result",6]],[[["onnxtensorelementdatatype",4],["allocator",3]],["result",6]],[[["memoryinfo",3]],["result",6]],[[["str",15],["ortlogginglevel",4],["ortloggingfunction",6],["option",4]],["result",6]],[[["str",15],["ortlogginglevel",4],["ortloggingfunction",6],["threadingoptions",3],["option",4]],["result",6]],[[["threadingoptions",3],["str",15],["ortlogginglevel",4]],["result",6]],[[["mutex",3],["sessionoptions",3],["arc",3]],["result",6]],[[["asref",8],["sessionoptions",3],["mutex",3],["path",3],["arc",3]],["result",6]],[[],["onnxtype",4]],[[],[["usize",15],["result",6]]],[[["usize",15]],[["result",6],["string",3]]],[[["usize",15],["allocator",3]],[["result",6],["string",3]]],[[],[["vec",3],["result",6]]],[[["allocator",3]],["result",6]],[[["usize",15]],[["result",6],["typeinfo",3]]],[[],[["usize",15],["result",6]]],[[["usize",15]],[["result",6],["string",3]]],[[["usize",15],["allocator",3]],[["result",6],["string",3]]],[[],[["vec",3],["result",6]]],[[["allocator",3]],["result",6]],[[["usize",15]],[["result",6],["typeinfo",3]]],[[["allocator",3]],[["result",6],["string",3]]],[[],["u64",15]],[[["option",4],["runoptions",3]],["result",6]],[[["option",4],["runoptions",3]],["result",6]],[[["runoptions",3],["iobinding",3]],["result",6]],[[["option",4],["runoptions",3]],["result",6]],[[["option",4],["runoptions",3]],["result",6]],[[["option",4],["runoptions",3]],["result",6]],[[["option",4],["runoptions",3]],["result",6]],[[["option",4],["runoptions",3]],["result",6]],[[["i32",15]],["result",6]],[[]],[[["onnxtensorelementdatatype",4]]],[[["executionmode",4]]],[[]],[[["i32",15]]],[[["i32",15]]],[[["bool",15]]],[[["graphoptimizationlevel",4]]],[[["i32",15]]],[[["i32",15]]],[[["ortlanguageprojection",4]]],[[["str",15]],["result",6]],[[["i32",15]]],[[["i32",15]]],[[["i32",15]]],[[["i32",15]]],[[["asref",8],["path",3]],["result",6]],[[["str",15]],["result",6]],[[]],[[],[["option",4],["error",8]]],[[],[["vec",3],["result",6]]],[[],[["vec",3],["result",6]]],[[],[["result",6],["str",15]]],[[],["result",6]],[[],["result",6]],[[],[["tensortypeandshapeinfo",3],["result",6]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],["string",3]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],[["result",6],["option",4]]],[[]],[[],[["result",6],["onnxtype",4]]],[[],["i64",15]],null,null,null,null,null,null],"p":[[4,"OrtAllocatorType"],[4,"Error"],[4,"ONNXTensorElementDataType"],[4,"ONNXType"],[4,"GraphOptimizationLevel"],[4,"OrtErrorCode"],[4,"OrtLoggingLevel"],[4,"ExecutionMode"],[4,"OrtLanguageProjection"],[4,"OrtMemType"],[3,"SessionOptions"],[3,"Allocator"],[3,"MemoryInfo"],[3,"UnownedMemoryInfo"],[8,"AsONNXTensorElementDataType"],[3,"IoBinding"],[3,"Env"],[3,"RunOptions"],[3,"ModelMetadata"],[3,"Session"],[3,"TensorTypeAndShapeInfo"],[3,"UnownedTensorTypeAndShapeInfo"],[3,"TypeInfo"],[3,"Value"],[3,"ThreadingOptions"],[3,"ArenaCfg"],[13,"OrtError"],[13,"IoError"],[13,"NulError"],[13,"TryFromIntError"],[13,"Utf8Error"]]}\
}');
if (window.initSearch) {window.initSearch(searchIndex)};