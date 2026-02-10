cmake_minimum_required(VERSION 3.20)

project(uvcc_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CURL REQUIRED)

option(UVCC_WITH_CUDA_NCCL "Enable CUDA + NCCL support for intra-party collectives" OFF)

add_library(
  uvcc_native
  uvcc/audit.cpp
  uvcc/arena.cpp
  uvcc/base64.cpp
  uvcc/bytes.cpp
  uvcc/blv1.cpp
  uvcc/clock.cpp
  uvcc/config.cpp
  uvcc/crc32c.cpp
  uvcc/dp.cpp
  uvcc/fss.cpp
  uvcc/frame.cpp
  uvcc/gemm_beaver.cpp
  uvcc/harness.cpp
  uvcc/ids.cpp
  uvcc/keys.cpp
  uvcc/lift.cpp
  uvcc/log.cpp
  uvcc/nccl.cpp
  uvcc/net.cpp
  uvcc/open.cpp
  uvcc/optim.cpp
  uvcc/ops.cpp
  uvcc/ops_exec.cpp
  uvcc/pp.cpp
  uvcc/reassembly.cpp
  uvcc/relay_http.cpp
  uvcc/relay_rawconn.cpp
  uvcc/runtime.cpp
  uvcc/sgir.cpp
  uvcc/sha256.cpp
  uvcc/slots.cpp
  uvcc/sks.cpp
  uvcc/step.cpp
  uvcc/subsession.cpp
  uvcc/tcf.cpp
  uvcc/tp.cpp
  uvcc/transcript_hooks.cpp
  uvcc/transport.cpp
  uvcc/testkit.cpp
  uvcc/topology.cpp
  uvcc/transcript.cpp
  uvcc/wvole.cpp
)

target_include_directories(uvcc_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(uvcc_native PUBLIC CURL::libcurl)

if(UVCC_WITH_CUDA_NCCL)
  # Do not require CMake's CUDA language; we only need headers + dynamic libs for cudart + nccl.
  find_path(UVCC_CUDA_INCLUDE_DIR cuda_runtime_api.h
    PATHS /usr/local/cuda/include /usr/include
  )
  find_library(UVCC_CUDART_LIB cudart
    PATHS /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
  )
  find_path(UVCC_NCCL_INCLUDE_DIR nccl.h
    PATHS /usr/include /usr/local/cuda/include
  )
  find_library(UVCC_NCCL_LIB nccl
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
  )
  if(NOT UVCC_CUDA_INCLUDE_DIR OR NOT UVCC_CUDART_LIB OR NOT UVCC_NCCL_INCLUDE_DIR OR NOT UVCC_NCCL_LIB)
    message(FATAL_ERROR "UVCC_WITH_CUDA_NCCL=ON but CUDA/NCCL headers/libs not found. cuda_include=${UVCC_CUDA_INCLUDE_DIR} cudart=${UVCC_CUDART_LIB} nccl_include=${UVCC_NCCL_INCLUDE_DIR} nccl=${UVCC_NCCL_LIB}")
  endif()
  target_compile_definitions(uvcc_native PUBLIC UVCC_WITH_CUDA_NCCL=1)
  target_include_directories(uvcc_native PUBLIC ${UVCC_CUDA_INCLUDE_DIR} ${UVCC_NCCL_INCLUDE_DIR})
  target_link_libraries(uvcc_native PUBLIC ${UVCC_CUDART_LIB} ${UVCC_NCCL_LIB})
endif()

add_executable(uvcc_worker tools/uvcc_worker_main.cpp)
target_link_libraries(uvcc_worker PRIVATE uvcc_native)

enable_testing()

add_executable(test_sha256 tests/test_sha256.cpp)
target_link_libraries(test_sha256 PRIVATE uvcc_native)
add_test(NAME test_sha256 COMMAND test_sha256)

add_executable(test_ids_msgid tests/test_ids_msgid.cpp)
target_link_libraries(test_ids_msgid PRIVATE uvcc_native)
add_test(NAME test_ids_msgid COMMAND test_ids_msgid)

add_executable(test_ids_sid tests/test_ids_sid.cpp)
target_link_libraries(test_ids_sid PRIVATE uvcc_native)
add_test(NAME test_ids_sid COMMAND test_ids_sid)

add_executable(test_ids_msgid_v1 tests/test_ids_msgid_v1.cpp)
target_link_libraries(test_ids_msgid_v1 PRIVATE uvcc_native)
add_test(NAME test_ids_msgid_v1 COMMAND test_ids_msgid_v1)

add_executable(test_ids_opid_fssid tests/test_ids_opid_fssid.cpp)
target_link_libraries(test_ids_opid_fssid PRIVATE uvcc_native)
add_test(NAME test_ids_opid_fssid COMMAND test_ids_opid_fssid)

add_executable(test_blv1 tests/test_blv1.cpp)
target_link_libraries(test_blv1 PRIVATE uvcc_native)
add_test(NAME test_blv1 COMMAND test_blv1)

add_executable(test_frame_v1 tests/test_frame_v1.cpp)
target_link_libraries(test_frame_v1 PRIVATE uvcc_native)
add_test(NAME test_frame_v1 COMMAND test_frame_v1)

add_executable(test_transport_v1 tests/test_transport_v1.cpp)
target_link_libraries(test_transport_v1 PRIVATE uvcc_native)
add_test(NAME test_transport_v1 COMMAND test_transport_v1)

add_executable(test_lift_tlv tests/test_lift_tlv.cpp)
target_link_libraries(test_lift_tlv PRIVATE uvcc_native)
add_test(NAME test_lift_tlv COMMAND test_lift_tlv)

add_executable(test_transcript_v1 tests/test_transcript_v1.cpp)
target_link_libraries(test_transcript_v1 PRIVATE uvcc_native)
add_test(NAME test_transcript_v1 COMMAND test_transcript_v1)

add_executable(test_slots_ops tests/test_slots_ops.cpp)
target_link_libraries(test_slots_ops PRIVATE uvcc_native)
add_test(NAME test_slots_ops COMMAND test_slots_ops)

add_executable(test_open_arith_v1 tests/test_open_arith_v1.cpp)
target_link_libraries(test_open_arith_v1 PRIVATE uvcc_native)
add_test(NAME test_open_arith_v1 COMMAND test_open_arith_v1)

add_executable(test_gemm_beaver_v1 tests/test_gemm_beaver_v1.cpp)
target_link_libraries(test_gemm_beaver_v1 PRIVATE uvcc_native)
add_test(NAME test_gemm_beaver_v1 COMMAND test_gemm_beaver_v1)

add_executable(test_nccl_groups tests/test_nccl_groups.cpp)
target_link_libraries(test_nccl_groups PRIVATE uvcc_native)
add_test(NAME test_nccl_groups COMMAND test_nccl_groups)

add_executable(test_pp_schedule tests/test_pp_schedule.cpp)
target_link_libraries(test_pp_schedule PRIVATE uvcc_native)
add_test(NAME test_pp_schedule COMMAND test_pp_schedule)

add_executable(test_dp_optim tests/test_dp_optim.cpp)
target_link_libraries(test_dp_optim PRIVATE uvcc_native)
add_test(NAME test_dp_optim COMMAND test_dp_optim)

add_executable(test_phase6_determinism tests/test_phase6_determinism.cpp)
target_link_libraries(test_phase6_determinism PRIVATE uvcc_native)
add_test(NAME test_phase6_determinism COMMAND test_phase6_determinism)

add_executable(test_phase6_matrix tests/test_phase6_matrix.cpp)
target_link_libraries(test_phase6_matrix PRIVATE uvcc_native)
add_test(NAME test_phase6_matrix COMMAND test_phase6_matrix)

add_executable(test_tcf_v0a tests/test_tcf_v0a.cpp)
target_link_libraries(test_tcf_v0a PRIVATE uvcc_native)
add_test(NAME test_tcf_v0a COMMAND test_tcf_v0a)

add_executable(test_wvole_stub tests/test_wvole_stub.cpp)
target_link_libraries(test_wvole_stub PRIVATE uvcc_native)
add_test(NAME test_wvole_stub COMMAND test_wvole_stub)


