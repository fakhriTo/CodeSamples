add_executable(000CudaKNN
	src/main.cu
	src/nn.hpp
)

target_include_directories(000CudaKNN PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})