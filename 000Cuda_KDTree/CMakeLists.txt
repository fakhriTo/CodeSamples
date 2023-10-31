add_executable(000Cuda_KDTree
	src/main.cu
	src/kernel.cu
	src/kdtree.h
)

target_include_directories(000Cuda_KDTree PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})