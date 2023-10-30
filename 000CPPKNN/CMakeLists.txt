add_executable(000CPPKNN
	src/main.cpp
)


target_include_directories(000CPPKNN PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})