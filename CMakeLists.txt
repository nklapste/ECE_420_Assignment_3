cmake_minimum_required(VERSION 3.10)
project(ECE_420_Assignment_3)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(q5a q5a.c )
target_link_libraries(q5a ${MPI_C_LIBRARIES})
add_executable(q5b q5b.c)
target_link_libraries(q5b ${MPI_C_LIBRARIES})
add_executable(q5c q5c.c)
target_link_libraries(q5c ${MPI_C_LIBRARIES})
target_link_libraries(q5c m)
add_executable(q6 q6.c)
target_link_libraries(q6 ${MPI_C_LIBRARIES})
