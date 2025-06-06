set(Molevis_Sources
    Frustum.cpp
    Simulation.cu
    Molevis.cpp
)

if(MOLEVIS_USE_CUDA)
set(Molevis_Sources ${Molevis_Sources}
    Simulation.cu
)
endif()

add_executable(Molevis ${Molevis_Sources})
target_link_libraries(Molevis chemfiles Agpu ${SDL2_LIBRARY})
