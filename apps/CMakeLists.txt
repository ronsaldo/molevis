set(Molevis_Sources
    Frustum.cpp
    Simulation.cu
    Molevis.cpp
)

add_executable(Molevis ${Molevis_Sources})
target_link_libraries(Molevis chemfiles Agpu ${SDL2_LIBRARY})
