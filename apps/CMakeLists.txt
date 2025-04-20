set(Molevis_Sources
    Molevis.cpp
)

add_executable(Molevis ${Molevis_Sources})
target_link_libraries(Molevis chemfiles Agpu ${SDL2_LIBRARY})
