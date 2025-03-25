set(Mollevis_Sources
    Mollevis.cpp
)

add_executable(Mollevis ${Mollevis_Sources})
target_link_libraries(Mollevis chemfiles Agpu ${SDL2_LIBRARY})