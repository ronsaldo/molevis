# Mollevis - Mollecular simulator and visualizer

## Building

For building on Linux the Vulkan header files and the SDL2 development must be installed. This project must be cloned with the submodules in order to build the bundled version of abstract-gpu. A similar procedure is needed for building on Windows and Mac, where the SDL2 library must be specified. 

For recursive cloning with submodules, the following script can be used:

```bash
git clone --recursive https://github.com/ronsaldo/mollevis
```

The following bash script can be used for building on Linux:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Running

The building process produces the following build artifacts:

- *dist/Mollevis* The mollecular visualization and simulation spplication

Currently only randomly generated atoms and bonds are supported. The number of generated atoms can be specified by using *-gen-atoms N*, and the number of generated bonds is specified with *-gen-bonds N*
