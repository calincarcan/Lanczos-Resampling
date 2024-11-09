# Lanczos Resampling
 
## Overview

This project implements the Lanczos resampling algorithm in C. The Lanczos resampling algorithm is used for image scaling and reconstruction, providing high-quality results.

## Requirements

- `cmath` library
- `libpng` library

## Project Structure

- `src/`: Contains the source code for the project.
- `bin/`: Contains the compiled executables.
- `lanczos.c`: The main implementation of the Lanczos resampling algorithm.
- `Makefile`: Contains the build rules for the project.

## Build Instructions

To build the project, navigate to the project directory and run:

```sh
make all
```

This will compile the source code and generate the executables in the `bin` directory.

## Usage

To run the Lanczos resampling executable, use the following command:

```sh
./bin/lanczos
```

Or using the makefile run:

```sh
make run
```

## Cleaning Up

To remove the compiled executables and object files, run:

```sh
make clean
```

This will clean the `bin` directory and remove any intermediate build files.