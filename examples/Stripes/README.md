# Stripes Example

P. Judd, J. Albericio, T. Hetherington, T. Aamodt, A. Moshovos, 
[Stripes: Bit-Serial Deep Learning Computing, ACM/IEEE International Conference on Microarchitecture, Oct 2016](https://www.ece.ubc.ca/~aamodt/publications/papers/stripes-final.pdf)

## Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| n_lanes | uint32 | Number of concurrent multiplications per PE | Positive Number | 16 |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| n_tiles | uint32 | Number of tiles | Positive number | 16 |
| bits_pe | uint32 | Number of bits per PE | Positive number | 16 |

Example batch files:

*   Stripes_example: Performs Stripes simulation and calculates potentials 