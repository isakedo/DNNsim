### Stripes Example

P. Judd, J. Albericio, T. Hetherington, T. Aamodt, A. Moshovos, 
[Stripes: Bit-Serial Deep Learning Computing, ACM/IEEE International Conference on Microarchitecture, Oct 2016](https://www.ece.ubc.ca/~aamodt/publications/papers/stripes-final.pdf)

### Default Parameters Description   

Default for the architecture. This parameters are defined in core/include/core/Stripes.h

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| FC_MULTIPLEX_COLUMNS | bool | Fully connected layers are time-multiplexed across the columns | True-False | True |
   
### Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| weight_lanes | uint32 | Number of concurrent weights per PE | Positive Number | 16 |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| bits_pe | uint32 | Number of bits per PE | Positive number | 16 |

Example batch files in this folder are the following:

