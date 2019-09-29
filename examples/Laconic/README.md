### Laconic Example

S. Sharify, M. Mahmoud, A. Delmas Lascorz, M. Nikolic, A. Moshovos 
[Laconic Deep Learning Computing](https://dl.acm.org/citation.cfm?id=3322255)

### Default Parameters Description   

Default for the architecture. This parameters are defined in core/include/core/Laconic.h

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| BOOTH_ENCODING | bool | Activate booth encoding | True-False | True |
| ZERO_COUNT | bool | Zero values count as one cycle | True-False | True | 
   
### Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| n_lanes | uint32 | Number of concurrent multiplications per PE | Positive Number | 16 |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| n_tiles | uint32 | Number of tiles | Positive number | 16 |

Example batch files:

*   Laconic_example: Performs Laconic simulation and calculates potentials 