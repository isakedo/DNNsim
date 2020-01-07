# TensorDash Example


## Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| n_lanes | uint32 | Number of concurrent multiplications per PE | Positive Number | 16 |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| n_tiles | uint32 | Number of tiles | Positive number | 16 |
| lookahead_h | uint32 | Lookahead value of H | Positive number | 2 |
| lookaside_d | uint32 | Lookaside value of D | Positive number | 5 |
| search_shape | uint32 | Search shape for the scheduler | 'T'-'L' | 'T' |
| banks | uint32 | Number of banks on-chip memory | Positive number | 16 |

Example batch files:

*   TensorDash_example: Performs TensorDash simulation and calculates potentials 
