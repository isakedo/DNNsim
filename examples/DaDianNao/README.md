# DaDianNao Example

Y. Chen, T. Luo, S. Liu, S. Zhang, J. Wang, L. Li, T. Chen, Z. Xu, N. Sun, and O. Tenman
[DaDianNao: A Machine-Learning Supercomputer](https://ieeexplore.ieee.org/document/7011421)

A. Delmas Lascorz, P. Judd, D. Malone Stuart, Z. Poulos, M. Mahmoud, S. Sharify, M. Nikolic, K. Siu, and A. Moshovos
[Bit-Tactical: A Software/Hardware Approach to Exploiting Value and Bit Sparsity in Neural Networks](https://dl.acm.org/citation.cfm?id=3304041)

## Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| n_lanes | uint32 | Number of concurrent multiplications per PE | Positive Number | 16 |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| n_tiles | uint32 | Number of tiles | Positive number | 16 |
| bits_pe | uint32 | Number of bits per PE | Positive number | 16 |
| tactical | bool | Add BitTactical zero weight skipping on top | True-False | false |
| lookahead_h | uint32 | Lookahead value of H | Positive number | 2 |
| lookaside_d | uint32 | Lookaside value of D | Positive number | 5 |
| search_shape | uint32 | Search shape for the scheduler | 'T'-'L' | 'T' |

Example batch files:

*   DaDianNao_example: Performs DaDianNao simulation and calculates potentials 
*   BitTactical_example: Performs BitTactical simulation and calculates potentials