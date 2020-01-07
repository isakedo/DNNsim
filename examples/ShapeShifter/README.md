# ShapeShifter Example

A. Delmas Lascorz, S. Sarify, I. Edo, D. Malone Stuart, O. Mohamed Awad, P. Judd, M. Mahmoud, M. Nikolic, K. Siu, Z. Poulos, A. Moshovos 
[ShapeShifter: Enabling Fine-Grain Data Width Adaptation in Deep Learning](https://dl.acm.org/doi/10.1145/3352460.3358295)

M. Mahmoud, K. Siu, and A. Moshovos 
[Diffy: a déjà vu-free differential deep neural network accelerator](https://dl.acm.org/doi/10.1109/MICRO.2018.00020)

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
| group_size | uint32 | Number of columns per group | Positive number | 1 |
| column_register | uint32 | Number of registers per SIP | Positive number | 0 |
| minor_bit | bool | Calculate also the minor bit for dynamic precisions | True-False | false |
| diffy | bool | Add Differential Convolution on top | True-False | false |
| tactical | bool | Add BitTactical zero weight skipping on top | True-False | false |
| lookahead_h | uint32 | Lookahead value of H | Positive number | 2 |
| lookaside_d | uint32 | Lookaside value of D | Positive number | 5 |
| search_shape | uint32 | Search shape for the scheduler | 'T'-'L' | 'T' |

Example batch files:

*   ShapeShifter_example: Performs ShapeShifter simulation and calculates potentials 
*   ShapeShifterDiffy_example: Performs ShapeShifter Diffy simulation 
*   BitTacticalP_example: Performs BitTacticalP simulation and calculates potentials 