# BitPragmatic Example

J. Albericio, A. Delmas Lascorz, P. Judd, S. Sharify, G. O'Leary, R. Genov, and A. Moshovos, 
[Bit-Pragmatic Deep Learning Computing, IEEE/ACM MICRO 2017](https://dl.acm.org/citation.cfm?id=3123982)

M. Mahmoud, K. Siu, and A. Moshovos 
[Diffy: a déjà vu-free differential deep neural network accelerator](https://dl.acm.org/doi/10.1109/MICRO.2018.00020)

A. Delmas Lascorz, P. Judd, D. Malone Stuart, Z. Poulos, M. Mahmoud, S. Sharify, M. Nikolic, K. Siu, and A. Moshovos
[Bit-Tactical: A Software/Hardware Approach to Exploiting Value and Bit Sparsity in Neural Networks](https://dl.acm.org/citation.cfm?id=3304041)


## Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| lanes | uint32 | Number of concurrent multiplications per PE | Positive Number | 16 |
| columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| tiles | uint32 | Number of tiles | Positive number | 16 |
| pe_width | uint32 | PE input bit-width | Positive number | 16 |
| bits_first_stage | uint32 | Number of bits of the first stage shifter | Positive number | 0 |
| column_register | uint32 | Number of registers per SIP | Positive number | 0 |
| booth_encoding | bool | Add booth encoding on top | True-False | false |
| diffy | bool | Add Differential Convolution on top | True-False | false |
| tactical | bool | Add BitTactical zero weight skipping on top | True-False | false |
| lookahead_h | uint32 | Lookahead value of H | Positive number | 2 |
| lookaside_d | uint32 | Lookaside value of D | Positive number | 5 |
| search_shape | uint32 | Search shape for the scheduler | 'T'-'L' | 'T' |

Example batch files:

*   BitPragmatic_example: Performs BitPragmatic simulation and calculates potentials 
*   BitPragmaticDiffy_example: Performs BitPragmatic Diffy simulation 
*   BitTacticalE_example: Performs BitTacticalE simulation and calculates potentials 
