# Loom Example

S. Sharify, A. Delmas, P. Judd, K. Siu, and A. Moshovos
[Loom: exploiting weight and activation precisions to accelerate convolutional neural networks](https://dl.acm.org/citation.cfm?id=3196072)

## Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| n_lanes | uint32 | Number of concurrent multiplications per PE | Positive Number | 16 |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| n_tiles | uint32 | Number of tiles | Positive number | 16 |
| bits_pe | uint32 | Number of bits per PE | Positive number | 16 |
| group_size | uint32 | Number of columns/rows per group | Positive number | 1 |
| pe_serial_bits | uint32 | Number of serial activations bits per PE | Positive Number | 1 |
| minor_bit | bool | Calculate also the minor bit for dynamic precisions | True-False | false |
| dynamic_weights | bool | Use dynamic precision for the weights | True-False | False |

Example batch files:

*   Loom_example: Performs Loom simulation and calculates potentials 
