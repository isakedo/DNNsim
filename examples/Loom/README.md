### Loom Example

S. Sharify, A. Delmas, P. Judd, K. Siu, and A. Moshovos
[Loom: exploiting weight and activation precisions to accelerate convolutional neural networks](https://dl.acm.org/citation.cfm?id=3196072)

### Default Parameters Description   

Default for the architecture. This parameters are defined in core/include/core/Loom.h

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| ZERO_COUNT | bool | Zero values count as one cycle | True-False | True | 
| FC_MULTIPLEX_COLUMNS | bool | Fully connected layers are time-multiplexed across the columns | True-False | True |
   
### Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| n_lanes | uint32 | Number of concurrent multiplications per PE | Positive Number | 16 |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| precision_granularity | uint32 | Size of the group of values | Positive number | 16 |
| leading bit | bool | Only the leading bit for dynamic precisions | True-False | False |
| pe_serial_bits | uint32 | Number of serial bits per PE | Positive Number | 1 |
| dynamic_weights | bool | Use dynamic precision for the weights | True-False | False |

Example batch files:

*   Loom_example: Performs Loom simulation and calculates potentials 