### DynamicStripes Example

A. Delmas, P. Judd, S. Sharify, A. Moshovos, 
[Dynamic Stripes: Exploiting the Dynamic Precision Requirements of Activation Values in Neural Networks, arxiv](https://arxiv.org/abs/1706.00504)

### Default Parameters Description   

Default for the architecture. This parameters are defined in core/include/core/DynamicStripes.h

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| FC_MULTIPLEX_COLUMNS | bool | Fully connected layers are time-multiplexed across the columns | True-False | True |
   
### Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| n_lanes | uint32 | Number of concurrent multiplications per PE | Positive Number | 16 |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| column_registers | uint32 | Number of registers per column to run-ahead | Positive number | 0 |
| precision_granularity | uint32 | Size of the group of values | Positive number | 16 |
| leading bit | bool | Only the leading bit for dynamic precisions | True-False | False |
| bits_pe | uint32 | Number of bits per PE | Positive number | 16 |
| diffy | bool | Simulate Diffy in top of the architecture | True-False | False |

Example batch files:

*   DynamicStripes_example: Performs DynamicStripes simulation and calculates potentials 