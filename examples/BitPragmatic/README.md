### BitPragmatic Example

J. Albericio, A. Delmas Lascorz, P. Judd, S. Sharify, G. O'Leary, R. Genov, and A. Moshovos, 
[Bit-Pragmatic Deep Learning Computing, IEEE/ACM MICRO 2017](https://dl.acm.org/citation.cfm?id=3123982)

### Default Parameters Description   

Default for the architecture. This parameters are defined in core/include/core/BitPragmatic.h

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| BOOTH_ENCODING | bool | Activate booth encoding | True-False | True |
| ZERO_COUNT | bool | Zero values count as one cycle | True-False | True | 
| FC_MULTIPLEX_COLUMNS | bool | Fully connected layers are time-multiplexed across the columns | True-False | True |
   
### Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| weight_lanes | uint32 | Number of concurrent weights per PE | Positive Number | 16 |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| column_registers | uint32 | Number of registers per column to run-ahead | Positive number | 0 |
| bits_first_stage | uint32 | Number of bits of the first stage shifter | Positive number | 0 |
| diffy | bool | Simulate Diffy in top of the architecture | True-False | False |

Example batch files in this folder are the following:

