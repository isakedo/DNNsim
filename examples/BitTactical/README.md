### Bit-Tactical Example

A. Delmas Lascorz, P. Judd, D. Malone Stuart, Z. Poulos, M. Mahmoud, S. Sharify, M. Nikolic, K. Siu, and A. Moshovos
[Bit-Tactical: A Software/Hardware Approach to Exploiting Value and Bit Sparsity in Neural Networks](https://dl.acm.org/citation.cfm?id=3304041)

Bit-Tactical frontend can be executed using two different Bit-Serial backends: BitTacticalE (Based on BitPragmatic), 
and BitTacticalP (Based on DynamicStripes)  


### Default Parameters Description   

Default for the architecture. This parameters are defined in core/include/core/BitTactical\[E|0\].h

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| BOOTH_ENCODING (Only TCLe) | bool | Activate booth encoding | True-False | True |
| ZERO_COUNT | bool | Zero values count as one cycle | True-False | True | 
| FC_MULTIPLEX_COLUMNS | bool | Fully connected layers are time-multiplexed across the columns | True-False | True |
   
### Input Parameters Description    

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| n_lanes | uint32 | Number of concurrent multiplications per PE | Positive Number | 16 |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| column_registers | uint32 | Number of registers per column to run-ahead | Positive number | 0 |
| precision_granularity (Only TCLp) | uint32 | Size of the group of values | Positive number | 16 |
| leading bit (Only TCLp)| bool | Only the leading bit for dynamic precisions | True-False | False |
| bits_first_stage (Only TCLe)| uint32 | Number of bits of the first stage shifter | Positive number | 0 |
| lookahead_h |uint32 | Lookahead window size | Positive number | 2 |
| lookaside_d |uint32 | Lookaside window size | Positive number | 5 | 
| search_shape | string | Shape of the scheduler search | L-T | L |
| read_schedule | bool | Read the scheduled weights from a Protobuf file | True-False | False |


Example batch files:

*   Schedule_example: Schedules weights statically, and performs simulation for BitTacticalE reading previous schedule
*   BitTacticalE: Performs BitTacticalE schedule/simulation, and calculates potentials 
*   BitTacticalP: Performs BitTacticalP schedule/simulation, and calculates potentials 