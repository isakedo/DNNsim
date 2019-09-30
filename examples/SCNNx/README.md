# SCNNx Example

## Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| Wt | uint32 | Number of PE columns | Positive number | 8 |
| Ht | uint32 | Number of PE rows | Positive number | 8 |
| I | uint32 | Column multipliers per PE | Positive number | 4 |
| F | uint32 | Number of PE columns | Positive number | 4 |
| out_acc_size | uint32 | Size of the output accumulator per PE | Positive number | 1024 |
| banks | uint32 | Number of banks in the output accumulator per PE | Positive number | 32 |
| pe_serial_bits | uint32 | Number of serial activations bits per PE | Positive Number | 1 |

Example batch files:

*   SCNNe_example: Performs SCNNe simulation and calculates potentials 
*   SCNNp_example: Performs SCNNe simulation and calculates potentials 