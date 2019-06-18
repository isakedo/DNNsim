### SCNN Example

A. Parashar, M. Rhu, A. Mukkara, A. Puglielli, R. Venkatesan, B. Khailany, J. Emer, S. W. Keckler, and W. J. Dally
[SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks](https://dl.acm.org/citation.cfm?id=3080254)

### Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| Wt | uint32 | Number of PE columns | Positive number | 8 |
| Ht | uint32 | Number of PE rows | Positive number | 8 |
| I | uint32 | Column multipliers per PE | Positive number | 4 |
| F | uint32 | Number of PE columns | Positive number | 4 |
| out_acc_size | uint32 | Size of the output accumulator per PE | Positive number | 1024 |
| banks | uint32 | Number of banks in the output accumulator per PE | Positive number | 32 |

Example batch files:

*   SCNN_example: Performs SCNN simulation and calculates potentials 