# Laconic Example

S. Sharify, M. Mahmoud, A. Delmas Lascorz, M. Nikolic, A. Moshovos 
[Laconic Deep Learning Computing](https://dl.acm.org/citation.cfm?id=3322255)
   
## Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| lanes | uint32 | Number of concurrent multiplications per PE | Positive Number | 16 |
| columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| tiles | uint32 | Number of tiles | Positive number | 16 |
| pe_width | uint32 | PE input bit-width | Positive number | 16 |
| booth_encoding | bool | Add booth encoding on top | True-False | false |

Example batch files:

*   Laconic_example: Performs Laconic simulation and calculates potentials 