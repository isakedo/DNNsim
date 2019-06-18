### BitFusion Example

H. Sharma, J. Park, N. Suda, L. Lai, B. Chau, V. Chandra, and H. Esmaeilzadeh
[Bit fusion: bit-level dynamically composable architecture for accelerating deep neural networks](https://dl.acm.org/citation.cfm?id=3306184)

### Input Parameters Description   

The following parameters are valid for this architecture:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| M | uint32 | Systolic array width (Parallel filters)| Positive number | 32 |
| N | uint32 | Systolic array height (Parallel windows)| Positive number | 16 |
| PMAX | uint32 | Maximum precision allowed per PE | Positive number | 8 |
| PMIN | uint32 | Minimum precision allowed per PE | Positive number | 2 |

Example batch files:

*   BitFusion_example: Performs BitFusion simulation and calculates potentials 