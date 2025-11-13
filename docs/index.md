# Generative Network Model Toolbox

## Description
Generative network models have been used extensively in research in previous years to explore microstructural and functional connectomics. Binary generative models (BGM), though powerful, are unable to reflect to extent to which nodes in a network are connected. Weighted generative models (WGM) were introduced to resolve this limitation. The tools provided here, grounded in graph theory, provide a computationally efficient and intuative method of implementing WGMs.

The aim of this project was to make GNMs more accessable to the wider research community and beyond. Optimized in Python, this code is more efficient than previous implementations of both binary and weighted network tools such as the Brain Connectivity Toolbox [1] and the supplimentary code to a recently published WGM paper [2].

## Table of Contents
- [Getting Started](getting-started.md)
- [Understanding Generative Network Models](understanding-gnms/index.md)
- [User Guide](user-guide/index.md)
- [API Reference](api-reference/index.md)
    - [model](api-reference/model.md)
    - [gnm.generative_rules](api-reference/generative-rules.md)
    - [gnm.weight_criteria](api-reference/weight-criteria.md)
    - [gnm.fitting](api-reference/fitting.md)
    - [gnm.defaults](api-reference/defaults.md)
    - [gnm.utils](api-reference/utils.md)

## Authors and Acknowledgements
- [Edward James Young](https://github.com/EdwardJamesYoung) (ey245@cam.ac.uk)
- [William Mills](https://github.com/wbmills) (william.mills@mrc-cbu.cam.ac.uk)
- [Francesco Poli](https://github.com/FrancescPoli) (francesco.poli@mrc-cbu.cam.ac.uk)


## References 
1. [Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: Uses and interpretations. NeuroImage, 52(3), 1059–1069.](https://doi.org/10.1016%2Fj.neuroimage.2009.10.003)
2. [Akarca, D., Schiavi, S., Achterberg, J., Genc, S., Jones, D. K., & Astle, D. E. (2023). A weighted generative model of the human connectome. Neuroscience.](https://doi.org/10.1101/2023.06.23.546237)