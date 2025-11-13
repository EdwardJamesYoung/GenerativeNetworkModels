# Fitting Generative Network Models

The likelihood that a Generative Network Model will produce a particular network is controlled by parameters $\eta$ and $\gamma$ which determine how the distance and affinity transforms, $d$ and $k$, are derived from the distance and affinity matrices, $D$ and $K$. As [discussed above](binary-gnms.md#the-binary-generative-network-model), it is not generally computationally tractable to compute the likelihood that any given parameter pair (and rule for computing affinities) will output a particular adjacency matrix. Instead, the parameters of the GNM are typically fit by generating many networks for each choice of parameters, and asking which choice of parameters produces networks that whose topology most accurately matches that seen in real brain networks. This section outlines the fitting procedure by which we determine which parameters give the best match to empirical data.

## Kolmogorov-Smirnov Measures of Topological Fit

The binary GNM aims to produce networks which have the same distributions of topological features as the networks found in real brains. Intuitively, this means that betweem the real and synthetic networks, there are an equal number of nodes that are highly or minimially clustered; an equal number of nodes that are highly or minimally connected; and an equal number of connections that are long or short. Importantly, topological similarity does not require that it is the same nodes in the real and synthetic network that are, for example, highly clustered. Two graphs may therefore have a very high topological similarity while the connection profile of each individual node looks remarkably different. Here, we make the notion of topological similarity we have just described rigorous with through the notion of a  distribution function. We will then discuss various different node-level features whose distributions we aim to match between real and synthetic networks.   

A [distribution](glossary.md#distribution) describes the collection of values taken by a particular network measure across all elements in a [network](glossary.md#network). For instance, the [degree](glossary.md#degree) [distribution](glossary.md#distribution) is the [degree](glossary.md#degree) of each [node](glossary.md#node) in the [network](glossary.md#network), whilst the [edge length](glossary.md#edge-length) [distribution](glossary.md#distribution) is the distance spanned by each [edge](glossary.md#edge) in the [network](glossary.md#network). One way of measuring topological similarity between networks is to measure the similarity between the distributions of various network measures between the networks.

To compare [distributions](glossary.md#distribution) between synthetic and empirical [networks](glossary.md#network), we use the Kolmogorov-Smirnov (KS) distance between those distributions. The KS distance begins with the [cumulative distribution functions](glossary.md#cumulative-distribution-function) for each of the two distributions. A cumulative distribution function at a given point is the fraction of values in the [distribution](glossary.md#distribution) that are less than or equal to that point. For example, if we have [degrees](glossary.md#degree) of 1, 3, 5, 7, and 9 across five [nodes](glossary.md#node), the CDF at the point 5 would be 0.6, since three out of five values (1, 3, and 5) are less than or equal to 5. From the CDF for the synthetic and empirical distribution over network properties, the KS distance can then be computed as the maximum difference between the CDFs. $$
\mathrm{KS} = \max_x | F_{\mathrm{empirical}}(x) - F_{\mathrm{synthetic}}(x) |
$$ The KS distance ranges from 0 (identical [distributions](glossary.md#distribution)) to 1 (completely different [distributions](glossary.md#distribution)).

For any two networks we wish to compare (*i.e.*, empirical and synthetic), we can obtain a KS distance for each network measure of interest. For example, we may compute the degree KS distance and the clustering coefficient KS distance. Note that these distances are in general different; networks can have a very similar distribution of degrees while having very different distributions of clustering coefficients. [Below](#statistics-for-binary-network-fitting), we give various network measures for which we typically compute the KS distance when attempting to fit the parameters of a binary GNM.

![Figure 4](figures-png/fig4.png)

<div class="figure-caption" id="fig-4">
<strong>Figure 4.</strong> Grid search parameter space showing the systematic exploration of η and γ values with corresponding energy landscapes. The colour scale represents the energy values, with darker regions indicating lower energy (better fit) parameter combinations.
</div>


### Statistics for Binary Network Fitting

There are four network statistics typically used to quantify fit between an empirical and synthetic network in the GNM: degree, clustering coefficient, betweenness centrality, and edge length.

The [degree](glossary.md#degree) of a [node](glossary.md#node) is equal to the total number of connections that [node](glossary.md#node) has, $s_i = \sum_j A_{ij}$. The [degree](glossary.md#degree) [distribution](glossary.md#distribution) across all [nodes](glossary.md#node) in a [network](glossary.md#network) can reveal whether there is roughly uniform level of connectivity between all nodes in the network, of if some nodes have much higher connectivity than others. See [Node-Level Measures](networks-and-graphs.md#node-level-measures) for a more detailed explanation of [degree](glossary.md#degree).

The [clustering coefficient](glossary.md#clustering-coefficient), $c_i$ of a [node](glossary.md#node) measures the extent to which its [neighbours](glossary.md#neighbour) are also connected to each other, quantifying local network clustering. It is equal to the fraction of possible connections between a node's neighbours that are actually present. See [Node-Level Measures](networks-and-graphs.md#node-level-measures) for a more detailed explanation of [clustering coefficients](glossary.md#clustering-coefficient).

[Betweenness centrality](glossary.md#betweenness-centrality) quantifies how often a [node](glossary.md#node) sits on the [shortest paths](glossary.md#shortest-path) between other [nodes](glossary.md#node) in the [network](glossary.md#network). The [betweenness centrality](glossary.md#betweenness-centrality) of a [node](glossary.md#node) $i$ is the fraction of pairs of other nodes in the network for which the [shortest paths](glossary.md#shortest-path) between those nodes passes through $i$. See [Node-Level Measures](networks-and-graphs.md#node-level-measures) for a more detailed explanation of [betweenness centrality](glossary.md#betweenness-centrality).

Finally, the [edge length](glossary.md#edge-length) of an [edge](glossary.md#edge) represents the physical distance between [nodes](glossary.md#node) connected by that [edge](glossary.md#edge). In particular, if $i$ and $j$ are connected (*i.e.*, $A_{ij} = 1$), then the edge length of their connection is $D_{ij}$ where $D$ is the distance matrix. The edge length [distribution](glossary.md#distribution) typically shows an overrepresentation of short connections compared to what would be expected from random wiring, reflecting the influence of spatial constraints on [network](glossary.md#network) formation. Note that edge length is the only feature included here which is not purely topological, since it depends on the position of the nodes in the network in physical space. 

### Statistics for Weighted Network Fitting

The above network measures can be adapted to the case where we wish to instead measure the difference between weighted networks. When working with weighted [networks](glossary.md#network), connection strengths are typically normalised before computing [network](glossary.md#network) statistics to ensure meaningful comparisons between [networks](glossary.md#network) with different overall weight scales. Normalisation is performed by dividing each individual weight by the maximum weight within the [network](glossary.md#network): $W'_{ij} = \frac{ W_{ij} }{ \max_{ab} W_{ab} }$. This ensures that all weights fall within the range \[0,1\] whilst preserving the relative relationships between connection strengths. Without this normalisation, [networks](glossary.md#network) with systematically higher weights would appear different even if their relative weight patterns were identical.

Weighted [node strength](glossary.md#node-strength-s) extends the concept of [degree](glossary.md#degree) to weighted [networks](glossary.md#network) by summing the weights of all connections attached to a [node](glossary.md#node) rather than simply counting them. It is computed analogously as $s_i = \sum_j W_{ij}$. This measure captures not only how many connections a [node](glossary.md#node) has, but also how strong those connections are.

Weighted [betweenness centrality](glossary.md#betweenness-centrality) adapts the concept of [betweenness centrality](glossary.md#betweenness-centrality) to weighted [networks](glossary.md#network) by considering the weights of [edges](glossary.md#edge) when computing [shortest paths](glossary.md#shortest-path). Rather than counting the number of [edges](glossary.md#edge) in a [path](glossary.md#path), weighted [betweenness centrality](glossary.md#betweenness-centrality) modifies the lengths of paths according to the weighting used along those paths.

Weighted [clustering coefficient](glossary.md#clustering-coefficient) generalises local clustering to weighted [networks](glossary.md#network) by incorporating the weights of connections within local neighbourhoods. This measure captures not only whether a [node's](glossary.md#node) [neighbours](glossary.md#neighbour) are connected to each other, but also how strongly they are connected.

## Correlational Measures of Topographical Fit

Except for edge length, the network statistics computed for the KS distances which comprise the energy are all purely topological, in the sense that they are agnostic to the identities of the nodes in the network. Because KS distance measures only the difference between distributions, two networks may have a very low degree KS distance while the nodes with a high degree in the first network may be the nodes with a low degree in the second network. Ideally, we would like our GNM to not only capture the topological properties of real brain networks, but also their *topographical* properties - in other words, the models should place particular features in the right place.

To capture topographical fit, we use the [correlation](glossary.md#correlation) between node-level properties between the two networks. In other words, if the property is large for a node in one network, it ought to be large for the other node as well. While the goal is to match nodes exactly, matching to nearby nodes is also good. As such, we typically apply spatial smoothing to the network properties in which the propery is averaged over nearby nodes. Specifically, for a node-level network property $X_i$, the unnormalised spacial smoothing weights $\tilde{\Omega}$ are given by $$ 
\tilde{\Omega}\_{ij} = \exp\left(-\frac{D\_{ij}^2}{2\sigma^2}\right).
$$ For each node $i$, a larger weight is assigned to those nodes $j$ which are closer to $i$ in space (*i.e.*, have a lower value of $D_{ij}$). The parameter $\sigma$ controls the spatial scale of the smoothing. These weights do not sum to $1$, so we normalise by performing $$
\Omega\_{ij} = \frac{\tilde{\Omega}\_{ij}}{\sum\_k \tilde{\Omega}\_{ik} }.
$$ The node-level property is then smoothed using these weights to obtain $\sum_j \Omega_{ij} X_j$. The correlation between these smoothed quantities across the two networks quantifies the degree of topographical fit.

The properties used for correlational fit measures are the same as those used for topological fit: degree, clustering, and betweenness centrality.

## Energy

Having computed both *topological* and *topographical* measures for the empirical and synthetic networks, we now define energy (an error score) that will guide parameter fitting. Energies are constructed so that **lower is better**: an energy of 0 indicates a perfect match for the included measures, whereas larger values indicate increasing discrepancy.

Depending on whether we are interested in generating synthetic connectomes which resembles real ones in terms of topology, topography, or both, we will change our energy function.

### Topological Energy 

Having computed a set of KS distances for a variety of network measures, we combine them into a single measure called the energy. For binary networks, the energy is defined as $$ E = \max\left( \text{KS}\_{\text{degree}}, \text{KS}\_{\text{clustering}}, \text{KS}\_{\text{betweenness}}, \text{KS}\_{\text{edge length}} \right), $$ where each KS term represents the [Kolmogorov-Smirnov distance](glossary.md#kolmogorov-smirnov-ks-distance) between the [distributions](glossary.md#distribution) of that property in the synthetic and empirical [networks](glossary.md#network). Lower energies indicate better agreement between the [network](glossary.md#network) [topology](glossary.md#topology) generated by the model and the real [network](glossary.md#network) [topology](glossary.md#topology).

For weighted [networks](glossary.md#network), energy is computed similarly but uses weighted versions of [network](glossary.md#network) properties and excludes [edge length](glossary.md#edge-length) (since this is a property of the spatial embedding rather than the weight structure):
$$
E\_{\text{weighted}} = \max\left( \text{KS}\_{\text{strength}}, \text{KS}\_{\text{weighted clustering}}, \text{KS}\_{\text{weighted betweenness}} \right).
$$
Taking the maximum enforces *multi-objective* fidelity: a model must match **all** included distributions to achieve low energy (poor fit on any one statistic dominates the score).

Implementation details for energy computation can be found in the [`gnm.evaluation`](../api-reference/evaluation.md) module, which provides comprehensive evaluation criteria for both binary and weighted [networks](glossary.md#network).

### Topographical Energy 

While we have seen how to compute correlational measures of topography, we now want to use them to compute the energy. Correlations are in the range $[-1, 1]$, but we want a final metric in the range $[0,1]$ (as with topological energy), where lower numbers indicate better fit. Thus, we map correlation to energy via a linear transform that treats $\rho=1$ as perfect alignment ($e=0$) and $\rho=-1$ as maximal misalignment ($e=1$): $$ e_X = 1 - \frac{\rho_X + 1}{2} = \frac{1 - \rho_X}{2}. $$

Then, we can aggregate across properties. The topographical energy is the maximum error across the three properties: $$ E_{\text{topography}} = \max \left(   e_{\text{degree}},   e_{\text{clustering}},   e_{\text{betweenness}} \right). $$

$E_{\text{topography}} = 0$ means all smoothed property maps are perfectly aligned across the empirical and synthetic networks; larger values indicate that at least one property is expressed in different anatomical territories.


### Combined Energy 

In many applications we want models that simultaneously reproduce *distributions* (**topology**) and *spatial layouts* (**topography**). Because $E_{\text{binary}}$, $E_{\text{weighted}}$, and $E_{\text{topography}}$ are all bounded in $[0,1]$, they can be mixed using a convex combination with user-specified weight $\beta$:

**Binary case** (if your synthetic and empirical networks are binary): $$ E_{\text{total}} = \beta\,E_{\text{binary}} + (1-\beta)\,E_{\text{topography}}. $$

**Weighted case** (if you are comparing weighted networks): $$ E_{\text{total}} = \beta\,E_{\text{weighted}} + (1-\beta)\,E_{\text{topography}}. $$

$\beta = 0.5$ gives equal weight to topological and topographical fit. Increasing $\beta$ emphasises matching the *distributions* of graph statistics; decreasing $\beta$ emphasises placing those statistics in the correct *anatomical locations*. You may fix $\beta$ a priori (common in grid searches) or treat it as an additional hyperparameter.

## Parameter Sweeps

Now that we have specified a method for specifying the level of agreement between empirical and model-generated networks, we can describe how to find the parameter set which reliably produces networks that have the best fit. This is done by searching over the parameter space to find configurations that have the best average fit between the empirical network and the model-generated networks. This search encompasses not only continuous parameters like [$\eta$](glossary.md#eta-eta) and [$\gamma$](glossary.md#gamma-gamma), but also discrete choices such as which [generative rule](glossary.md#generative-rule) to use for computing the [affinity matrix](glossary.md#affinity-matrix-k) or which [loss](glossary.md#loss) function to optimise for weighted GNMs.

The stochastic nature of GNMs introduces additional complexity into parameter fitting. The same parameter configuration will typically produce different [networks](glossary.md#network) across multiple simulation runs due to the probabilistic sampling of connections. To account for this variability, it is necessary to generate many [networks](glossary.md#network) from each parameter set and aggregate the fit measures across these simulations. Common aggregation methods include computing the mean energy across simulations, which provides a stable estimate of expected performance, and quantile aggregation, which can focus on typical or best-case performance. The median represents a special case of quantile aggregation at the 50th percentile, offering robustness to outlier simulations with unusually poor fit or unusually good fit.

Grid searches involve systematically sweeping over the parameter space by evaluating the model at regularly spaced parameter combinations. The coarseness or fineness of the grid determines the resolution of the search, with finer grids providing more thorough exploration at the cost of increased computational requirements. Grid searches are particularly suitable when the parameter space is low-dimensional and computational resources allow for exhaustive evaluation.

![Figure 5](figures-png/fig5.png)

<div id="grid-search-algorithm" class="algorithm-anchor">
  <div class="algorithm-box">
    <div class="algorithm-banner">Algorithm: Grid Search for Parameter Fitting</div>
    
    <div class="algorithm-content">
      <p><strong>Input</strong>: Parameter ranges, grid resolution, number of simulations per parameter set</p>
      
      <p><strong>For</strong> each parameter combination in the grid <strong>do</strong>:</p>
      <ol>
        <li><strong>For</strong> each run in the batch <strong>do</strong>:
          <ul>
            <li>Generate new synthetic network using current parameters.</li>
            <li>Compute energy between synthetic and empirical network.</li>
          </ul>
        </li>
        <li>Aggregate energy across all simulations in the batch (using, e.g., the mean or median energy for the batch).</li>
        <li>Store aggregated energy for current parameter combination,</li>
      </ol>
      <p><strong>End for</strong></p>
      
      <p><strong>Return</strong>: Parameter combination with lowest aggregated energy.</p>
    </div>
  </div>
</div>


Bayesian optimisation methods provide a more sophisticated method to search parameter space in which an estimate of the dependence of the energy on parameter values is formed based on previously observed energy values. This model can then guide the search toward promising regions of parameter space, increasing the efficiency of the search process over the exhausive enumeration used by grid search. These methods are particularly valuable when the search space becomes very large, as they can achieve good results with fewer parameter evaluations than grid searches. The downside is that they are non-exhausive, and as such there is always a possibility that the optimal parameter combination was not found by the model. For comprehensive guidance on Bayesian optimisation techniques and implementations, see [relevant Weights & Biases documentation](placeholder-link).

Implementation details for parameter sweeps can be found in the [`gnm.fitting`](../api-reference/fitting.md) module, which provides tools for both grid search and Bayesian optimisation approaches to model fitting.