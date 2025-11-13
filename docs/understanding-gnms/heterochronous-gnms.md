# Heterochronous Generative Network Models

## Motivation: why account for developmental time?

Classic [binary generative network models](binary-gnms.md) (GNMs) grow synthetic connectomes using probabilities computed from a topological affinity term, and a spacial cost term. These models are able to reproduce hallmark *topological* properties of empirical connectomes. However, they routinely misplace topologicla features in physical space—for example, hubs appear in anatomically implausible locations and modules drift away from canonical lobes. To remedy this, the heterochronous GNM introduces a key biological feature missing from the classical formulation: **heterochronicity**. Heterochronicity captures the fact that different cortical and sub‑cortical areas begin wiring at different times. Incorporating this third, time‑dependent term into the wiring probabilities allows GNMs to respect this developmental programme. Heterochronous GNMs therefore provide a principled bridge between biological mechanism and computational model.

## The Heterochronous Generative Network Model

The heterochronous generative network model modifies the standard [binary GNM](binary-gnms.md#binary-generative-network-models) algorithm by modifying how wiring probabilities are computed at each iteration of the algorithm. Recall that the standard binary GNM computes (unnormalised) wiring probabilities as the product of a distance transform [distance transform](glossary.md#distance-transform-d), $d_{ij}$, derived from the Euclidean distance matrix $D$, and controlled by parameter $\eta$, and the [affinity transform](glossary.md#affinity-transform-k), $k_{ij}$, derived from the affinity matrix $K$, and controlled by parameter $\gamma$. The heterochronous GNM includes an additional term in this product, known as the **heterochronicity transform**, $h_{ij}(t)$, which is similarly derived from a time-varying heterochronicity matrix $H(t)$, and controlled by a parameter $\lambda$. Thus, at iteration $t$ of the algorithm, we compute the unnormalised wiring probabilities as

$$
    \tilde{P}_{ij} = d_{ij} \times k_{ij} \times h_{ij}(t) 
$$

From this point, the algorithm proceeds identically to the binary GNM. The updated algorithm is given below. 

<div id="heterochronous-gnm-algorithm" class="algorithm-anchor">
  <div class="algorithm-box">
    <div class="algorithm-banner">Algorithm: The Heterochronous Generative Network Model</div>
    
    <div class="algorithm-content">
      <p><strong>Input</strong>: (Possibly empty) seed adjacency matrix, \(A_{ij}\)</p>
      
      <p><strong>For</strong> \(t\) from \(1,\dots,T\) <strong>do</strong>:</p>
      <ol>
        <li>Compute distance transform \(d_{ij} = D_{ij}^\eta\) if distance relationship type is powerlaw and \(d_{ij} = \exp( \eta D_{ij} )\) if the distance relationship type is exponential.</li>
        <li>Compute affinity matrix \(K_{ij}\) from current adjacency matrix \(A_{ij}\).</li>
        <li>Compute affinity transform \(k_{ij} = K_{ij}^\gamma\) if affinity relationship type is powerlaw and \(k_{ij} = \exp( \gamma K_{ij})\) if the affinity relationship type is exponential.</li>
        <li>Compute heterochronous matrix \(H_{ij}(t)\) based on current time step $t$.</li>
        <li>Compute heterochronous transform \(h_{ij}(t) = H_{ij}^\lambda\) if heterochronous relationship type is powerlaw and \(h_{ij}(t) = \exp( \gamma H_{ij}(t))\) if the heterochronous relationship type is exponential.</li>
        <li>Compute unnormalised connection probabilities as the product of the distance, affinity, and heterochronous transforms: 
            $$\tilde{P}_{ij} \gets k_{ij} \times d_{ij} \times h_{ij}(t) $$</li>
        <li>Set probability of already present edges to zero: \(\tilde{P}_{ij} \gets (1 - A_{ij}) \tilde{P}_{ij}\)</li>
        <li>Set probability of self-connections to zero: \(\tilde{P}_{ii} \gets 0\)</li>
        <li>Normalise wiring probabilities:
            $$P_{ij} \gets \frac{\tilde{P}_{ij}}{ \sum_{ab} \tilde{P}_{ab} }$$</li>
        <li>Sample edge \((i,j)\) with probability \(P_{ij}\)</li>
        <li>Add edge to the adjacency matrix: \(A_{ij} \gets 1\) and \(A_{ji} \gets 1\)</li>
      </ol>
      <p><strong>End for</strong></p>
      
      <p><strong>Return</strong>: Adjacency matrix \(A_{ij}\)</p>
    </div>
  </div>
</div>

## Modelling the heterochronous gradient

In principle, the heterochronous matrix $H_{ij}(t)$ can be any (time-varying) matrix we choose. However, as we are modelling brain networks, we will aim to choose a biologically plausible wiring rule. In particular, we know that we want to capture a smooth change over space and time in the probability that a certain node to make a connection. For this reason, we model heterochronicity using a time‑varying cumulative Gaussian function. For each node $i$, we first calculate

$$
    g_i(t) = \exp\!\left[-\frac{ ||d_i - \mu(t)||^2 }{2\sigma^2}\right],
$$

where $d_i$ is the Euclidean distance of node $i$ from the origin point, $\sigma$ is the standard deviation of the Gaussian, and the centre $\mu(t)$ changes across time according to

$$
    \mu(t) = \frac{t-1}{T}\, d_{\max},
$$

where $d_{\max}$ is the spacial location of the node with the furthest distance from the origin point. Importantly, once a node becomes *active* and starts making connections, it remains active for the remainder of the heterochronous process. To model this, we take the maximum of the time-varying Gaussian over all previous time steps:

$$
 g^{a}_i(t) = \max_{t' \le t} g_i(t').
$$

The heterochronous matrix for any two nodes $i$ and $j$ is then given by:

$$
    H_{ij}(t) = \max\!\bigl(g^{a}_i(t), g^{a}_j(t) \bigr). 
$$
