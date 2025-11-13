# Weighted Generative Network Models

## Historical Background

[ Leave blank for now ]

## Weighted Networks

Connectivity in real brains is characterised not just by the presence or absence of connections, but by the strength or *weight* of those connections. From a structural perspective, the most popular approach is to estimate these weights using the number of white matter fibres connecting two brain regions, as estimated through tractography techniques applied to diffusion imaging data. Functional weights capture the magnitude of statistical dependencies between the activities of different brain regions. In both cases, the weight of a connection encodes the capacity for information flow between [nodes](glossary.md#node), with stronger connections denoting stronger communication. These differences in communication between nodes are an important detail in understanding brain functioning. As such, it is natural to search for an extension of the standard binary GNM which can account for not just the presence and absence of connections, but additionally their magnitude.   

Whilst binary [networks](glossary.md#network) are represented using an [adjacency matrix](glossary.md#adjacency-matrix-a) containing only values of $0$ and $1$ indicating the absence and presence of connections respectively, weighted networks are represented using a [weight matrix](glossary.md#weight-matrix-w) $W$ where $W_{ij}$ captures the strength of the connection between [nodes](glossary.md#node) $i$ and $j$. As discussed in the [Networks and Graphs](networks-and-graphs.md#representing-networks) section, the [weight matrix](glossary.md#weight-matrix-w) preserves all the structural information of the [adjacency matrix](glossary.md#adjacency-matrix-a) whilst providing additional detail about connection strengths. Many [network](glossary.md#network) measures can be adapted from binary to weighted versions by incorporating these connection strengths into the calculations.

## Weight Optimisation by Loss Minimisation

One approach to understanding weightes within a network is from a *normative* perspective. Normative methodologies within neuroscience  make predictions about an aspect of biology by hypothesising that the system has been optimised for some objective, subject to biological constraints. For example, wiring economy and the metabolic costs associated with neural communication would suggest that the brain ought to minimise the presence of long-range, high-weight connection. Alternatively, we may argue that the brain has been optimised for efficient information flow, and so ought to have a low characteristic path length; that is, information ought to be able to move through the network with in as few steps as possible. Different objectives correspond to different theories about the underlying organisational principles for neural systems.   

These hypotheses can be formalised using a loss function, $L(W)$, which we take the weights to minimise. [Below](#weight-optimisation-criteria), we discuss various choices for which loss function to use. Once we have specified a loss function, we must then optimise the weights of our network according to this loss function. We minimise the loss using a method known as *gradient descent*.  Gradient descent finds local minima of the loss function by repeatedly updating the weights in the direction of steepest descent for the loss function. The gradient $\frac{\partial L}{\partial W_{ij}}$ indicates the rate of change of in the loss function with respect to the weight $W_{ij}$. To minimise the loss, weights are updated in the opposite direction: $W_{ij} \leftarrow W_{ij} - \alpha \frac{\partial L}{\partial W_{ij}}$, where [$\alpha$](glossary.md#alpha-alpha) is the learning rate that controls the size of each update step. Larger values of [$\alpha$](glossary.md#alpha-alpha) result in greater changes to the weights. By repeatedly performing such updates, we can find weight configurations which achieve a low loss and are therefore more plausible according to our hypothesis. 

Although the wGNM uses gradient descent to optimise the weights within the network, this should not be understood as articulating a theory of how weight changes are effected within the developing brain. Instead, wGNMs give a normative theory of what principles organise networks; the choice of gradient descent as an optimisation method is secondary. Indeed, the gradient descent optimisation employed in weighted GNMs should be understood as a computational method for finding weight configurations that satisfy particular normative principles, rather than as a model of the actual biological processes that modify synaptic strengths during brain development. The value of these models lies in their ability to generate [networks](glossary.md#network) whose final properties resemble those of found in real brain [networks](glossary.md#network), not in the biological plausibility of their intermediate computational steps.

## The Weighted Generative Network Model

The [weighted generative network model](glossary.md#weighted-generative-network-model) extends the [binary generative network model](glossary.md#binary-generative-network-model) by incorporating both binary updates that build the [network](glossary.md#network) topology and weighted updates that adjust the strengths of existing connections. The wGNM model interleaves binary and weighted updates; at each iteration, some number of binary updates are performed, followed by some number of weighted updates. In the simplest case, the ratio of binary to weighted updates is 1:1. During binary updates, new [edges](glossary.md#edge) are added to the [network](glossary.md#network) according to the same mechanism as in the [binary generative network model](binary-gnms.md), using [distance](glossary.md#distance-transform-d) and [affinity transforms](glossary.md#affinity-transform-k) to determine [wiring probabilities](glossary.md#wiring-probabilities-p_ij). During weighted updates, the strengths of existing connections are modified through gradient descent on a specified loss function. After the gradient descent update, weights are clipped between minimum and maximum values. In the simplest case, $W_{\text{lower}} = 0$ to ensure that weights remain non-negative, while $W_{\text{upper}} = \infty$, allowing for unbounded positive weights. 

<div id="weighted-gnm-algorithm" class="algorithm-anchor">
  <div class="algorithm-box">
    <div class="algorithm-banner">Algorithm: The Weighted Generative Network Model</div>
    
    <div class="algorithm-content">
      <p><strong>Input</strong>: Seed adjacency matrix \(A_{ij}\), seed weight matrix \(W_{ij}\), binary parameters, weighted parameters</p>
      
      <p><strong>For</strong> number of iterations <strong>do</strong>:</p>
      <ol>
        <li><strong>For</strong> number of binary updates per iteration <strong>do</strong>:
          <ul>
            <li>Compute wiring probabilities $P_{ij}$ as in the binary GNM.</li>
            <li>Sample a new edge $(i,j)$ from the wiring probabilities $P_{ij}$.</li>
            <li>Add the new edge to adjacency matrix: \(A_{ij} \gets 1\) and \(A_{ji} \gets 1\).</li>
            <li>Initialise corresponding weights within the weight matrix: \(W_{ij} \gets 1\) and \(W_{ji} \gets 1\)</li>
          </ul>
        </li>
        <li><strong>For</strong> number of weighted updates per iteration <strong>do</strong>:
          <ul>
            <li>Compute gradients of the loss function: \(\frac{\partial L}{\partial W_{ij}}\) for all \(i,j\) where \(A_{ij} = 1\)</li>
            <li>Update weights via gradient descent $$ W_{ij} \gets W_{ij} - \alpha \frac{\partial L}{\partial W_{ij}}, $$ (or \(+\) if maximising the loss)</li>
            <li>Optionally clip the weights between bounds: \(W_{ij} \gets \text{clip}(W_{ij}, W_{\text{lower}}, W_{\text{upper}})\)</li>
          </ul>
        </li>
      </ol>
      <p><strong>End for</strong></p>
      
      <p><strong>Return</strong>: Final adjacency matrix \(A_{ij}\) and weight matrix \(W_{ij}\)</p>
    </div>
  </div>
</div>

## Weight Optimisation Criteria

Several different [weight optimisation criteria](glossary.md#weight-optimisation-criterion) are available for guiding the evolution of connection strengths in weighted generative network models. However, the toolbox also provide a way to easily implement and test additional criteria. The crtieria implemented are often parametrised by a parameter, $\omega$. 

The simplest criterion implemented within the toolbox is the weight criteria, $L(W) = \sum_{ij} W_{ij}^\omega$. For $\omega = 1$, this is simply the total weight present in the network; minimisation of this criterion therefore corresponds to a principle of wiring efficiency whereby stronger connections are discouraged. Weighted distance augments this by penalising strong connections more the larger the distance they span is. The loss in this case is given by $L(W) = \sum_{ij} \left( W_{ij} D_{ij} \right)^\omega$.  The biological motivation for minimising weighted distance lies in the energetic costs of building and maintaining neural connections: longer connections require more resources and are therefore subject to stronger evolutionary pressure for efficiency. Normalised weight and normalised weighted distance provide further varients on this core principle. 

A more sophisticated set of criteria is built upon [communicability](glossary.md#communicability). Recall that the number of paths of length $n$ between nodes $i$ and $j$ is simply $[A^n]_{ij}$, where $A^n$ is the $n$-th power of the adjacency matrix $A$. In a binary network, communicability can be understood as the sum of all possible paths between $i$ and $j$, with decreasing weighting assigned to paths of longer length. Specifically, a path of length $n$ recieves a weighting of $1/n!$ where $n! = n \times (n-1) \times \dots \times 2 \times 1$ is the factorial of $n$. In this manner communicability can be seen as a measure of the total information flow between nodes via any path, with lower information flow through longer paths. It is computed as 
$$
C_{ij} = [A^0]\_{ij} + \frac{1}{1!} A\_{ij} + \frac{1}{2!} [A^2]\_{ij} + \frac{1}{3!} [A^3]\_{ij} = \exp( A )\_{ij}, 
$$
where $\exp(A)$ is the *matrix exponential* of the adjacency matrix $A$. 

For a weighted network, we must adjust the definition of [communicability](glossary.md#communicability). Communicability is still the matrix exponential, but now of the normalised [weight matrix](glossary.md#weight-matrix-w) rather than the adjacency matrix. Specifically, we first normalising the weight matrix by the [node strengths](glossary.md#node-strength-s), $S^{-1/2}W S^{-1/2} = W_{ij}/\sqrt{s_i s_j}$, where $S$ is a diagonal matrix with $S_{ii} = s_i$. By normalising like so, we include in our notion of information flow between nodes that information is less likely to travel down one connection if there are many other strong connections out of the same node. We then apply the matrix exponential to this normalised weight matrix, 
$$
C_{ij} = \exp( S^{-1/2} W S^{-1/2} )\_{ij}.
$$ 
The communicability loss is then just the sum of all communicabilities in the network, raised to the power of $\omega$, 
$$
L(W) = \sum_{ij} C\_{ij}^\omega.
$$
The basic communicability loss can be extended in various ways, including weighting the communicabilities by the distance between nodes and normalising the total communicability by the maximum communicability in the network. See the table below.  

<div class="table-container">
  <div class="table-banner">Table: Weight optimisation criteria for weighted generative network models</div>
  
  <table class="gnm-table">
    <thead>
      <tr>
        <th>Criterion</th>
        <th>Loss function, $L(W)$</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Weight</td>
        <td class="formula-column">\(\sum_{ij} W_{ij}^\omega\)</td>
        <td class="notes-column">Minimises the sum of connection weights raised to power \(\omega\)</td>
      </tr>
      <tr>
        <td>Normalised Weight</td>
        <td class="formula-column">\(\sum_{ij} \left( \frac{W_{ij}}{\max_{ab} W_{ab}} \right)^\omega\)</td>
        <td class="notes-column">Weight criterion normalised by maximum weight in the network</td>
      </tr>
      <tr>
        <td>Weighted Distance</td>
        <td class="formula-column">\(\sum_{ij} (W_{ij} D_{ij})^\omega\)</td>
        <td class="notes-column">Penalises strong connections between distant nodes</td>
      </tr>
      <tr>
        <td>Normalised Weighted Distance</td>
        <td class="formula-column">\(\sum_{ij} \left( \frac{W_{ij} D_{ij}}{\max_{ab} W_{ab} D_{ab}} \right)^\omega\)</td>
        <td class="notes-column">Weighted distance criterion normalised by maximum weighted distance</td>
      </tr>
      <tr>
        <td>Communicability</td>
        <td class="formula-column">\(\sum_{ij} C_{ij}^\omega\)</td>
        <td class="notes-column">Minimises total communicability between all node pairs</td>
      </tr>
      <tr>
        <td>Normalised Communicability</td>
        <td class="formula-column">\(\sum_{ij} \left(\frac{C_{ij}}{\max_{ab} C_{ab}}\right)^\omega\)</td>
        <td class="notes-column">Communicability criterion normalised by maximum communicability</td>
      </tr>
      <tr>
        <td>Distance-Weighted Communicability</td>
        <td class="formula-column">\(\sum_{ij} (C_{ij} D_{ij})^\omega\)</td>
        <td class="notes-column">Combines communicability with spatial constraints</td>
      </tr>
      <tr>
        <td>Normalised Distance-Weighted Communicability</td>
        <td class="formula-column">\(\sum_{ij} \left(\frac{D_{ij} C_{ij}}{\max_{ab} D_{ab} C_{ab}}\right)^\omega\)</td>
        <td class="notes-column">Distance-weighted communicability normalised by maximum value</td>
      </tr>
    </tbody>
  </table>
</div>