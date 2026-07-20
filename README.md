# Differentially-Private Federated Recommendation System (DP-SC2)

This repository contains a prototype implementation of a **Privacy-Preserving Sequential Recommendation Framework** utilizing Federated Learning (FL), Differential Privacy (DP), and Transformer-based Recommendation Models. It features **Behavioral Client Clustering**, **Risk-Aware Privacy Budgets**, and **Privacy-and-Trust-Aware Aggregation** to optimize the trade-off between personalization accuracy and privacy guarantees.

---

## 🌟 Key Architecture & Concepts

Traditional federated recommendation systems often apply a uniform privacy budget and a single global model update logic (e.g., standard FedAvg). This prototype addresses these limitations by introducing:

### 1. Behavioral Client Clustering ([clustering.py](clustering.py))
- **Local Representation Extraction**: In [extract_behavior_metrics](clustering.py#L37), each client computes a semantic behavioral vector representing their sequence history. It does this by extracting item embeddings from its local [TransformerRecommender](model.py#L4) model, averaging the embeddings over the session length, and averaging across batches.
- **Differential Privacy (DP) for Behavior**: To prevent leakage of the client's interests during clustering, Laplace noise is added to the local representation before sending it to the server.
- **Server-Side Clustering**: In the [AdaptiveClusteredStrategy](server.py#L10), the server uses KMeans ([ClientClusterer](clustering.py#L5)) to group clients into cluster assignments.

### 2. Transformer-Based Sequential Recommendations ([model.py](model.py))
- Implementation of a custom [TransformerRecommender](model.py#L4) in PyTorch.
- Utilizes an item embedding layer, a trainable positional encoding layer, multi-head self-attention (via `nn.TransformerEncoderLayer`), and a final projection layer targeting the next item ID in the sequence.

### 3. Risk-Aware Privacy Budgets ([privacy.py](privacy.py))
- Local training utilizes the Opacus library to run Differentially-Private Stochastic Gradient Descent (DP-SGD).
- Rather than using static size thresholds, the server dynamically evaluates privacy leakage risk ($R_k$) per cluster and calibrates cluster-specific noise multipliers ($\sigma_k$).
- Incorporates Jaccard stability feedback ($S_k^t$) to scale noise depending on how cluster memberships evolve over federated rounds (unstable clusters get higher noise, stable clusters get lower noise).

---

## 🔢 Understanding the Numbers & Patent Significance

For academic and patent evaluation, the following details explain what the numbers in this project represent, how the model responds to new inputs, and the patent's core contributions:

### 1. What the Numbers Represent
- **Item IDs (1 to 2000)**: The integers in interaction sequences (e.g., `15, 122, 504`) represent unique catalog items (e.g., movies, products, articles) in the recommendation vocabulary.
- **Client IDs (1 to 4)**: Represent simulated edge devices (e.g., mobile phones).
- **Behavioral Bias**: 
  - Clients 1 & 2 primarily interact with items in the range `1-1000`.
  - Clients 3 & 4 primarily interact with items in the range `1000-2000`.
  - This simulates distinct user personas/demographics.

### 2. What Happens When New Numbers Are Inputted
When a sequence of numbers (e.g., `15, 122, 504`) is sent to the recommendation engine:
1. **Preprocessing**: The sequence is padded with `0`s (indicating padding) or truncated to match the model's expected sequence length of `10` (e.g., `[0, 0, 0, 0, 0, 0, 0, 15, 122, 504]`).
2. **Embedding Lookups**: The integers are mapped to continuous vectors via `nn.Embedding`.
3. **Self-Attention Modeling**: The sequence is passed through the Transformer layers, which model sequential dependencies (e.g., "users who liked item 15, then item 122, then item 504 are likely to view...").
4. **Logit Projection & Top-K**: The model computes output scores across all 2000 items. The top-3 highest-scoring item IDs are returned.
5. **Contextual Shift**: If you input numbers corresponding to Clients 1 & 2's interest range (e.g., `12, 45, 87`), the model recommends items in that region. If you enter numbers in the `1000-2000` range (e.g., `1012, 1045, 1087`), the model dynamically recommends items matching the cluster behaviors of Clients 3 & 4.

### 3. What the Faculty/Patent Evaluators Need to Understand (Core Innovations)
- **Innovative Client Clustering (Privacy-First)**: Clients cluster based on behavior without exposing raw histories. Instead of sharing raw interaction logs, clients extract noisy session embeddings.
- **Risk-Driven Privacy Budget Allocation (Patent Claim)**: Dynamic budget allocation based on cluster-level privacy leakage risk estimation ($R_k$), scaling noise down in large, homogeneous clusters and scaling it up in small or high-variance clusters. A cluster stability feedback loop monitors temporal cluster evolution to adjust noise.
- **Privacy-and-Trust Aware Aggregation (Patent Claim)**: Custom aggregation mechanism replacing FedAvg. Re-weights client contributions based on Data Quality ($DQ_i$), Update Consistency ($UC_i$, consensus deviation), and individual Privacy Risk ($PR_i$, behavioral distance to centroid).
- **Federated Transformer Aggregation**: Showcases how high-capacity sequential models like Transformers can be aggregated federatedly while respecting strict differential privacy constraints.

---

## 📝 Patent-Grade System Core Algorithm (DP-SC2)

This section details the step-by-step execution flow of the **Differentially-Private Cluster-Specific Federated Learning Algorithm**. This is the core novelty of the patent, replacing static, uniform federated learning (like standard FedAvg) with a dynamic, risk-aware, and trust-weighted approach.

### Phase 1: Local Semantic Profiling (On Each Client $i$)
1. **Embedding Extraction**: Retrieve local session sequences $S_i$. Pass them through the local Transformer model's embedding layers to extract user representations.
2. **Mean Representation**: Average the sequence embeddings to obtain a session profile vector:
   $$v_i = \text{mean}(Embedding(S_i))$$
3. **Differentially Private Perturbation**: Obfuscate the vector with Laplace noise to prevent interest leakage during clustering:
   $$\tilde{v}_i = v_i + \text{Laplace}(0, \lambda)$$
4. **Telemetry**: Client $i$ transmits the noisy profile $\tilde{v}_i$ to the Federated Server.

### Phase 2: Behavioral Clustering & Risk-Aware Budgeting (On the Server)
*(Patent Claim 1: Dynamic privacy budget allocation based on cluster-level privacy leakage risk estimation)*

1. **KMeans Clustering**: Group clients into $K$ distinct clusters based on their noisy behavioral features:
   $$C = \text{KMeans}(\tilde{v}_1, \dots, \tilde{v}_U)$$
2. **Risk Evaluation**: For each cluster $k$, calculate the **Privacy Leakage Risk ($R_k$)**:
   - $N_k$: Cluster size (smaller size = higher risk of deanonymization).
   - $V_k$: Cluster variance (higher variance = less homogeneous, higher risk).
   - $O_k$: Outlier score based on distance from the global centroid.
   
   **Adaptive Risk Estimator (Fully Adaptive)**:
   Define $A = \frac{1}{N_k}$, $B = V_k$, and $C = O_k$.
   Compute dynamic weighting coefficients (no arbitrary constants, no training):
   $$\alpha = \frac{A}{A+B+C}, \quad \beta = \frac{B}{A+B+C}, \quad \gamma = \frac{C}{A+B+C}$$
   $$R_k = \alpha A + \beta B + \gamma C$$

3. **Budget Allocation (Privacy-Utility Optimization)**: 
   We dynamically compute privacy budgets through a risk-aware optimization framework rather than arbitrarily assigning noise values.
   $$\sigma_k = \arg\max (U(\sigma_k) - \lambda R_k) \quad \text{subject to} \quad \epsilon_k \le \epsilon_{max}$$
4. **Cluster Stability Feedback**: Measure the Jaccard similarity ($S_k^t$) of cluster memberships between round $t$ and $t-1$. Scale the noise to penalize unstable clusters:
   $$\sigma_k \leftarrow \sigma_k \times (1.5 - 0.5 S_k^t)$$

### Phase 3: Cluster-Specific Training (Local DP-SGD)
1. **Broadcast**: Server sends the cluster-specific model weights $W^t_k$ and the designated noise multiplier $\sigma_k$ to all clients in cluster $k$.
2. **Local DP-SGD Update**: Clients update their local models using their private data. 
   - Gradients are clipped to a max norm $C_{\text{clip}}$.
   - Gaussian noise calibrated to the server-assigned $\sigma_k$ is added:
     $$g \leftarrow g + \mathcal{N}(0, \sigma_k^2 C_{\text{clip}}^2)$$
3. **Feedback**: Clients send their updated parameters $W^{t+1}_{i}$ back to the server.

### Phase 4: Privacy-and-Trust-Aware Aggregation (On the Server)
*(Patent Claim 2: Custom aggregation re-weighting based on consensus deviation, data quality, and privacy risk)*

1. **Compute Trust Weights ($w_i$)**: Instead of blindly averaging client models (like FedAvg), the server evaluates each client $i \in C_k$:
   - **Update Consistency ($UC_i$)**: Measures deviation from the cluster's consensus update. Malicious or divergent updates get penalized.
     $$UC_i = \frac{1}{1 + \text{MSE}(\Delta W_i, \overline{\Delta W_k})}$$
   - **Privacy Risk ($PR_i$)**: The behavioral distance of the client to their cluster's centroid.
     $$PR_i = \|v_i - \mu_k\|^2$$
   - **Data Quality ($D_i$)**: Number of local training samples.
   - **Final Weight (Solves the Minority-User Criticism)**:
     $$w_i = \max\left(w_{min}, D_i UC_i \frac{1}{1 + PR_i}\right)$$
2. **Cluster & Global Aggregation**: Aggregate client models using the normalized trust weights to form the new cluster models ($W^{t+1}_k$) and the global model.

---

## 📂 Codebase Walkthrough

Here is a breakdown of the core files in the repository:

* **[app.py](app.py)**: The Flask web backend that orchestrates the simulation and serving of inference requests.
  - Serves the frontend web assets.
  - Exposes the `/stream` SSE endpoint to launch and monitor the FL simulation run in real-time.
  - Exposes the `/recommend` POST endpoint to generate predictions using the trained global model `global_model.npz`.
* **[server.py](server.py)**: The Flower simulation server running a custom `AdaptiveClusteredStrategy`. Coordinates model parameter aggregation, clustering assignments, and dynamic privacy budget computation.
* **[client.py](client.py)**: Flower client representing individual users. Performs local training with DP-SGD via Opacus and evaluates parameters sent by the server.
* **[model.py](model.py)**: PyTorch definition of the sequence recommendation Transformer architecture.
* **[clustering.py](clustering.py)**: Features clustering utilities including the KMeans grouping class and the noisy session representation extractor.
* **[dataset.py](dataset.py)**: Handles sequence dataset synthesis and partitions client data with custom skews to simulate different behavioral clusters.
* **[privacy.py](privacy.py)**: Utility to set up the Opacus `PrivacyEngine` mapping.
* **[run_simulation.sh](run_simulation.sh)**: Shell script executing 1 Flower server instance and 4 clients in parallel.
* **[static/](static)**: Single-page application assets:
  - **[index.html](static/index.html)**: Interactive dashboard structure.
  - **[app.js](static/app.js)**: Feeds event stream data, renders live updates on charts (Chart.js), tracks metrics, and fires recommendation predictions.
  - **[style.css](static/style.css)**: Sleek, responsive dark-mode styling with subtle animations.

---

## 🛠️ Installation & Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Set Up Virtual Environment & Dependencies
Initialize a virtual environment to manage dependencies locally:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Run the Dashboard Application
The best way to interact with the project is through the Flask dashboard:
```bash
python3 app.py
```
This runs the Flask server on `http://127.0.0.1:5000`.

### 4. Interactive Simulation & Recommendations
1. Open `http://127.0.0.1:5000` in your web browser.
2. Click **Start Simulation** to spin up the local federated learning simulation.
3. Observe live updates in the dashboard panels:
   - **System Logs**: Displays output from the server and clients in real-time.
   - **Convergence Graph**: Tracks the evaluation loss across rounds.
   - **Behavioral Clusters**: Visualizes the client-to-cluster assignments.
   - **Privacy Budget**: Displays the dynamic noise multipliers applied to each cluster.
4. Once the training completes, the global model will be saved as `global_model.npz`.
5. Enter a sequence of items (e.g., `15, 122, 504`) in the **Get Recommendations** widget to generate real-time recommendations. Alternatively, upload a custom CSV file containing item sequences.
