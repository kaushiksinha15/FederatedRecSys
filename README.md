# Federated Recommendation System with Semantic Behavior-Aware Privacy

A privacy-preserving federated recommendation system that combines **Transformer-based sequential recommendation**, **Semantic Behavior Characterization**, **Behavior-based Client Clustering**, **Behavior-aware Differential Privacy**, and **Trust-aware Secure Aggregation** to generate secure personalized recommendations without sharing raw user data.

---

# Architecture

```text
                User Interactions
                       │
                       ▼
        Transformer Semantic Encoder
                       │
                       ▼
     Semantic Behavior Characterization
                       │
                       ▼
      Behavior-based Client Clustering
                       │
                       ▼
     Behavior-aware Differential Privacy
                       │
                       ▼
      Trust-aware Secure Aggregation
                       │
                       ▼
          Global Recommendation Model
```

---

# Features

* Transformer-based sequential recommendation model
* Federated Learning using Flower
* Semantic Behavior Characterization from user interaction embeddings
* Behavior-based client clustering
* Behavior-aware Differential Privacy using Opacus
* Trust-aware Secure Aggregation
* REST API for recommendation inference
* Web interface for simulation
* CSV and manual recommendation input support

---

# Project Structure

```text
.
├── app.py                  # Flask web application
├── client.py               # Flower client implementation
├── server.py               # Flower server with clustered aggregation
├── model.py                # Transformer recommender model
├── clustering.py           # Semantic behavior characterization & clustering
├── privacy.py              # Differential Privacy setup
├── dataset.py              # Synthetic federated dataset
├── run_simulation.sh       # Linux/macOS simulation launcher
├── demo_upload.csv         # Sample upload file
├── sample_seq.csv          # Sample recommendation sequences
├── requirements.txt
├── static/
│   └── index.html
└── README.md
```

---

# Technologies Used

* Python
* PyTorch
* Flower
* Opacus
* Flask
* NumPy
* Scikit-Learn

---

# Installation

Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

Create virtual environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Federated Simulation

## macOS / Linux

Make the script executable

```bash
chmod +x run_simulation.sh
```

Run

```bash
./run_simulation.sh
```

---

## Windows

Since `.sh` files don't run natively, start the components separately.

### Terminal 1

```powershell
venv\Scripts\activate

python server.py
```

---

### Terminal 2

```powershell
venv\Scripts\activate

python client.py --client_id 1
```

---

### Terminal 3

```powershell
venv\Scripts\activate

python client.py --client_id 2
```

---

### Terminal 4

```powershell
venv\Scripts\activate

python client.py --client_id 3
```

---

### Terminal 5

```powershell
venv\Scripts\activate

python client.py --client_id 4
```

The server automatically performs three federated learning rounds and saves

```
global_model.npz
```

after completion.

---

# Running the Flask Application

Open another terminal.

### Windows

```powershell
venv\Scripts\activate

python app.py
```

### macOS / Linux

```bash
source venv/bin/activate

python3 app.py
```

Open

```
http://127.0.0.1:5000
```

---

# API Usage

## Manual Recommendation

**POST**

```
/recommend
```

Example

```json
{
    "sequence":[12,45,102,500,320]
}
```

or

```json
{
    "sequence":"12,45,102,500,320"
}
```

---

## CSV Upload

Upload a CSV containing one interaction sequence per line.

Example

```text
12,45,102,500
1050,1200,1010
5,15,25,35,45
```

---

# Simulation Workflow

1. Each client trains locally on its own interaction history.
2. Transformer embeddings are generated from local interaction sequences.
3. Semantic Behavior Characterization derives behavioral descriptors.
4. Clients are clustered according to behavioral similarity.
5. Differential privacy noise is configured according to cluster behavior.
6. Trust-aware aggregation combines client updates.
7. The global recommendation model is updated and redistributed.

---

# Behavioral Descriptors

Each client computes:

* Preference Diversity
* Behavioral Consistency
* Temporal Stability
* Behavioral Uniqueness
* Cluster Affinity

These descriptors guide privacy allocation before federated aggregation.

---

# Output

After training, the server generates

```
global_model.npz
```

which is later loaded by the Flask application for recommendation inference.

---

# Sample Recommendation Request

```bash
curl -X POST http://127.0.0.1:5000/recommend \
-H "Content-Type: application/json" \
-d "{\"sequence\":[12,45,102,500,320]}"
```

---

# Requirements

* Python 3.10+
* PyTorch 2.x
* Flower 1.7+
* Opacus 1.4+
* Flask
* NumPy
* Scikit-Learn

---

# Notes

* The project currently uses a synthetic dataset for demonstration purposes.
* Behavioral descriptors are computed using placeholder statistical functions over Transformer embeddings and can be replaced with more sophisticated characterization methods.
* The implementation is modular, allowing replacement of the recommendation model, privacy mechanism, or clustering strategy without affecting the overall pipeline.

---

# Citation

If you use this project in academic work, please cite the associated publication or patent when available.
