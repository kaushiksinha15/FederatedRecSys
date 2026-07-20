from flask import Flask, Response, request, jsonify, send_from_directory
import subprocess
import os
import torch
import numpy as np
from collections import OrderedDict
from model import TransformerRecommender
from opacus.validators import ModuleValidator

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/stream')
def stream():
    def generate():
        # Make sure there are no other simulations running
        os.system("pkill -f server.py ; pkill -f client.py")
        
        # Start the simulation using the shell script
        # We use stdbuf to disable stdout buffering or run inside bash
        process = subprocess.Popen(
            ["bash", "./run_simulation.sh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1 # Line buffered
        )

        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            # Send the line as Server-Sent Event (SSE)
            yield f"data: {line}\n\n"
            
        process.stdout.close()
        process.wait()
        
        yield "data: [SIMULATION_COMPLETE]\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Check if request is JSON for manual input
    if request.is_json:
        data = request.get_json()
        if not data or 'sequence' not in data:
            return jsonify({'error': 'Missing "sequence" in JSON body'}), 400
        sequence_data = data['sequence']
        if isinstance(sequence_data, str):
            try:
                seq = [int(x.strip()) for x in sequence_data.split(',') if x.strip()]
            except ValueError:
                return jsonify({'error': 'Invalid sequence format. Expected comma-separated integers.'}), 400
        elif isinstance(sequence_data, list):
            try:
                seq = [int(x) for x in sequence_data]
            except ValueError:
                return jsonify({'error': 'Invalid sequence elements. Expected integers.'}), 400
        else:
            return jsonify({'error': 'Invalid sequence format. Expected list or string.'}), 400
            
        sequences = [seq]
    else:
        # Fallback to file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file part or JSON data provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        try:
            content = file.read().decode('utf-8')
            lines = content.strip().split('\n')
            sequences = []
            for line in lines:
                if not line.strip():
                    continue
                seq = [int(x.strip()) for x in line.split(',') if x.strip()]
                sequences.append(seq)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
    
    if not os.path.exists("global_model.npz"):
        return jsonify({'error': 'Model not trained yet. Please run the simulation first.'}), 400

    try:
        # Load the saved model parameters
        npzfile = np.load("global_model.npz")
        parameters = [npzfile[f'arr_{i}'] for i in range(len(npzfile.files))]
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerRecommender()
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
        model = model.to(device)
        
        # Set parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        results = []
        for seq in sequences:
            original_seq = list(seq)
            
            # Pad or truncate to seq_len=10
            if len(seq) > 10:
                seq = seq[-10:]
            while len(seq) < 10:
                seq.insert(0, 0) # Pad with 0
                
            input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                # Get top 3 recommendations
                topk = torch.topk(outputs[0], 3)
                predicted_ids = topk.indices.cpu().numpy().tolist()
                
            results.append({
                'input_sequence': original_seq,
                'recommendations': predicted_ids
            })
            
        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
