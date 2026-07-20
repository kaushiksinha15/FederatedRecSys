document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const terminal = document.getElementById('terminal');
    const statusBadge = document.getElementById('status-badge');
    
    // Metrics DOM Elements
    const roundValue = document.getElementById('round-value');
    const clientsValue = document.getElementById('clients-value');
    const lossValue = document.getElementById('loss-value');
    const clusterContainer = document.getElementById('cluster-container');
    const epsilonContainer = document.getElementById('epsilon-container');
    
    // DOM Elements for Recommendation Section
    const tabManual = document.getElementById('tabManual');
    const tabFile = document.getElementById('tabFile');
    const manualSection = document.getElementById('manualSection');
    const fileSection = document.getElementById('fileSection');
    const csvFileInput = document.getElementById('csvFileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const manualSequenceInput = document.getElementById('manualSequenceInput');
    const recommendBtn = document.getElementById('recommendBtn');
    const recommendationsContainer = document.getElementById('recommendations-container');
    
    let simulationCompleted = false;

    // Chart.js Setup
    const ctx = document.getElementById('lossChart').getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(124, 58, 237, 0.5)');
    gradient.addColorStop(1, 'rgba(124, 58, 237, 0.0)');

    const lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Global Model Loss',
                data: [],
                borderColor: '#7c3aed',
                backgroundColor: gradient,
                borderWidth: 2,
                pointBackgroundColor: '#3b82f6',
                pointBorderColor: '#fff',
                pointRadius: 4,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });

    // Helper: Append to Terminal
    function logToTerminal(message, type = 'info') {
        const line = document.createElement('div');
        line.className = `log-line ${type}`;
        line.textContent = message;
        terminal.appendChild(line);
        terminal.scrollTop = terminal.scrollHeight;
    }

    // Helper: Enable/Disable recommendation inputs
    function updateRecommendationControls() {
        if (!simulationCompleted) {
            recommendBtn.disabled = true;
            uploadBtn.disabled = true;
            return;
        }
        recommendBtn.disabled = manualSequenceInput.value.trim() === '';
        uploadBtn.disabled = csvFileInput.files.length === 0;
    }

    // Tab Switching Logic
    tabManual.addEventListener('click', () => {
        tabManual.classList.add('active');
        tabFile.classList.remove('active');
        manualSection.classList.add('active');
        fileSection.classList.remove('active');
    });

    tabFile.addEventListener('click', () => {
        tabFile.classList.add('active');
        tabManual.classList.remove('active');
        fileSection.classList.add('active');
        manualSection.classList.remove('active');
    });

    startBtn.addEventListener('click', () => {
        // Reset State
        startBtn.disabled = true;
        startBtn.querySelector('.btn-text').textContent = 'Simulation Running...';
        statusBadge.textContent = 'Running';
        statusBadge.className = 'badge running';
        
        terminal.innerHTML = '';
        logToTerminal('Initializing Federated Learning framework...', 'system');
        
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.update();
        
        roundValue.textContent = '0 / 3';
        clientsValue.textContent = '0';
        lossValue.textContent = '--';
        clusterContainer.innerHTML = '';
        epsilonContainer.innerHTML = '';
        
        simulationCompleted = false;
        updateRecommendationControls();

        // Start SSE Stream
        const eventSource = new EventSource('/stream');

        let activeClients = 0;

        eventSource.onmessage = (event) => {
            const data = event.data;

            if (data === '[SIMULATION_COMPLETE]') {
                eventSource.close();
                logToTerminal('Simulation Finished Successfully.', 'success');
                startBtn.disabled = false;
                startBtn.querySelector('.btn-text').textContent = 'Start Simulation';
                statusBadge.textContent = 'Idle';
                statusBadge.className = 'badge idle';
                
                simulationCompleted = true;
                updateRecommendationControls();
                return;
            }

            // Raw Log Output
            logToTerminal(data);

            // --- Parsing Logic for Dashboard Updates ---

            // Detect Clients Starting
            if (data.includes('Starting client')) {
                activeClients++;
                clientsValue.textContent = activeClients.toString();
            }

            // Detect Client Finished Training & Epsilon (Keep for legacy/debugging)
            if (data.includes('finished training. Epsilon spent:')) {
                const match = data.match(/Client (\d+) finished training\. Epsilon spent: ([\d\.]+)/);
                if (match) {
                    // We now emphasize cluster-level budgets, so we just log this to terminal
                    logToTerminal(`Client ${match[1]} used local DP epsilon: ${match[2]}`);
                }
            }

            // Detect DP-SC2 Dynamic Budgets
            if (data.includes('-> Assigned Noise=')) {
                const match = data.match(/Cluster (\d+).*?Risk=([\d\.]+).*?Assigned Noise=([\d\.]+)/);
                if (match) {
                    const clusterId = match[1];
                    const risk = match[2];
                    const epsilon = match[3];
                    
                    let pill = document.getElementById(`eps-cluster-${clusterId}`);
                    if (!pill) {
                        pill = document.createElement('div');
                        pill.id = `eps-cluster-${clusterId}`;
                        pill.className = `client-pill cluster-${clusterId}`;
                        epsilonContainer.appendChild(pill);
                    }
                    pill.innerHTML = `<span>Cluster ${clusterId} Optimal Budget (Risk: ${risk})</span> <span style="color:var(--warning)">Noise = ${epsilon}</span>`;
                }
            }

            // Highlight Trust-Aware Aggregation Weights
            if (data.includes('Aggregation weight ->')) {
                const lines = terminal.getElementsByClassName('log-line');
                if (lines.length > 0) {
                    lines[lines.length - 1].classList.add('highlight');
                }
            }

            // Detect Round Start
            if (data.includes('--- Server Round')) {
                const match = data.match(/Round (\d+) ---/);
                if (match) {
                    roundValue.textContent = `${match[1]} / 3`;
                }
            }

            // Detect Clusters
            if (data.includes('Client Clusters:')) {
                clusterContainer.innerHTML = ''; // Clear previous
                // Parse dict string e.g. {'0': 1, '1': 1, '2': 0, '3': 0}
                const match = data.match(/{.*}/);
                if (match) {
                    try {
                        const str = match[0].replace(/'/g, '"');
                        const clusters = JSON.parse(str);
                        
                        Object.keys(clusters).forEach((id, idx) => {
                            const clusterId = clusters[id];
                            const pill = document.createElement('div');
                            // Map simple id to 1-4 for display based on index
                            pill.className = `client-pill cluster-${clusterId}`;
                            pill.innerHTML = `<span>Client ${idx + 1}</span> <span>Cluster ${clusterId}</span>`;
                            clusterContainer.appendChild(pill);
                        });
                    } catch (e) {
                        console.error('Failed to parse clusters', e);
                    }
                }
            }

            // Detect Loss Evaluation
            if (data.includes('round ') && data.includes(':')) {
                const match = data.match(/round (\d+): ([\d\.]+)/);
                if (match) {
                    const roundNum = parseInt(match[1]);
                    const lossVal = parseFloat(match[2]);
                    
                    lossValue.textContent = lossVal.toFixed(4);
                    
                    // Update Chart
                    lossChart.data.labels.push(`Round ${roundNum}`);
                    lossChart.data.datasets[0].data.push(lossVal);
                    lossChart.update();
                }
            }
        };

        eventSource.onerror = (error) => {
            console.error('EventSource failed:', error);
            eventSource.close();
            logToTerminal('Connection to simulation stream lost.', 'error');
            startBtn.disabled = false;
            startBtn.querySelector('.btn-text').textContent = 'Start Simulation';
            statusBadge.textContent = 'Error';
            statusBadge.className = 'badge idle';
        };
    });

    // Handle Manual Sequence input changes
    manualSequenceInput.addEventListener('input', () => {
        updateRecommendationControls();
    });

    // Handle Manual Recommendation Submit
    recommendBtn.addEventListener('click', async () => {
        const seqVal = manualSequenceInput.value.trim();
        if (!seqVal) return;

        recommendBtn.disabled = true;
        recommendBtn.querySelector('.btn-text').textContent = 'Processing...';
        recommendationsContainer.innerHTML = '<div class="empty-state">Processing your sequence...</div>';

        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ sequence: seqVal })
            });
            const data = await response.json();

            if (response.ok) {
                recommendationsContainer.innerHTML = '';
                if (data.results && data.results.length > 0) {
                    data.results.forEach((res) => {
                        const itemDiv = document.createElement('div');
                        itemDiv.className = 'recommendation-item';
                        itemDiv.innerHTML = `
                            <div class="rec-sequence">Input Sequence: [${res.input_sequence.join(', ')}]</div>
                            <div class="rec-prediction">Top Recommendations: ${res.recommendations.join(', ')}</div>
                        `;
                        recommendationsContainer.appendChild(itemDiv);
                    });
                    logToTerminal(`Generated recommendations for sequence: [${seqVal}]`, 'success');
                } else {
                    recommendationsContainer.innerHTML = '<div class="empty-state">No valid recommendations generated.</div>';
                }
            } else {
                throw new Error(data.error || 'Server error');
            }
        } catch (error) {
            console.error('Manual recommend error:', error);
            recommendationsContainer.innerHTML = `<div class="empty-state" style="color: var(--error);">Error: ${error.message}</div>`;
            logToTerminal(`Recommendation error: ${error.message}`, 'error');
        } finally {
            recommendBtn.disabled = false;
            recommendBtn.querySelector('.btn-text').textContent = 'Get Recommendations';
            updateRecommendationControls();
        }
    });

    // Handle File Upload and Recommendations
    csvFileInput.addEventListener('change', () => {
        updateRecommendationControls();
    });

    uploadBtn.addEventListener('click', async () => {
        const file = csvFileInput.files[0];
        if (!file) return;

        uploadBtn.disabled = true;
        uploadBtn.querySelector('.btn-text').textContent = 'Processing...';
        recommendationsContainer.innerHTML = '<div class="empty-state">Processing your file...</div>';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                recommendationsContainer.innerHTML = '';
                if (data.results && data.results.length > 0) {
                    data.results.forEach((res, index) => {
                        const itemDiv = document.createElement('div');
                        itemDiv.className = 'recommendation-item';
                        itemDiv.innerHTML = `
                            <div class="rec-sequence">Sequence ${index + 1}: [${res.input_sequence.join(', ')}]</div>
                            <div class="rec-prediction">Top Recommendations: ${res.recommendations.join(', ')}</div>
                        `;
                        recommendationsContainer.appendChild(itemDiv);
                    });
                    logToTerminal(`Generated recommendations for ${data.results.length} sequences.`, 'success');
                } else {
                    recommendationsContainer.innerHTML = '<div class="empty-state">No valid sequences found in file.</div>';
                }
            } else {
                throw new Error(data.error || 'Server error');
            }
        } catch (error) {
            console.error('Upload error:', error);
            recommendationsContainer.innerHTML = `<div class="empty-state" style="color: var(--error);">Error: ${error.message}</div>`;
            logToTerminal(`Recommendation error: ${error.message}`, 'error');
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.querySelector('.btn-text').textContent = 'Get Recommendations';
            updateRecommendationControls();
        }
    });
});
