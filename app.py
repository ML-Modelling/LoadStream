from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import random
import os
import imageio
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static', template_folder='templates')
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MAX_LATENCY = 300
MAX_THROUGHPUT = 1000
MAX_HEALTH = 100

def reward_function(server):
    load_penalty = -server['load'] / max(1, server['capacity'])
    latency_score = -server['latency'] / MAX_LATENCY
    throughput_score = server['throughput'] / MAX_THROUGHPUT
    health_penalty = -((MAX_HEALTH - server['health']) / MAX_HEALTH)
    response_penalty = -server['response_time'] / 100
    failure_penalty = -server['failure_rate']
    return load_penalty + latency_score + throughput_score + health_penalty + response_penalty + failure_penalty

def select_server(servers):
    advantages = np.array([reward_function(server) for server in servers])
    exp_advantages = np.exp(advantages - np.max(advantages))
    probs = exp_advantages / np.sum(exp_advantages)
    return np.random.choice(range(len(servers)), p=probs)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json
    NUM_SERVERS = int(data['num_servers'])
    NUM_REQUESTS = int(data['num_requests'])

    servers = [{
        'load': 0, 'latency': random.randint(50, MAX_LATENCY),
        'throughput': random.randint(500, MAX_THROUGHPUT), 'health': random.randint(50, MAX_HEALTH),
        'capacity': random.randint(10, 50), 'response_time': random.randint(10, 100),
        'failure_rate': random.uniform(0, 0.1)
    } for _ in range(NUM_SERVERS)]
    
    frames = []
    cumulative_reward = 0
    
    for i in range(NUM_REQUESTS):
        selected_server = select_server(servers)
        servers[selected_server]['load'] += 1
        servers[selected_server]['health'] = max(0, servers[selected_server]['health'] - 1)
        
        reward = reward_function(servers[selected_server])
        cumulative_reward += reward
        
        fig, ax = plt.subplots()
        ax.bar(range(NUM_SERVERS), [server['load'] for server in servers], color='blue')
        ax.set_title(f'Request {i+1}/{NUM_REQUESTS} - Server {selected_server}')
        frame_file = f'{OUTPUT_FOLDER}/frame_{i:03d}.png'
        plt.savefig(frame_file)
        frames.append(frame_file)
        plt.close()
    
    gif_file = f'{OUTPUT_FOLDER}/simulation.gif'
    with imageio.get_writer(gif_file, mode='I', duration=0.1) as gif_writer:
        for frame in frames:
            image = imageio.imread(frame)
            gif_writer.append_data(image)
    
    for frame in frames:
        os.remove(frame)
    
    return jsonify({'message': 'Simulation complete!', 'gif_url': gif_file})

@app.route('/download_gif')
def download_gif():
    return send_file(f'{OUTPUT_FOLDER}/simulation.gif', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
