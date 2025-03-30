from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO
import numpy as np
import os
import imageio
import multiprocessing as mp
import matplotlib.pyplot as plt
from env import LoadBalancer

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, async_mode='eventlet')
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Smooth line function
def smooth_line(x, y, num_points=300):
    x_new = np.linspace(min(x), max(x), num_points)
    return x_new, np.interp(x_new, x, y)

# Function to generate a single frame
def generate_frame(i, time_steps, demand, storage, rewards):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#121212')
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    if i > 1:
        x_new, demand_smooth = smooth_line(time_steps[:i], demand[:i])
        x_new, storage_smooth = smooth_line(time_steps[:i], storage[:i])
        x_new, rewards_smooth = smooth_line(time_steps[:i], rewards[:i])

        plt.plot(x_new, demand_smooth, label='Demand', color='cyan')
        plt.plot(x_new, storage_smooth, label='Storage', color='magenta')
        plt.plot(x_new, rewards_smooth, label='Rewards', color='yellow')

    plt.title('Smart Grid Load Balancing', color='white')
    plt.xlabel('Time Step', color='white')
    plt.ylabel('Value', color='white')
    plt.legend(title='Metric', facecolor='#C0BDBD')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    frame_filename = f'{OUTPUT_FOLDER}/frame_{i:03d}.png'
    plt.savefig(frame_filename, facecolor='#121212')
    plt.close()
    return frame_filename

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('run_simulation')
def run_simulation(data):
    NUM_STEPS = int(data['num_requests'])

    env = LoadBalancer()
    obs = env.reset()
    time_steps, demand, storage, rewards = [], [], [], []

    for step in range(NUM_STEPS):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        time_steps.append(step)
        demand.append(env.demand_history[-1])
        storage.append(env.storage_history[-1])
        rewards.append(env.rewards_history[-1])

        # Send live update to UI
        socketio.emit('update_chart', {
            'time': step,
            'demand': demand[-1],
            'storage': storage[-1],
            'rewards': rewards[-1]
        })

    # Generate GIF in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        frames = pool.starmap(generate_frame, [(i, time_steps, demand, storage, rewards) for i in range(1, len(time_steps) + 1)])

    gif_file = f'{OUTPUT_FOLDER}/sim_results.gif'
    with imageio.get_writer(gif_file, mode='I', duration=0.1) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
        for _ in range(10):
            writer.append_data(imageio.imread(frames[-1]))

    for frame in frames:
        os.remove(frame)

    # Send final GIF URL to UI
    socketio.emit('simulation_complete', {'gif_url': gif_file})

@app.route('/download_gif')
def download_gif():
    return send_file(f'{OUTPUT_FOLDER}/sim_results.gif', as_attachment=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
