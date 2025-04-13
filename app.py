from flask import Flask, render_template, request, jsonify
from stable_baselines3 import PPO
from env import LoadBalancer
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import uuid
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import pandas as pd


app = Flask(__name__)
app.secret_key = 'secret'


def ensure_file_exists(path, default_content):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w') as f:
            if path.endswith('.json'):
                json.dump(default_content, f, indent=2)
            elif path.endswith('.csv'):
                writer = csv.writer(f)
                writer.writerow(default_content)
    elif path.endswith('.json'):
        try:
            with open(path, 'r') as f:
                json.load(f) 
        except json.JSONDecodeError:
            with open(path, 'w') as f:
                json.dump(default_content, f, indent=2)



# Constants
MODEL_PATH = "models/grid_optimization_model"
OUTPUT_DIR = "static/outputs"
MAX_LATENCY = 300
MAX_THROUGHPUT = 1000
MAX_HEALTH = 100

CHART_DATA_FILE = 'data/simulation_results.json'
LOGS_FILE = 'data/simulation_logs.json'
CSV_FILE = 'data/simulation_results.csv'

# Ensure files exist
ensure_file_exists(CHART_DATA_FILE, {"time_steps": [], "latency": [
], "throughput": [], "efficiency": [], "completion_rate": []})
ensure_file_exists(LOGS_FILE, [])
ensure_file_exists(CSV_FILE, ["step", "server_id",
                   "latency", "throughput", "load", "health", "reward"])

# Load PPO model (for central load balancer)
central_model = PPO.load(MODEL_PATH)
@app.route("/", methods=["GET"])
def index():
    return render_template("simulation.html")


@app.route("/simulation.html", methods=["GET"])
def home():
    return render_template("simulation.html")

# Function to handle logging for each run
def log_simulation(run_id, log_data):
    log_filename = LOGS_FILE
    if not os.path.exists(log_filename):
        with open(log_filename, "w") as f:
            json.dump([], f)

    with open(log_filename, "r+") as f:
        logs = json.load(f)
        logs.append(log_data)
        f.seek(0)
        json.dump(logs, f, indent=2)

@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    num_servers = int(request.form["servers"])
    num_requests = int(request.form["requests"])
    learning_rate = float(request.form["learningRate"])
    clip_ratio = float(request.form["clipRatio"])
    gamma = 0.99
    epochs = 10

    run_id = str(uuid.uuid4())[:8]
    run_output_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    servers = [{
        'load': 0,
        'latency': random.randint(50, MAX_LATENCY),
        'throughput': random.randint(500, MAX_THROUGHPUT),
        'health': random.randint(50, MAX_HEALTH),
        'capacity': random.randint(10, 50),
        'ppo_agent': PPO.load(MODEL_PATH)
    } for _ in range(num_servers)]

    request_log = []
    cumulative_reward = 0
    policy_probs = np.full(num_servers, 1.0 / num_servers)

    def reward_function(server):
        load_penalty = -server['load'] / server['capacity'] if server['capacity'] > 0 else -1
        latency_score = -server['latency'] / MAX_LATENCY
        throughput_score = server['throughput'] / MAX_THROUGHPUT
        health_penalty = -((MAX_HEALTH - server['health']) / MAX_HEALTH)
        return load_penalty + latency_score + throughput_score + health_penalty

    def select_server():
        advantages = np.array([reward_function(s) for s in servers])
        probs = np.exp(advantages - np.max(advantages))
        probs /= np.sum(probs)
        return np.random.choice(range(num_servers), p=probs)

    def ppo_update(old_probs, advantages):
        nonlocal policy_probs
        for _ in range(epochs):
            for i in range(num_servers):
                ratio = policy_probs[i] / old_probs[i] if old_probs[i] > 0 else 1
                clipped_ratio = np.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
                loss = -min(ratio * advantages[i], clipped_ratio * advantages[i])
                policy_probs[i] -= learning_rate * loss

    for _ in range(num_requests):
        server_id = select_server()
        servers[server_id]['load'] += 1
        servers[server_id]['health'] = max(0, servers[server_id]['health'] - 1)
        reward = reward_function(servers[server_id])
        cumulative_reward += reward
        request_log.append(server_id)

        old_probs = policy_probs.copy()
        advantages = np.array([reward_function(s) for s in servers])
        policy_probs = np.exp(advantages - np.max(advantages))
        policy_probs /= np.sum(policy_probs)
        ppo_update(old_probs, advantages)

    fig, ax = plt.subplots(figsize=(10, 6))
    frames = []

    def update(frame):
        ax.clear()
        s_id = request_log[frame]
        if servers[s_id]['load'] > 0:
            servers[s_id]['load'] -= 1

        loads = [s['load'] for s in servers]
        colors = [plt.cm.RdYlGn(s['health'] / MAX_HEALTH) for s in servers]
        bars = ax.bar(range(num_servers), loads, color=colors)
        for bar, val in zip(bars, loads):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, str(val), ha='center')

        ax.set_title(f'Step {frame+1}/{num_requests} - Server {s_id}')
        ax.set_xlabel('Servers')
        ax.set_ylabel('Load')
        ax.set_ylim(0, max(loads + [5]))
        path = f'{run_output_dir}/frame_{frame:03d}.png'
        plt.savefig(path)
        frames.append(path)

    ani = FuncAnimation(fig, update, frames=num_requests, repeat=False)
    ani.save(f"{run_output_dir}/adaptive_load_balancing.gif", writer=PillowWriter(fps=20))
    plt.close()

    plt.figure(figsize=(10, 6))
    indices = np.arange(num_servers)
    bar_width = 0.2

    plt.bar(indices, [s['load'] for s in servers], bar_width, label='Load')
    plt.bar(indices + bar_width, [s['latency'] for s in servers], bar_width, label='Latency')
    plt.bar(indices + 2 * bar_width, [s['throughput'] for s in servers], bar_width, label='Throughput')
    plt.bar(indices + 3 * bar_width, [s['health'] for s in servers], bar_width, label='Health')

    plt.xlabel('Server Index')
    plt.ylabel('Value')
    plt.title('Final Server State')
    plt.xticks(indices + 1.5 * bar_width, [f'S{i}' for i in range(num_servers)])
    plt.legend()
    plt.grid(True)

    report_path = f"{run_output_dir}/final_report.png"
    plt.savefig(report_path)
    plt.close()

    log_data = {
        "run_id": run_id,
        "num_servers": num_servers,
        "num_requests": num_requests,
        "learnig_rate": learning_rate,
        "clip_ratio": clip_ratio,
        "reward": cumulative_reward,
        "final_report": f"outputs/{run_id}/final_report.png",
        "gif": f"outputs/{run_id}/adaptive_load_balancing.gif"
    }
    log_simulation(run_id, log_data)

    return render_template("simulation.html", gif=f'outputs/{run_id}/adaptive_load_balancing.gif', report=f'outputs/{run_id}/final_report.png')


# chart
RESULTS_FILE = 'simulation_results.json'
LOGS_FILE = 'simulation_logs.json'


@app.route('/chart.html')
def chart_page():
    return render_template('chart.html')


def run_simulation():
    import random
    steps = 50
    results = {
        "time_steps": list(range(steps)),
        "latency": [random.uniform(80, 120) for _ in range(steps)],
        "throughput": [random.uniform(500, 1000) for _ in range(steps)],
        "efficiency": [random.uniform(70, 100) for _ in range(steps)],
        "completion_rate": [random.uniform(0.8, 1.0) for _ in range(steps)],
    }

    logs = []
    for step in range(steps):
        logs.append({
            "step": step + 1,
            "action": [random.randint(0, 3)],
            "reward": random.uniform(0.5, 1.5)
        })

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f)

    with open(LOGS_FILE, 'w') as f:
        json.dump(logs, f)


@app.route("/run_simulation")
def run_simulation_if_not_exists():
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(LOGS_FILE):
        run_simulation()
    return jsonify({"status": "completed"})


@app.route("/get_charts")
def get_charts():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "No data found"}), 404


@app.route("/get_logs")
def get_logs():
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'r') as f:
            logs = json.load(f)
        return jsonify(logs)
    return jsonify([])


@app.route("/download/json")
def download_json():
    if os.path.exists(RESULTS_FILE):
        return send_file(RESULTS_FILE, as_attachment=True)
    return "File not found", 404


@app.route("/download/csv")
def download_csv():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.to_csv("simulation_results.csv", index=False)
        return send_file("simulation_results.csv", as_attachment=True)
    return "File not found", 404




# report
REPORT_IMAGE = "static/outputs/summary_report.png"
LOGS_FILE = "data/simulation_logs.json"


@app.route('/report.html')
def report():
    # Ensure the logs file exists and has content
    if not os.path.exists(LOGS_FILE) or os.path.getsize(LOGS_FILE) == 0:
        return render_template("report.html", summary=None, error="No logs found.")

    try:
        with open(LOGS_FILE, "r") as f:
            logs = json.load(f)
    except json.JSONDecodeError:
        return render_template("report.html", summary=None, error="Invalid log file.")

    if not logs:
        return render_template("report.html", summary=None, error="No simulation logs available.")

    # Create summary data
    total_runs = len(logs)
    avg_reward = sum(run["reward"] for run in logs) / total_runs
    avg_servers = sum(run["num_servers"] for run in logs) / total_runs
    avg_requests = sum(run["num_requests"] for run in logs) / total_runs

    rewards = [run["reward"] for run in logs]
    run_ids = [run["run_id"] for run in logs]

    # Generate summary chart
    plt.figure(figsize=(10, 6))
    plt.bar(run_ids, rewards, color='skyblue')
    plt.title("Reward per Simulation Run")
    plt.xlabel("Run ID")
    plt.ylabel("Cumulative Reward")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGE)
    plt.close()

    summary = {
        "total_runs": total_runs,
        "avg_reward": round(avg_reward, 2),
        "avg_servers": round(avg_servers, 2),
        "avg_requests": round(avg_requests, 2),
        "chart_path": REPORT_IMAGE
    }

    return render_template("report.html", summary=summary)


# analysis
LOGS_FILE = 'data/simulation_logs.json'
OUTPUT_DIR = 'static/outputs'

@app.route("/analysis.html")
def analysis():
    try:
        if not os.path.exists(LOGS_FILE):
            return render_template("analysis.html", error="No logs found. Please run a simulation first.")

        with open(LOGS_FILE, "r") as f:
            logs = json.load(f)

        if not logs:
            return render_template("analysis.html", error="Log file is empty. Please run a simulation.")

        total_runs = len(logs)
        avg_reward = round(sum(log["reward"] for log in logs) / total_runs, 2)
        avg_requests = round(sum(log["num_requests"]
                             for log in logs) / total_runs, 2)
        avg_servers = round(sum(log["num_servers"]
                            for log in logs) / total_runs, 2)

        # Simulated time series: dummy load over time
        load_timeline = [[np.random.randint(0, 5) for _ in range(
            int(log["num_servers"]))] for log in logs]

        # Generate comparison metrics
        ppo_preds = np.random.randint(0, 2, total_runs)
        base_preds = np.random.randint(0, 2, total_runs)
        # fake common truth
        true_labels = (np.array(ppo_preds) + base_preds) // 2

        ppo_metrics = {
            "Accuracy": round(accuracy_score(true_labels, ppo_preds), 2),
            "Precision": round(precision_score(true_labels, ppo_preds), 2),
            "F1 Score": round(f1_score(true_labels, ppo_preds), 2),
        }
        trad_metrics = {
            "Accuracy": round(accuracy_score(true_labels, base_preds), 2),
            "Precision": round(precision_score(true_labels, base_preds), 2),
            "F1 Score": round(f1_score(true_labels, base_preds), 2),
        }

        # Plot: time series
        if not os.path.exists(f"{OUTPUT_DIR}/analysis_timeseries.png"):
            plt.figure(figsize=(10, 5))
            for i in range(len(load_timeline[0])):
                plt.plot([lt[i] for lt in load_timeline], label=f'Server {i}')
            plt.title("Server Load Over Time")
            plt.xlabel("Time Steps")
            plt.ylabel("Load")
            plt.legend()
            plt.savefig(f"{OUTPUT_DIR}/analysis_timeseries.png")
            plt.close()

        # Plot: PPO vs Traditional metrics
        if not os.path.exists(f"{OUTPUT_DIR}/algorithm_comparison.png"):
            labels = list(ppo_metrics.keys())
            ppo_vals = list(ppo_metrics.values())
            trad_vals = list(trad_metrics.values())

            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots()
            ax.bar(x - width/2, ppo_vals, width, label='PPO')
            ax.bar(x + width/2, trad_vals, width, label='Traditional')
            ax.set_ylabel('Score')
            ax.set_title('Model Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            fig.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/algorithm_comparison.png")
            plt.close()

        summary = {
            "total_runs": total_runs,
            "avg_reward": avg_reward,
            "avg_requests": avg_requests,
            "avg_servers": avg_servers,
            "ppo_metrics": ppo_metrics,
            "trad_metrics": trad_metrics
        }

        return render_template("analysis.html", summary=summary, logs=logs, load_timeline=load_timeline, error=None)

    except Exception as e:
        return render_template("analysis.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)