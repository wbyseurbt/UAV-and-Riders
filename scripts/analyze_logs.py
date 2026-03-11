import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

def analyze_tensorboard_logs(logdir):
    # Find all event files recursively
    event_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    
    if not event_files:
        print(f"No event files found in {logdir}")
        return

    print(f"Found {len(event_files)} event files.")
    
    # Sort by modification time to get the latest run
    event_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = event_files[0]
    print(f"Analyzing latest file: {latest_file}")

    ea = EventAccumulator(latest_file)
    ea.Reload()

    # Get all available tags (scalars)
    tags = ea.Tags()['scalars']
    
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        # Extract values and steps
        values = [e.value for e in events]
        steps = [e.step for e in events]
        
        if values:
            data[tag] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "last": values[-1],
                "first": values[0],
                "count": len(values)
            }

    # Print Report
    print("\n" + "="*50)
    print("  TENSORBOARD ANALYSIS REPORT  ")
    print("="*50)
    
    # Group by category for better readability
    categories = {
        "Rewards": ["reward", "ep_rew_mean"],
        "Losses": ["loss", "policy_gradient_loss", "value_loss"],
        "Reward Components": ["active_order", "delivered", "wait", "overtime", "uav", "overflow", "handoff"],
        "System Status": ["active_orders_count", "delivered_count"] # If available
    }

    # Print categorized data
    for cat, keywords in categories.items():
        print(f"\n--- {cat} ---")
        found = False
        for tag, stats in data.items():
            if any(k in tag for k in keywords):
                print(f"{tag:<40} | Last: {stats['last']:<10.4f} | Mean: {stats['mean']:<10.4f} | Min/Max: {stats['min']:.2f}/{stats['max']:.2f}")
                found = True
        if not found:
            print("(No data found)")

    # Print remaining tags
    print("\n--- Other Metrics ---")
    used_tags = set().union(*categories.values())
    for tag, stats in data.items():
        if not any(k in tag for k in used_tags):
             print(f"{tag:<40} | Last: {stats['last']:<10.4f} | Mean: {stats['mean']:<10.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./logs", help="Path to logs directory")
    args = parser.parse_args()
    
    analyze_tensorboard_logs(args.logdir)
