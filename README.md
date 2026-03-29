# Supply Chain Agentic AI - OpenEnv Hackathon

This repository contains an autonomous agent designed to manage supply chain inventory levels using the OpenEnv framework.

## Project Structure
- `envs/sc_env/`: Custom Supply Chain environment.
- `inference.py`: Refined agent logic with dynamic ordering.
- `Dockerfile`: Container configuration for Hugging Face Spaces.

## How it Works
The agent monitors `p_ord` (Pending Orders) and adjusts `p_qty` (Purchase Quantity) to maintain optimal capacity without overstocking.
