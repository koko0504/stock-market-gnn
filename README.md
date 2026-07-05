# stock-market-gnn
Temporal GCN for S&amp;P 500 stock movement prediction using dynamic correlation graphs — 57.2% accuracy vs 49.4% static baseline

# Stock Market Prediction using Temporal GCN

> Temporal Graph Convolutional Network built from scratch in PyTorch  
> to predict next-week S&P 500 stock direction using dynamic correlation graphs.

## Results

| Model | Test Accuracy |
|---|---|
| Random baseline | ~50.00% |
| MLP (no graph) | ???% |
| Static GCN | 49.43% |
| **Temporal GCN (ours)** | **57.19%** |

## Architecture

## Key Finding

Dynamic weekly graphs outperform a static graph by **7.76 percentage points**  
— evidence that evolving graph topology carries predictive signal  
that static correlation structures miss.

## Project Structure

## Data

- 50 S&P 500 stocks across tech, finance, energy, healthcare, consumer sectors
- 6 years of daily price data (2018–2024) via yfinance
- Time-based 80/20 train/test split — no data leakage

## Stack

PyTorch · NumPy · pandas · yfinance · NetworkX · scikit-learn

## Built from scratch

No PyG (PyTorch Geometric) used — message passing, degree normalization,  
and A_norm computation implemented manually to demonstrate deep  
architectural understanding.
