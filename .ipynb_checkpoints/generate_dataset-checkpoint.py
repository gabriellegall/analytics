import numpy as np
import pandas as pd

def generate_dataset(num_records=1000, seed=42):
    np.random.seed(seed)

    # Generate member IDs
    member_ids = np.arange(1, num_records + 1)

    # Generate skewed balances: most are around [0, 10] or [2000, 100000]
    balance_small = np.random.exponential(scale=5, size=int(num_records * 0.7))  # small balances
    balance_large = np.random.exponential(scale=20000, size=int(num_records * 0.3)) + 2000  # large balances
    balances = np.concatenate([balance_small, balance_large])
    np.random.shuffle(balances)

    # Generate skewed card transactions: most are [0, 15] or >100
    card_small = np.random.poisson(lam=5, size=int(num_records * 0.7))  # small transactions
    card_large = np.random.poisson(lam=150, size=int(num_records * 0.3))  # large transactions
    card_transactions = np.concatenate([card_small, card_large])
    np.random.shuffle(card_transactions)

    # Ensure correlation between balances and card transactions
    correlation_factor = 0.8
    card_transactions = (
        card_transactions * (1 - correlation_factor) + balances / 1000 * correlation_factor
    ).astype(int)
    card_transactions = np.clip(card_transactions, 0, 250)

    # Determine churn probability based on balances and card transactions
    churn_probs = 1 / (1 + np.exp(-0.01 * (10_000 - balances) - 0.05 * (50 - card_transactions)))
    churn = np.random.choice(["Y", "N"], size=num_records, p=[churn_probs.mean(), 1 - churn_probs.mean()])

    # Create DataFrame
    df = pd.DataFrame({
        "member_id": member_ids,
        "nb_card_transactions_last_90d": card_transactions,
        "average_balance_last_90d": balances,
        "has_churned": churn
    })

    return df