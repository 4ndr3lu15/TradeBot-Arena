from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class TradingPair(Enum):
    ABC_XYZ = "ABC/XYZ"
    ABC_OOO = "ABC/OOO"
    XYZ_OOO = "XYZ/OOO"

@dataclass
class Order:
    agent_id: int
    pair: TradingPair
    is_buy: bool
    price: float
    amount: float
    timestamp: int
    order_id: int

class Agent:
    def __init__(self, agent_id: int, initial_balance: float = 1000):
        self.agent_id = agent_id
        self.balances = {
            "ABC": initial_balance,
            "XYZ": initial_balance,
            "OOO": initial_balance
        }
        self.active_orders: List[Order] = []
        
    def can_afford(self, asset: str, amount: float) -> bool:
        return self.balances[asset] >= amount

    def get_net_worth(self, prices: Dict[TradingPair, float]) -> float:
        worth = self.balances["OOO"]
        worth += self.balances["ABC"] * prices[TradingPair.ABC_OOO]
        worth += self.balances["XYZ"] * prices[TradingPair.XYZ_OOO]
        return worth

def run_simulation(
    n_agents: int = 10,
    n_steps: int = 1000,
    payment_interval: int = 100,
    reward_interval: int = 50,
    alpha: float = 0.95,
    initial_balance: float = 1000
):
    # Initialize agents and environment state
    agents = [Agent(i, initial_balance) for i in range(n_agents)]
    active_agents = set(range(n_agents))
    order_books = {pair: [] for pair in TradingPair}
    order_counter = 0
    current_step = 0
    
    # Track statistics
    price_history = {pair: [1.0] for pair in TradingPair}  # Start all prices at 1.0
    agent_history = {
        'active_agents': [n_agents],
        'total_ooo': [n_agents * initial_balance],
        'net_worth': [initial_balance * 3 * n_agents]  # ABC + XYZ + OOO
    }
    trade_history = []

    # Simulation loop
    while current_step < n_steps and len(active_agents) > 1:
        current_step += 1
        
        # Process OOO payments
        if current_step % payment_interval == 0:
            for agent_id in list(active_agents):
                if agents[agent_id].balances["OOO"] < 100:
                    active_agents.remove(agent_id)
                else:
                    agents[agent_id].balances["OOO"] -= 100

        # Process OOO rewards
        if current_step % reward_interval == 0:
            n = current_step // reward_interval
            reward = 100 * (alpha ** n)
            for agent_id in active_agents:
                agents[agent_id].balances["OOO"] += reward

        # Random actions for each active agent
        for agent_id in active_agents:
            action_type = np.random.choice(['place', 'fill', 'cancel'])
            
            if action_type == 'place':
                # Random order placement
                pair = np.random.choice(list(TradingPair))
                is_buy = np.random.choice([True, False])
                base_price = price_history[pair][-1]
                price = base_price * np.random.uniform(0.8, 1.2)  # ±20% from last price
                amount = np.random.uniform(0, 100)
                
                # Check if agent can afford the order
                agent = agents[agent_id]
                base, quote = pair.value.split('/')
                
                if is_buy:
                    if agent.can_afford(quote, amount * price):
                        order = Order(agent_id, pair, is_buy, price, amount, current_step, order_counter)
                        order_books[pair].append(order)
                        agent.active_orders.append(order)
                        order_counter += 1
                else:
                    if agent.can_afford(base, amount):
                        order = Order(agent_id, pair, is_buy, price, amount, current_step, order_counter)
                        order_books[pair].append(order)
                        agent.active_orders.append(order)
                        order_counter += 1
                        
            elif action_type == 'fill':
                # Try to fill a random order
                pair = np.random.choice(list(TradingPair))
                orders = [o for o in order_books[pair] if o.agent_id != agent_id]
                
                if orders:
                    order = np.random.choice(orders)
                    maker = agents[order.agent_id]
                    taker = agents[agent_id]
                    base, quote = order.pair.value.split('/')
                    
                    if order.is_buy:
                        if taker.can_afford(base, order.amount):
                            # Execute trade
                            taker.balances[base] -= order.amount
                            taker.balances[quote] += order.amount * order.price
                            maker.balances[base] += order.amount
                            maker.balances[quote] -= order.amount * order.price
                            
                            # Update price history
                            price_history[pair].append(order.price)
                            
                            # Record trade
                            trade_history.append({
                                'step': current_step,
                                'pair': pair,
                                'price': order.price,
                                'amount': order.amount
                            })
                            
                            # Remove filled order
                            order_books[pair].remove(order)
                            maker.active_orders.remove(order)
                    else:
                        if taker.can_afford(quote, order.amount * order.price):
                            # Execute trade
                            taker.balances[base] += order.amount
                            taker.balances[quote] -= order.amount * order.price
                            maker.balances[base] -= order.amount
                            maker.balances[quote] += order.amount * order.price
                            
                            # Update price history
                            price_history[pair].append(order.price)
                            
                            # Record trade
                            trade_history.append({
                                'step': current_step,
                                'pair': pair,
                                'price': order.price,
                                'amount': order.amount
                            })
                            
                            # Remove filled order
                            order_books[pair].remove(order)
                            maker.active_orders.remove(order)
            
            else:  # cancel
                agent = agents[agent_id]
                if agent.active_orders:
                    order = np.random.choice(agent.active_orders)
                    order_books[order.pair].remove(order)
                    agent.active_orders.remove(order)
        
        # Record statistics
        total_ooo = sum(agent.balances["OOO"] for agent in agents)
        total_net_worth = sum(agent.get_net_worth(
            {pair: price_history[pair][-1] for pair in TradingPair}
        ) for agent in agents)
        
        agent_history['active_agents'].append(len(active_agents))
        agent_history['total_ooo'].append(total_ooo)
        agent_history['net_worth'].append(total_net_worth)
        
        # Ensure price history has an entry for this step
        for pair in TradingPair:
            if len(price_history[pair]) <= current_step:
                price_history[pair].append(price_history[pair][-1])
    
    return {
        'price_history': price_history,
        'agent_history': agent_history,
        'trade_history': trade_history,
        'final_state': {
            'agents': agents,
            'active_agents': active_agents,
            'steps_completed': current_step
        }
    }

def plot_simulation_results(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot prices
    for pair, prices in results['price_history'].items():
        ax1.plot(prices, label=pair.value)
    ax1.set_title('Asset Prices')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Plot active agents
    ax2.plot(results['agent_history']['active_agents'])
    ax2.set_title('Active Agents')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Number of Agents')
    
    # Plot total OOO
    ax3.plot(results['agent_history']['total_ooo'])
    ax3.set_title('Total OOO in System')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Amount')
    
    # Plot total net worth
    ax4.plot(results['agent_history']['net_worth'])
    ax4.set_title('Total Net Worth')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Amount')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run simulation
    print("Running simulation...")
    results = run_simulation(
        n_agents=100,
        n_steps=1500,
        payment_interval=100,
        reward_interval=50,
        alpha=0.9,
        initial_balance=1000
    )
    
    # Print summary
    print("\nSimulation completed!")
    print(f"Steps completed: {results['final_state']['steps_completed']}")
    print(f"Surviving agents: {len(results['final_state']['active_agents'])}")
    print(f"Total trades: {len(results['trade_history'])}")
    print("\nFinal prices:")
    for pair in TradingPair:
        print(f"{pair.value}: {results['price_history'][pair][-1]:.4f}")
    
    # Plot results
    print("\nPlotting results...")
    plot_simulation_results(results)