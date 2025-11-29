"""
Blockchain Reward System Integration for Collatz AI Training

Connects training progress to Ethereum smart contract rewards.
Users earn SCR tokens for contributing computational power to training.
"""

from web3 import Web3
import json
import os

# Contract ABIs
SCIENTIST_REWARD_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "to", "type": "address"}, 
                   {"internalType": "uint256", "name": "amount", "type": "uint256"}],
        "name": "mint",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

REWARD_DISTRIBUTOR_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "user", "type": "address"},
                   {"internalType": "uint256", "name": "steps", "type": "uint256"}],
        "name": "addSteps",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "userSteps",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]

class BlockchainRewardSystem:
    def __init__(self, rpc_url, contract_address, private_key, user_wallet):
        """
        Initialize blockchain reward system
        
        Args:
            rpc_url: Ethereum RPC endpoint (e.g., Infura, Alchemy)
            contract_address: RewardDistributor contract address
            private_key: Private key for signing transactions
            user_wallet: User's wallet address to receive rewards
        """
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.user_wallet = Web3.to_checksum_address(user_wallet)
        self.private_key = private_key
        
        # Load contract
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=REWARD_DISTRIBUTOR_ABI
        )
        
        # Account from private key
        self.account = self.w3.eth.account.from_key(private_key)
        
        print(f"✅ Blockchain Reward System initialized")
        print(f"   Contract: {self.contract_address}")
        print(f"   User Wallet: {self.user_wallet}")
        print(f"   Network: {self.w3.eth.chain_id}")
    
    def submit_training_steps(self, steps):
        """
        Submit training steps to smart contract
        
        Args:
            steps: Number of training steps completed
        
        Returns:
            Transaction hash if successful, None otherwise
        """
        try:
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            tx = self.contract.functions.addSteps(
                self.user_wallet,
                steps
            ).build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            print(f"✅ Submitted {steps} steps to blockchain")
            print(f"   TX Hash: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                print(f"✅ Transaction confirmed! Block: {receipt['blockNumber']}")
                return tx_hash.hex()
            else:
                print(f"❌ Transaction failed!")
                return None
                
        except Exception as e:
            print(f"❌ Error submitting steps: {e}")
            return None
    
    def get_user_steps(self):
        """Get user's current step count from contract"""
        try:
            steps = self.contract.functions.userSteps(self.user_wallet).call()
            return steps
        except Exception as e:
            print(f"❌ Error getting user steps: {e}")
            return 0
    
    def get_estimated_rewards(self, steps):
        """
        Calculate estimated SCR tokens for given steps
        
        Args:
            steps: Number of training steps
        
        Returns:
            Estimated SCR tokens (in ether units)
        """
        # 0.1 SCR per 1000 steps
        return (steps / 1000) * 0.1


# Configuration loader
def load_blockchain_config():
    """Load blockchain configuration from file"""
    config_path = "blockchain_config.json"
    
    if not os.path.exists(config_path):
        print("⚠️  No blockchain_config.json found. Rewards disabled.")
        print("💡 Create blockchain_config.json with your settings.")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return BlockchainRewardSystem(
        rpc_url=config['rpc_url'],
        contract_address=config['contract_address'],
        private_key=config['private_key'],
        user_wallet=config['user_wallet']
    )


# Example usage
if __name__ == "__main__":
    # Test configuration
    config = {
        "rpc_url": "https://mainnet.infura.io/v3/YOUR_INFURA_KEY",
        "contract_address": "0x...",  # Your RewardDistributor address
        "private_key": "0x...",  # Your private key (KEEP SECRET!)
        "user_wallet": "0x70b16C2b89C6d306104A77013EA53E26f9943C33"
    }
    
    # Initialize
    reward_system = BlockchainRewardSystem(**config)
    
    # Submit 5000 steps (should mint 0.5 SCR)
    reward_system.submit_training_steps(5000)
    
    # Check balance
    steps = reward_system.get_user_steps()
    print(f"Current steps: {steps}")
    print(f"Estimated rewards: {reward_system.get_estimated_rewards(steps)} SCR")
