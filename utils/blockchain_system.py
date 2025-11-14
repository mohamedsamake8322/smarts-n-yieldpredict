
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any
import uuid

class BlockchainRecord:
    def __init__(self, data: Dict[str, Any], previous_hash: str = ""):
        self.timestamp = datetime.now().isoformat()
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the block"""
        block_string = json.dumps({
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 2):
        """Simple proof of work mining"""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

class AgriculturalBlockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 2
        self.pending_transactions = []
        self.certified_farms = {}
        self.product_records = {}
    
    def create_genesis_block(self) -> BlockchainRecord:
        """Create the first block in the chain"""
        genesis_data = {
            'type': 'genesis',
            'message': 'Agricultural Traceability Blockchain Genesis Block',
            'created_by': 'Agricultural Analytics Platform'
        }
        return BlockchainRecord(genesis_data)
    
    def get_latest_block(self) -> BlockchainRecord:
        """Get the most recent block"""
        return self.chain[-1]
    
    def add_block(self, data: Dict[str, Any]) -> BlockchainRecord:
        """Add a new block to the chain"""
        previous_block = self.get_latest_block()
        new_block = BlockchainRecord(data, previous_block.hash)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        return new_block
    
    def register_farm(self, farm_data: Dict[str, Any]) -> str:
        """Register a farm on the blockchain"""
        farm_id = str(uuid.uuid4())
        
        registration_data = {
            'type': 'farm_registration',
            'farm_id': farm_id,
            'farm_name': farm_data.get('name', ''),
            'location': farm_data.get('location', ''),
            'owner': farm_data.get('owner', ''),
            'size_hectares': farm_data.get('size', 0),
            'certification_type': farm_data.get('certification', 'conventional'),
            'registration_date': datetime.now().isoformat()
        }
        
        block = self.add_block(registration_data)
        self.certified_farms[farm_id] = {
            'data': registration_data,
            'block_hash': block.hash,
            'verified': False
        }
        
        return farm_id
    
    def create_crop_record(self, crop_data: Dict[str, Any]) -> str:
        """Create a crop cultivation record"""
        crop_id = str(uuid.uuid4())
        
        crop_record = {
            'type': 'crop_record',
            'crop_id': crop_id,
            'farm_id': crop_data.get('farm_id', ''),
            'crop_type': crop_data.get('crop_type', ''),
            'variety': crop_data.get('variety', ''),
            'planting_date': crop_data.get('planting_date', ''),
            'area_planted': crop_data.get('area', 0),
            'seeds_source': crop_data.get('seeds_source', ''),
            'organic_certified': crop_data.get('organic', False),
            'gmo_free': crop_data.get('gmo_free', True),
            'created_date': datetime.now().isoformat()
        }
        
        block = self.add_block(crop_record)
        self.product_records[crop_id] = {
            'data': crop_record,
            'block_hash': block.hash,
            'treatments': [],
            'harvest_data': None
        }
        
        return crop_id
    
    def add_treatment_record(self, crop_id: str, treatment_data: Dict[str, Any]) -> bool:
        """Add treatment (fertilizer, pesticide, etc.) record to a crop"""
        if crop_id not in self.product_records:
            return False
        
        treatment_record = {
            'type': 'treatment_record',
            'crop_id': crop_id,
            'treatment_type': treatment_data.get('type', ''),
            'product_name': treatment_data.get('product', ''),
            'application_date': treatment_data.get('date', ''),
            'amount_applied': treatment_data.get('amount', 0),
            'application_method': treatment_data.get('method', ''),
            'weather_conditions': treatment_data.get('weather', ''),
            'applicator': treatment_data.get('applicator', ''),
            'certification_compliant': treatment_data.get('compliant', True),
            'recorded_date': datetime.now().isoformat()
        }
        
        block = self.add_block(treatment_record)
        self.product_records[crop_id]['treatments'].append({
            'data': treatment_record,
            'block_hash': block.hash
        })
        
        return True
    
    def record_harvest(self, crop_id: str, harvest_data: Dict[str, Any]) -> bool:
        """Record harvest information"""
        if crop_id not in self.product_records:
            return False
        
        harvest_record = {
            'type': 'harvest_record',
            'crop_id': crop_id,
            'harvest_date': harvest_data.get('date', ''),
            'yield_amount': harvest_data.get('yield', 0),
            'quality_grade': harvest_data.get('quality', ''),
            'moisture_content': harvest_data.get('moisture', 0),
            'storage_location': harvest_data.get('storage', ''),
            'harvester': harvest_data.get('harvester', ''),
            'weather_at_harvest': harvest_data.get('weather', ''),
            'recorded_date': datetime.now().isoformat()
        }
        
        block = self.add_block(harvest_record)
        self.product_records[crop_id]['harvest_data'] = {
            'data': harvest_record,
            'block_hash': block.hash
        }
        
        return True
    
    def verify_product_authenticity(self, product_id: str) -> Dict[str, Any]:
        """Verify the authenticity and trace the history of a product"""
        if product_id not in self.product_records:
            return {'verified': False, 'error': 'Product not found'}
        
        product = self.product_records[product_id]
        
        # Verify blockchain integrity
        is_valid = self.is_chain_valid()
        
        # Get complete traceability data
        traceability_data = {
            'verified': is_valid,
            'product_id': product_id,
            'crop_data': product['data'],
            'treatments': [treatment['data'] for treatment in product['treatments']],
            'harvest_data': product['harvest_data']['data'] if product['harvest_data'] else None,
            'farm_info': None,
            'certification_status': 'verified' if is_valid else 'unverified'
        }
        
        # Get farm information
        farm_id = product['data'].get('farm_id')
        if farm_id in self.certified_farms:
            traceability_data['farm_info'] = self.certified_farms[farm_id]['data']
        
        return traceability_data
    
    def is_chain_valid(self) -> bool:
        """Validate the integrity of the blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block's hash is valid
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Check if current block points to previous block
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_environmental_score(self, product_id: str) -> Dict[str, Any]:
        """Calculate environmental impact score for certification"""
        if product_id not in self.product_records:
            return {'error': 'Product not found'}
        
        product = self.product_records[product_id]
        score = 100  # Start with perfect score
        
        # Deduct points for treatments
        for treatment in product['treatments']:
            treatment_data = treatment['data']
            if treatment_data['treatment_type'] == 'pesticide':
                score -= 10
            elif treatment_data['treatment_type'] == 'synthetic_fertilizer':
                score -= 5
            
        # Bonus for organic certification
        if product['data'].get('organic_certified', False):
            score += 20
        
        # Bonus for GMO-free
        if product['data'].get('gmo_free', True):
            score += 10
        
        score = max(0, min(100, score))  # Keep between 0-100
        
        return {
            'environmental_score': score,
            'certification_eligible': score >= 80,
            'premium_eligible': score >= 90,
            'recommendations': self._get_improvement_recommendations(score, product)
        }
    
    def _get_improvement_recommendations(self, score: int, product: Dict) -> List[str]:
        """Get recommendations for improving environmental score"""
        recommendations = []
        
        if score < 80:
            recommendations.append("Consider reducing synthetic pesticide usage")
            recommendations.append("Implement integrated pest management (IPM)")
            
        if score < 70:
            recommendations.append("Transition to organic fertilizers")
            recommendations.append("Implement cover cropping for soil health")
            
        if not product['data'].get('organic_certified', False):
            recommendations.append("Consider organic certification for premium pricing")
        
        return recommendations

# Global blockchain instance
agricultural_blockchain = AgriculturalBlockchain()
