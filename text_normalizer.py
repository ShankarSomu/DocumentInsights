import re
import json
import os
from typing import Dict, List, Set

class TextNormalizer:
    def __init__(self):
        self.mapping_file = "local_data/text_mappings.json"
        self.canonical_mappings = {}
        self._load_mappings()
        
        # Common normalization patterns
        self.normalization_patterns = [
            # Punctuation variations
            (r'[–—−]', '-'),  # Em dash, en dash to hyphen
            (r'\s*[-–—−]\s*', ' - '),  # Standardize dash spacing
            (r'\s*[/\\]\s*', ' / '),  # Standardize slash spacing
            (r'\s*:\s*', ': '),  # Standardize colon spacing
            
            # Common phrase variations
            (r'\bfor\s+project\b', 'for Project'),
            (r'\bproject\s+charter\b', 'Project Charter'),
            (r'\bproject\s+plan\b', 'Project Plan'),
            
            # Multiple spaces to single
            (r'\s+', ' '),
        ]
        
        # Semantic equivalents
        self.semantic_equivalents = {
            'for': ['–', '—', '-', 'of', 'regarding'],
            'project': ['proj', 'prj'],
            'charter': ['document', 'doc'],
            'plan': ['planning', 'strategy'],
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent matching"""
        
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase for processing
        normalized = text.strip().lower()
        
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def get_canonical_form(self, text: str) -> str:
        """Get canonical form of text, creating mapping if needed"""
        
        normalized = self.normalize_text(text)
        
        # Check if we already have a canonical mapping
        if normalized in self.canonical_mappings:
            return self.canonical_mappings[normalized]
        
        # Find similar existing canonical forms
        canonical_form = self._find_similar_canonical(normalized)
        
        if not canonical_form:
            # Create new canonical form
            canonical_form = self._create_canonical_form(text)
        
        # Store the mapping
        self.canonical_mappings[normalized] = canonical_form
        self._save_mappings()
        
        return canonical_form
    
    def _find_similar_canonical(self, normalized_text: str) -> str:
        """Find similar existing canonical forms"""
        
        # Calculate similarity with existing canonical forms
        best_match = None
        best_score = 0.0
        
        for existing_normalized, canonical in self.canonical_mappings.items():
            similarity = self._calculate_similarity(normalized_text, existing_normalized)
            
            if similarity > 0.85 and similarity > best_score:  # High threshold for auto-mapping
                best_score = similarity
                best_match = canonical
        
        return best_match
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        
        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Direct word overlap
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        base_similarity = intersection / union
        
        # Boost similarity for semantic equivalents
        semantic_boost = 0.0
        for word1 in words1:
            for word2 in words2:
                if self._are_semantically_equivalent(word1, word2):
                    semantic_boost += 0.2
        
        return min(1.0, base_similarity + semantic_boost)
    
    def _are_semantically_equivalent(self, word1: str, word2: str) -> bool:
        """Check if two words are semantically equivalent"""
        
        if word1 == word2:
            return True
        
        # Check semantic equivalents
        for base_word, equivalents in self.semantic_equivalents.items():
            if (word1 == base_word and word2 in equivalents) or \
               (word2 == base_word and word1 in equivalents) or \
               (word1 in equivalents and word2 in equivalents):
                return True
        
        return False
    
    def _create_canonical_form(self, original_text: str) -> str:
        """Create a canonical form from original text"""
        
        # Use title case and clean formatting
        canonical = original_text.strip()
        
        # Standardize common patterns
        canonical = re.sub(r'\s*[–—−]\s*', ' - ', canonical)
        canonical = re.sub(r'\s+', ' ', canonical)
        
        # Title case for better readability
        canonical = canonical.title()
        
        return canonical
    
    def add_manual_mapping(self, variations: List[str], canonical_form: str):
        """Manually add mappings for known variations"""
        
        for variation in variations:
            normalized = self.normalize_text(variation)
            self.canonical_mappings[normalized] = canonical_form
        
        self._save_mappings()
    
    def get_search_variations(self, text: str) -> List[str]:
        """Get all possible variations of text for search"""
        
        variations = [text]
        normalized = self.normalize_text(text)
        
        # Add normalized version
        if normalized != text.lower():
            variations.append(normalized)
        
        # Add canonical form
        canonical = self.get_canonical_form(text)
        if canonical not in variations:
            variations.append(canonical)
        
        # Add semantic variations
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in self.semantic_equivalents:
                for equivalent in self.semantic_equivalents[word]:
                    variant_words = words.copy()
                    variant_words[i] = equivalent
                    variations.append(' '.join(variant_words))
        
        return list(set(variations))
    
    def _load_mappings(self):
        """Load canonical mappings from file"""
        
        try:
            if os.path.exists(self.mapping_file):
                with open(self.mapping_file, 'r') as f:
                    self.canonical_mappings = json.load(f)
        except Exception as e:
            print(f"Error loading text mappings: {e}")
            self.canonical_mappings = {}
    
    def _save_mappings(self):
        """Save canonical mappings to file"""
        
        try:
            os.makedirs(os.path.dirname(self.mapping_file), exist_ok=True)
            with open(self.mapping_file, 'w') as f:
                json.dump(self.canonical_mappings, f, indent=2)
        except Exception as e:
            print(f"Error saving text mappings: {e}")
    
    def get_mapping_stats(self) -> Dict:
        """Get statistics about current mappings"""
        
        return {
            "total_mappings": len(self.canonical_mappings),
            "unique_canonical_forms": len(set(self.canonical_mappings.values())),
            "sample_mappings": dict(list(self.canonical_mappings.items())[:5])
        }