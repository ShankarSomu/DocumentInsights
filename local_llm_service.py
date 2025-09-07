from transformers import pipeline
from typing import List, Dict
from local_vector_store import LocalVectorStore
from models import ChatResponse

class LocalLLMService:
    def __init__(self):
        self.vector_store = LocalVectorStore()
        # Use a better model for question answering
        try:
            self.llm = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",  # Smaller, faster
                device=-1  # CPU
            )
        except:
            # Fallback to simple text processing if model fails
            self.llm = None
            print("Using fallback text processing (no LLM)")
        print("Local LLM loaded successfully")
    
    def generate_answer(self, question: str, user_id: str) -> ChatResponse:
        # Retrieve relevant context
        comprehensive_keywords = ['all', 'list', 'show', 'total', 'count', 'every', 'complete']
        needs_all_data = any(keyword in question.lower() for keyword in comprehensive_keywords)
        
        if needs_all_data:
            relevant_chunks = self.vector_store.search(question, user_id, top_k=50)
        else:
            relevant_chunks = self.vector_store.search(question, user_id, top_k=10)
        
        if not relevant_chunks:
            return ChatResponse(
                answer="I don't have enough data to answer that question. Please upload some files first.",
                sources=[]
            )
        
        # Build context
        context = self._build_context(relevant_chunks)
        
        # Generate answer using local LLM
        prompt = f"Based on this project data, answer the question: {question}\n\nData: {context}\n\nAnswer:"
        
        if self.llm:
            try:
                # Truncate prompt if too long
                if len(prompt) > 1000:
                    prompt = prompt[:1000] + "..."
                
                response = self.llm(prompt, max_length=len(prompt) + 150, num_return_sequences=1, temperature=0.3)
                answer = response[0]['generated_text'][len(prompt):].strip()
                
                # Fallback to simple data extraction if LLM fails
                if not answer or len(answer) < 10:
                    answer = self._extract_simple_answer(question, relevant_chunks)
                
            except Exception as e:
                print(f"LLM generation failed: {e}")
                answer = self._extract_simple_answer(question, relevant_chunks)
        else:
            # Use rule-based extraction
            answer = self._extract_simple_answer(question, relevant_chunks)
        
        sources = list(set([chunk.get('file_name', 'Unknown') for chunk in relevant_chunks]))
        return ChatResponse(answer=answer, sources=sources)
    
    def _build_context(self, chunks: List[Dict]) -> str:
        context_parts = []
        total_chars = 0
        max_chars = 1500  # Smaller limit for local LLM
        
        for chunk in chunks:
            chunk_text = f"{chunk['content']}"
            
            if total_chars + len(chunk_text) > max_chars:
                remaining_chars = max_chars - total_chars
                if remaining_chars > 50:
                    chunk_text = chunk_text[:remaining_chars] + "..."
                    context_parts.append(chunk_text)
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        return " | ".join(context_parts)
    
    def _extract_simple_answer(self, question: str, chunks: List[Dict]) -> str:
        """Fallback method to extract answers without LLM"""
        question_lower = question.lower()
        
        if 'list' in question_lower or 'show' in question_lower:
            # Extract unique values for listing
            items = []
            for chunk in chunks[:20]:  # Limit to first 20 chunks
                content = chunk['content']
                # Simple extraction - split by common delimiters
                parts = content.replace(',', '|').replace(';', '|').split('|')
                for part in parts:
                    part = part.strip()
                    if len(part) > 3 and len(part) < 100:  # Reasonable length
                        items.append(part)
            
            unique_items = list(set(items))[:15]  # Limit to 15 items
            if unique_items:
                return f"Found {len(unique_items)} items:\n" + "\n".join([f"• {item}" for item in unique_items])
        
        # Default response with data summary
        return f"Found {len(chunks)} relevant data entries. Here's a sample: {chunks[0]['content'][:200]}..."