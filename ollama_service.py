import ollama
from typing import List, Dict
from local_vector_store import LocalVectorStore
from models import ChatResponse

class OllamaService:
    def __init__(self):
        self.vector_store = LocalVectorStore()
        self.model_name = "gemma3:1b"  # Using Gemma3 1B for faster responses
        
        # Check if Ollama is running and model is available
        try:
            ollama.list()
            print("Ollama connected successfully")
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            print("Please install and start Ollama first")
    
    def generate_answer(self, question: str, user_id: str) -> ChatResponse:
        # Retrieve relevant context
        comprehensive_keywords = ['all', 'list', 'show', 'total', 'count', 'every', 'complete']
        needs_all_data = any(keyword in question.lower() for keyword in comprehensive_keywords)
        
        if needs_all_data:
            relevant_chunks = self.vector_store.search(question, user_id, top_k=100)
        else:
            relevant_chunks = self.vector_store.search(question, user_id, top_k=20)
        
        if not relevant_chunks:
            return ChatResponse(
                answer="I don't have enough data to answer that question. Please upload some files first.",
                sources=[]
            )
        
        # Build context
        context = self._build_context(relevant_chunks)
        
        # Generate answer using Ollama
        prompt = f"""You are ProjectIQ, an AI project management assistant. Format your responses professionally with clear structure.

Use markdown formatting:
- **Bold** for important points
- • Bullet points for lists
- Numbers for ordered lists
- ## Headers for sections

Project Data:
{context}

Question: {question}

Provide a well-structured, professional response:"""
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            )
            answer = response['response'].strip()
            
        except Exception as e:
            print(f"Ollama generation failed: {e}")
            # Fallback to simple data extraction
            answer = self._extract_simple_answer(question, relevant_chunks)
        
        sources = list(set([chunk.get('file_name', 'Unknown') for chunk in relevant_chunks]))
        return ChatResponse(answer=answer, sources=sources)
    
    def _build_context(self, chunks: List[Dict]) -> str:
        context_parts = []
        total_chars = 0
        max_chars = 8000  # Good limit for Ollama
        
        for chunk in chunks:
            chunk_text = f"From {chunk.get('file_name', 'file')}: {chunk['content']}"
            
            if total_chars + len(chunk_text) > max_chars:
                remaining_chars = max_chars - total_chars
                if remaining_chars > 100:
                    chunk_text = chunk_text[:remaining_chars] + "..."
                    context_parts.append(chunk_text)
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def _extract_simple_answer(self, question: str, chunks: List[Dict]) -> str:
        """Fallback method when Ollama fails"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['list', 'show', 'all']):
            items = []
            for chunk in chunks:  # Process all chunks
                content = chunk['content']
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if len(line) > 5 and len(line) < 100:
                        items.append(line)
            
            unique_items = list(set(items))  # Remove artificial limit
            if unique_items:
                return f"Found {len(unique_items)} items:\n" + "\n".join([f"• {item}" for item in unique_items])
        
        return f"Found {len(chunks)} relevant entries. Sample: {chunks[0]['content'][:300]}..."