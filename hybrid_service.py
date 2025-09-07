import os
from typing import List, Dict
from local_vector_store import LocalVectorStore
from models import ChatResponse

class HybridService:
    def __init__(self):
        self.vector_store = LocalVectorStore()
        self.use_cloud = os.getenv("USE_CLOUD_LLM", "true").lower() == "true"
        
        if self.use_cloud:
            try:
                from groq import Groq
                from secure_config import SecureConfig
                config = SecureConfig().get_api_keys()
                groq_key = config.get("GROQ_API_KEY")
                if groq_key:
                    self.groq_client = Groq(api_key=groq_key)
                else:
                    self.groq_client = None
                print("Using Groq cloud LLM (fast)")
            except:
                print("Groq not available, falling back to local processing")
                self.use_cloud = False
        else:
            print("Using local processing (slower but private)")
    
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
        
        if self.use_cloud:
            answer = self._generate_with_groq(question, relevant_chunks)
        else:
            answer = self._extract_simple_answer(question, relevant_chunks)
        
        sources = list(set([chunk.get('file_name', 'Unknown') for chunk in relevant_chunks]))
        return ChatResponse(answer=answer, sources=sources)
    
    def _generate_with_groq(self, question: str, chunks: List[Dict]) -> str:
        context = self._build_context(chunks)
        
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
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a project management assistant. Answer questions based only on the provided data context. Be concise and data-driven."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq failed: {e}")
            return self._extract_simple_answer(question, chunks)
    
    def _build_context(self, chunks: List[Dict]) -> str:
        context_parts = []
        total_chars = 0
        max_chars = 25000
        
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
        """Fast local processing without LLM"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['list', 'show', 'all']):
            items = []
            for chunk in chunks:
                content = chunk['content']
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if len(line) > 5 and len(line) < 100:
                        items.append(line)
            
            unique_items = list(set(items))
            if unique_items:
                return f"## Found {len(unique_items)} items:\n\n" + "\n".join([f"• {item}" for item in unique_items])
        
        # Count queries
        if 'count' in question_lower or 'how many' in question_lower:
            return f"**Total entries found:** {len(chunks)}\n\n**Sample data:**\n{chunks[0]['content'][:200]}..."
        
        # Status queries
        if 'status' in question_lower:
            status_items = []
            for chunk in chunks:
                content = chunk['content'].lower()
                if 'status' in content:
                    status_items.append(chunk['content'])
            
            if status_items:
                return f"## Status Information:\n\n" + "\n\n".join([f"• {item}" for item in status_items[:10]])
        
        return f"**Found {len(chunks)} relevant entries.**\n\n**Sample:**\n{chunks[0]['content'][:300]}..."