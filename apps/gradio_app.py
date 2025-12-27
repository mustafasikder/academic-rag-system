"""
Gradio web interface for Academic RAG System.

Provides interactive demo for document Q&A with PDF upload functionality
and visual feedback on retrieval process and answer generation.
"""

import gradio as gr
from typing import Tuple, List, Optional
import sys
import os
import tempfile
import yaml

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from academic_rag_system.preprocessing.pdf_cleaner import clean_academic_text
from academic_rag_system.preprocessing.adaptive_chunker import AcademicPaperChunker
from academic_rag_system.retrieval.hybrid_retriever import HybridRetriever
from academic_rag_system.generation.answer_generator import RAGAnswerGenerator


class RAGDemo:
    """
    Gradio demo interface for RAG system with PDF upload.
    
    Provides interactive Q&A with visual feedback showing:
    - Retrieved chunks with relevance scores
    - Chunk previews (first 100 chars)
    - Section metadata
    - LLM input preview
    """
    
    def __init__(self):
        """Initialize RAG demo without document (will be uploaded later)."""
        self.retriever = None
        self.generator = None
        self.chunks = None
        self.chunk_objects = None
        self.current_pdf_name = None
        
        # Load HybridRetriever config from YAML
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'hybrid_retriever.yaml')
        with open(config_path, 'r') as f:
            self.retriever_config = yaml.safe_load(f)
        # Load AdaptiveChunker config from YAML
        chunker_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'adaptive_chunker.yaml')
        with open(chunker_config_path, 'r') as f:
            self.chunker_config = yaml.safe_load(f)
        # Load AnswerGenerator config from YAML
        answer_gen_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'answer_generator.yaml')
        with open(answer_gen_config_path, 'r') as f:
            self.answer_gen_config = yaml.safe_load(f)
        
        print("RAG Demo initialized. Waiting for PDF upload...")
    
    def load_and_index_document(
        self,
        pdf_file,
        model_name: str = None,
        device: int = None
    ) -> str:
        """
        Load PDF, clean, chunk, index, and initialize generator.
        
        Args:
            pdf_file: Uploaded PDF file object from Gradio
            model_name: HuggingFace model for answer generation
            device: Device to run on (-1 for CPU, 0 for GPU)
            
        Returns:
            Status message
        """
        if pdf_file is None:
            return "❌ Please upload a PDF file first."
        
        try:
            import fitz
            
            # Get PDF path from Gradio file object
            pdf_path = pdf_file.name
            self.current_pdf_name = os.path.basename(pdf_path)
            
            print(f"Loading document: {self.current_pdf_name}")
            
            # Extract text from PDF
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text() for page in doc])
            doc.close()
            
            # Clean text
            text = clean_academic_text(text)
            
            # Adaptive chunking using config
            chunker = AcademicPaperChunker(
                target_chunk_size=self.chunker_config.get('target_chunk_size', 600),
                min_chunk_size=self.chunker_config.get('min_chunk_size', 200),
                max_chunk_size=self.chunker_config.get('max_chunk_size', 1000),
                overlap_sentences=self.chunker_config.get('overlap_sentences', 1),
                fallback_to_paragraphs=self.chunker_config.get('fallback_to_paragraphs', True)
            )
            
            self.chunk_objects = chunker.chunk_paper(text)
            self.chunks = chunker.chunks_to_list(self.chunk_objects)
            
            print(f"Created {len(self.chunks)} chunks")
            
            # Initialize hybrid retriever using config
            self.retriever = HybridRetriever(
                embedding_model_name=self.retriever_config.get('embedding_model_name', 'sentence-transformers/multi-qa-mpnet-base-cos-v1'),
                use_cosine=self.retriever_config.get('use_cosine', True),
                bm25_k1=self.retriever_config.get('bm25_k1', 1.5),
                bm25_b=self.retriever_config.get('bm25_b', 0.75)
            )
            
            # Index documents
            self.retriever.index_documents(self.chunks, show_progress=True)
            
            # Initialize answer generator using config
            ag_model_name = model_name if model_name is not None else self.answer_gen_config.get('model_name', 'google/long-t5-tglobal-base')
            ag_device = device if device is not None else self.answer_gen_config.get('device', -1)
            self.generator = RAGAnswerGenerator(
                ag_model_name,
                device=ag_device
            )
            
            return f"✅ Successfully loaded and indexed: {self.current_pdf_name}\n\nCreated {len(self.chunks)} chunks. Ready to answer questions!"
            
        except Exception as e:
            return f"❌ Error processing PDF: {str(e)}"
    
    def format_retrieved_chunks(
        self,
        retrieved_chunks: List[str],
        scores: List[float],
        indices: List[int]
    ) -> str:
        """
        Format retrieved chunks for display in Gradio.
        
        Shows:
        - Chunk index and relevance score
        - Section (from metadata)
        - Preview (first 100 chars)
        - Full chunk text (collapsible)
        
        Args:
            retrieved_chunks: List of chunk texts
            scores: Relevance scores
            indices: Chunk indices
            
        Returns:
            Formatted markdown string
        """
        output = "### 📚 Retrieved Context Chunks\n\n"
        
        for i, (chunk, score, idx) in enumerate(zip(retrieved_chunks, scores, indices)):
            # Get section metadata
            section = self.chunk_objects[idx].section if idx < len(self.chunk_objects) else "unknown"
            
            # Preview (first 100 chars)
            preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
            
            output += f"**Chunk {i+1}** (Index: {idx}, Section: `{section}`, Score: `{score:.4f}`)\n\n"
            output += f"📝 *Preview:* \"{preview}\"\n\n"
            output += f"<details>\n<summary>View full chunk</summary>\n\n{chunk}\n\n</details>\n\n"
            output += "---\n\n"
        
        return output
    
    def format_llm_input_preview(
        self,
        query: str,
        retrieved_chunks: List[str]
    ) -> str:
        """
        Format LLM input preview showing what gets sent to the model.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Formatted preview of LLM input
        """
        context = "\n\n".join(retrieved_chunks)
        
        # Show first 200 chars of combined context
        context_preview = context[:200] + "..." if len(context) > 200 else context
        
        # Count tokens
        total_tokens = self.generator.count_tokens(context + query)
        
        output = "### 🤖 LLM Input Preview\n\n"
        output += f"**Query:** {query}\n\n"
        output += f"**Context (combined {len(retrieved_chunks)} chunks):**\n\n"
        output += f"```\n{context_preview}\n```\n\n"
        output += f"**Total Input Tokens:** {total_tokens}\n\n"
        
        if total_tokens > self.generator.max_input_tokens:
            output += f"⚠️ *Context will be truncated (exceeds {self.generator.max_input_tokens} token limit)*\n\n"
        
        return output
    
    def answer_question(
        self,
        question: str,
        k: int = 3,
        method: str = "hybrid",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ) -> Tuple[str, str, str]:
        """
        Answer question using RAG pipeline.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            method: Retrieval method ('hybrid', 'dense', or 'sparse')
            dense_weight: Weight for dense retrieval (hybrid only)
            sparse_weight: Weight for sparse retrieval (hybrid only)
            
        Returns:
            Tuple of (answer, retrieved_chunks_display, llm_input_preview)
        """
        if self.retriever is None or self.generator is None:
            return (
                "❌ Please upload and index a PDF document first.",
                "",
                ""
            )
        
        if not question.strip():
            return "Please enter a question.", "", ""
        
        try:
            # Retrieve relevant chunks
            retrieved_chunks, scores, indices = self.retriever.search(
                question,
                k=k,
                method=method,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                fusion_method='weighted'
            )
            
            # Format retrieved chunks for display
            chunks_display = self.format_retrieved_chunks(
                retrieved_chunks,
                scores,
                indices
            )
            
            # Format LLM input preview
            llm_input = self.format_llm_input_preview(question, retrieved_chunks)
            
            # Generate answer
            answer = self.generator.generate_answer(
                question,
                retrieved_chunks,
                max_new_tokens=250,
                do_sample=False,
                ensure_complete=True
            )
            
            # Format final answer with metadata
            answer_display = f"### ✅ Generated Answer\n\n{answer}\n\n"
            answer_display += f"---\n\n"
            answer_display += f"*Document: {self.current_pdf_name}*\n"
            answer_display += f"*Retrieved from {len(retrieved_chunks)} chunks using {method} retrieval*\n"
            answer_display += f"*Model: {self.generator.model_name}*\n"
            
            return answer_display, chunks_display, llm_input
            
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}", "", ""


def create_interface(
    model_name: str = None,
    device: int = -1, 
    theme = gr.themes.Soft()
) -> gr.Blocks:
    """
    Create Gradio interface for RAG system with PDF upload.
    
    Args:
        model_name: HuggingFace model for answer generation
        device: Device to run on (-1 for CPU, 0 for GPU)
        
    Returns:
        Gradio Blocks interface
    """
    # Initialize RAG system (without document)
    rag_demo = RAGDemo()
    
    # Create Gradio interface
    with gr.Blocks(title="Academic RAG System") as demo:
        
        gr.Markdown(
            """
            # 📄 Academic Paper Q&A System
            
            Upload a research paper (PDF) and ask questions about it. The system uses hybrid retrieval 
            (semantic + keyword matching) and language models to generate accurate answers.
            
            **Model:** `{}`
            """.format(rag_demo.answer_gen_config.get('model_name'))
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # PDF upload section
                pdf_upload = gr.File(
                    label="📤 Upload PDF Document",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                load_btn = gr.Button("📚 Load & Index Document", variant="primary")
                
                status_output = gr.Markdown(
                    value="*Upload a PDF to get started*"
                )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Question input section
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What methods were used in this study?",
                    lines=3
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of chunks to retrieve (k)"
                    )
                    
                    method_radio = gr.Radio(
                        choices=["hybrid", "dense", "sparse"],
                        value="hybrid",
                        label="Retrieval method"
                    )
                    
                    with gr.Row():
                        dense_weight_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.6,
                            step=0.1,
                            label="Dense weight"
                        )
                        
                        sparse_weight_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.4,
                            step=0.1,
                            label="Sparse weight"
                        )
                
                submit_btn = gr.Button("🔍 Ask Question", variant="primary", size="lg")
                
                # Example questions
                gr.Examples(
                    examples=[
                        ["What is the main research question or objective of this study?"],
                        ["What methods were used to collect data in this study?"],
                        ["What were the most significant results or findings?"],
                        ["What limitations or weaknesses does the study acknowledge?"],
                        ["What are the practical or real-world applications of these findings?"]
                    ],
                    inputs=question_input,
                    label="Example Questions"
                )
        
        with gr.Row():
            with gr.Column():
                # Answer output
                answer_output = gr.Markdown(
                    label="Answer",
                    value="*Answer will appear here*"
                )
        
        with gr.Row():
            with gr.Column():
                # LLM input preview
                llm_input_output = gr.Markdown(
                    label="LLM Input Preview",
                    value="*LLM input preview will appear here*"
                )
            
            with gr.Column():
                # Retrieved chunks display
                chunks_output = gr.Markdown(
                    label="Retrieved Context",
                    value="*Retrieved chunks will appear here*"
                )
        
        # Connect PDF loading
        load_btn.click(
            fn=lambda pdf: rag_demo.load_and_index_document(pdf, model_name, device),
            inputs=[pdf_upload],
            outputs=[status_output]
        )
        
        # Connect question answering
        submit_btn.click(
            fn=rag_demo.answer_question,
            inputs=[
                question_input,
                k_slider,
                method_radio,
                dense_weight_slider,
                sparse_weight_slider
            ],
            outputs=[answer_output, chunks_output, llm_input_output]
        )
        
        # Also trigger on Enter key
        question_input.submit(
            fn=rag_demo.answer_question,
            inputs=[
                question_input,
                k_slider,
                method_radio,
                dense_weight_slider,
                sparse_weight_slider
            ],
            outputs=[answer_output, chunks_output, llm_input_output]
        )
        
        gr.Markdown(
            """
            ---
            
            ### How it works:
            
            1. **Upload:** Drag and drop or upload a research paper (PDF format)
            2. **Index:** System extracts text, cleans it, creates chunks, and builds retrieval indices
            3. **Ask:** Enter your question about the paper
            4. **Retrieve:** System searches through document chunks using hybrid dense (semantic) + sparse (keyword) retrieval
            5. **Generate:** Language model synthesizes answer based only on retrieved context
            6. **Display:** Shows answer, retrieved chunks with scores, and LLM input preview
            
            **System Components:**
            - PDF text extraction with artifact removal (dehyphenation, boilerplate removal)
            - Adaptive chunking (200-1000 chars, respects section boundaries)
            - Hybrid retrieval (MPNet embeddings + BM25)
            - Multi-model LLM support (T5, Flan-T5, LongT5, Phi-3, TinyLlama)
            
            **Note:** First-time loading may take 1-2 minutes to download models.
            """
        )
    
    return demo


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch Gradio demo for Academic RAG System"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name (default: from answer_generator.yaml)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device to run on (-1 for CPU, 0 for GPU)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )
    
    args = parser.parse_args()
    
    # Create and launch interface
    demo = create_interface(
        model_name=args.model,
        device=args.device
    )
    
    demo.launch(
        # server_name="0.0.0.0",
        server_port=args.port,
        share=args.share, 
        theme=gr.themes.Soft()
    )