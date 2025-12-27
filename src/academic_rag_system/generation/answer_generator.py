# Answer Generator module
"""
Answer generation module for RAG system.

Provides LLM-based answer generation with support for multiple models,
smart context truncation, and post-processing to ensure complete sentences.
"""

import re
from typing import List, Optional, Tuple
from transformers import pipeline, AutoTokenizer


class RAGAnswerGenerator:
    """
    LLM-based answer generator for RAG systems.
    
    Supports multiple model architectures (T5-style encoder-decoder and 
    GPT-style decoder-only models) with automatic detection and appropriate
    handling for each type.
    
    Features:
    - Multi-model support (LongT5, Flan-T5, Phi-3, TinyLlama)
    - Smart context truncation when exceeding token limits
    - Automatic sentence completion for clean outputs
    - Model-specific prompt formatting
    
    Attributes:
        model_name (str): HuggingFace model identifier
        pipeline: HuggingFace text generation pipeline
        tokenizer: Model tokenizer for token counting
        is_causal_lm (bool): Whether model is decoder-only (GPT-style)
        max_input_tokens (int): Maximum input token limit for model
    """
    
    def __init__(
        self,
        model_name: str = "google/long-t5-tglobal-base",
        device: int = -1,
        max_input_tokens: Optional[int] = None
    ) -> None:
        """
        Initialize answer generator.
        
        Args:
            model_name: HuggingFace model identifier. Supported models:
                - google/long-t5-tglobal-base (recommended, 16K context)
                - google/flan-t5-large (512 tokens)
                - google/flan-t5-xl (512 tokens, better quality)
                - microsoft/phi-3-mini-4k-instruct (4K tokens, GPU recommended)
                - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (2K tokens)
            device: Device to run on (-1 for CPU, 0 for GPU)
            max_input_tokens: Maximum input tokens. Auto-detected if None.
        """
        print(f"Loading model: {model_name}")
        
        self.model_name = model_name
        
        # Detect model type (encoder-decoder vs decoder-only)
        self.is_causal_lm = any(
            x in model_name.lower() 
            for x in ['phi', 'gpt', 'llama', 'mistral', 'tinyllama']
        )
        
        # Load appropriate pipeline
        if self.is_causal_lm:
            # Decoder-only models (GPT-style)
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=device,
                trust_remote_code=True
            )
            self.pipeline_task = "text-generation"
        else:
            # Encoder-decoder models (T5-style)
            self.pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                device=device
            )
            self.pipeline_task = "text2text-generation"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Detect max input length
        if max_input_tokens is None:
            try:
                self.max_input_tokens = (
                    self.pipeline.model.config.max_position_embeddings
                )
            except AttributeError:
                # Fallback for models without max_position_embeddings
                self.max_input_tokens = 2048
        else:
            self.max_input_tokens = max_input_tokens
        
        print(
            f"✅ Model loaded ({self.pipeline_task}). "
            f"Max input tokens: {self.max_input_tokens}"
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text string
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text, add_special_tokens=True))
    
    def truncate_context(
        self,
        context: str,
        query: str,
        prompt_template: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Truncate context to fit within token limit.
        
        Strategy: Keep first 60% and last 40% of context tokens to preserve
        both introduction and conclusion while staying within limits.
        
        Args:
            context: Retrieved context chunks (combined)
            query: User query
            prompt_template: Full prompt template with {context} and {query} placeholders
            max_tokens: Maximum allowed tokens. Uses model max - 200 if None.
            
        Returns:
            Truncated context that fits within limits
        """
        if max_tokens is None:
            # Reserve space for prompt structure + answer generation
            max_tokens = self.max_input_tokens - 200
        
        # Build full prompt to measure
        test_prompt = prompt_template.format(context=context, query=query)
        current_tokens = self.count_tokens(test_prompt)
        
        if current_tokens <= max_tokens:
            return context  # No truncation needed
        
        # Calculate how many context tokens we can afford
        query_tokens = self.count_tokens(query)
        template_tokens = self.count_tokens(
            prompt_template.format(context="", query="")
        )
        available_context_tokens = (
            max_tokens - query_tokens - template_tokens - 50  # Buffer
        )
        
        # Truncate context tokens
        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        
        if len(context_tokens) > available_context_tokens:
            # Keep first 60% and last 40% to preserve intro and conclusion
            keep_start = int(available_context_tokens * 0.6)
            keep_end = available_context_tokens - keep_start
            
            truncated_tokens = (
                context_tokens[:keep_start] +
                self.tokenizer.encode(
                    "\n\n[...context truncated...]\n\n", 
                    add_special_tokens=False
                ) +
                context_tokens[-keep_end:]
            )
            
            context = self.tokenizer.decode(truncated_tokens)
            print(
                f"⚠️  Context truncated: {len(context_tokens)} → "
                f"{len(truncated_tokens)} tokens"
            )
        
        return context
    
    def complete_sentence(self, text: str) -> str:
        """
        Ensure text ends with a complete sentence.
        
        If text ends mid-sentence, truncate to last sentence-ending punctuation.
        If no punctuation found in last 50% of text, adds a period.
        
        Args:
            text: Generated text that may end mid-sentence
            
        Returns:
            Text ending with complete sentence
        """
        text = text.strip()
        
        if not text:
            return text
        
        # Check if ends with sentence-ending punctuation
        if text[-1] not in '.!?':
            # Find last sentence-ending punctuation
            last_period = text.rfind('.')
            last_exclaim = text.rfind('!')
            last_question = text.rfind('?')
            
            last_punct = max(last_period, last_exclaim, last_question)
            
            # Only truncate if we're not losing too much (>50% of text)
            if last_punct > len(text) * 0.5:
                text = text[:last_punct + 1]
                print("ℹ️  Truncated to last complete sentence")
            else:
                # If no good punctuation found, add period
                text = text + "."
                print("ℹ️  Added terminal punctuation")
        
        return text
    
    def clean_answer(self, text: str) -> str:
        """
        Clean up common generation artifacts.
        
        Removes:
        - Repeated punctuation (e.g., "..." → ".")
        - Extra whitespace
        - Spacing issues around punctuation
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove any repeated punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[str],
        max_new_tokens: int = 250,
        min_length: int = 30,
        temperature: float = 0.7,
        do_sample: bool = False,
        ensure_complete: bool = True,
        display_truncation_message: bool = True
    ) -> str:
        """
        Generate answer from query and retrieved chunks.
        
        Args:
            query: User question
            retrieved_chunks: List of retrieved text chunks (in relevance order)
            max_new_tokens: Maximum tokens to generate
            min_length: Minimum answer length (encoder-decoder models only)
            temperature: Sampling temperature. Only used if do_sample=True.
            do_sample: Whether to use sampling (False = greedy/deterministic)
            ensure_complete: Whether to ensure complete sentences
            display_truncation_message: Whether to display truncation messages

        Returns:
            Generated answer text
            
        Example:
            >>> generator = RAGAnswerGenerator("google/long-t5-tglobal-base")
            >>> chunks = ["Methods: We used surveys...", "Sample size was 100..."]
            >>> answer = generator.generate_answer(
            ...     "What methods were used?",
            ...     chunks,
            ...     do_sample=False
            ... )
        """
        # Combine chunks into context
        context = "\n\n".join(retrieved_chunks)
        
        # Build prompt based on model type
        if self.is_causal_lm:
            # Decoder-only models need instruction-style prompts
            prompt_template = """Answer the following Question using only the information in the Context.
DO NOT generate any further question/response. Do not repeat the Question.

Context: {context}

Question: {query}

Response:"""
        else:
            # Encoder-decoder models use simpler prompts
            prompt_template = """Based on the context below, provide a comprehensive and accurate answer to the question. Use only information from the context.

Context: {context}

Question: {query}

Answer:"""
        
        # Truncate context if needed
        context = self.truncate_context(context, query, prompt_template)
        
        # Format final prompt
        prompt = prompt_template.format(context=context, query=query)
        
        # Count tokens for logging
        input_tokens = self.count_tokens(prompt)
        print(
            f"Input: {input_tokens} tokens | "
            f"Generating up to {max_new_tokens} tokens"
        )
        
        # Build generation parameters
        generation_params = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        
        # Add model-specific parameters
        if self.is_causal_lm:
            generation_params["return_full_text"] = False
            generation_params["pad_token_id"] = self.tokenizer.eos_token_id
        else:
            generation_params["min_length"] = min_length
            generation_params["early_stopping"] = False
        
        if do_sample:
            generation_params["temperature"] = temperature
            generation_params["top_p"] = 0.9
        
        # Generate answer
        output = self.pipeline(prompt, **generation_params)
        answer_text = output[0]['generated_text']
        
        # Clean up decoder-only model artifacts
        if self.is_causal_lm:
            answer_text = answer_text.strip()
            
            # Remove "Answer:" or "Response:" prefix if present
            for prefix in ["Answer:", "Response:"]:
                if answer_text.startswith(prefix):
                    answer_text = answer_text[len(prefix):].strip()
            
            # Stop at first new question or section marker
            stop_markers = [
                "\n\nQuestion:",
                "\nQuestion:",
                "\n###",
                "\n\n###",
                "### Question:",
                "### Context:",
                "\n\n\n"  # Multiple blank lines
            ]
            
            for marker in stop_markers:
                if marker in answer_text:
                    answer_text = answer_text.split(marker)[0].strip()
                    if display_truncation_message:
                        print(f"⚠️  Truncated at '{marker}' to prevent over-generation")
                    break
        
        # Ensure complete sentences if requested
        if ensure_complete:
            answer_text = self.complete_sentence(answer_text)
        
        # Clean up any artifacts
        answer_text = self.clean_answer(answer_text)
        
        return answer_text
    
    def generate_with_fallback(
        self,
        query: str,
        retrieved_chunks: List[str],
        **kwargs
    ) -> str:
        """
        Generate answer with automatic fallback for length errors.
        
        If generation fails due to length issues, automatically retries 
        with more aggressive truncation.
        
        Args:
            query: User question
            retrieved_chunks: List of retrieved chunks
            **kwargs: Additional arguments passed to generate_answer()
            
        Returns:
            Generated answer text
        """
        try:
            return self.generate_answer(query, retrieved_chunks, **kwargs)
        except Exception as e:
            if "length" in str(e).lower() or "token" in str(e).lower():
                print(
                    f"⚠️  Generation failed (likely length issue). "
                    f"Retrying with more truncation..."
                )
                
                # Reduce context more aggressively
                context = "\n\n".join(retrieved_chunks)
                context_tokens = self.tokenizer.encode(
                    context, 
                    add_special_tokens=False
                )
                
                # Keep only first chunk or first 300 tokens
                max_context_tokens = min(300, len(context_tokens) // 2)
                truncated_tokens = context_tokens[:max_context_tokens]
                truncated_context = self.tokenizer.decode(truncated_tokens)
                
                return self.generate_answer(
                    query,
                    [truncated_context],
                    **kwargs
                )
            else:
                raise


# Usage example
if __name__ == "__main__":
    # Example with LongT5 (recommended)
    generator = RAGAnswerGenerator(
        model_name="google/long-t5-tglobal-base",
        device=-1  # CPU
    )
    
    query = "What methods were used in this study?"
    chunks = [
        "Methods: We conducted cross-sectional household surveys in southern Syria.",
        "Data collection occurred between June 2016 and February 2017.",
        "The sample size was calculated to be 87-96 households per sub-district.",
    ]
    
    answer = generator.generate_answer(
        query,
        chunks,
        max_new_tokens=200,
        do_sample=False,  # Deterministic for reproducibility
        ensure_complete=True
    )
    
    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")