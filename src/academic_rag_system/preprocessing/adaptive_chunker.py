# Adaptive Chunker module
import re
import nltk
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class Chunk:
    """Represents a semantic chunk with metadata"""
    text: str
    section: str
    chunk_id: int
    start_char: int
    end_char: int
    sentences: int
    
    def __str__(self):
        return f"Chunk {self.chunk_id} [{self.section}]: {len(self.text)} chars, {self.sentences} sentences"


class AcademicPaperChunker:
    """
    Adaptive chunking for academic papers that respects document structure.
    
    Strategy:
    1. Identify major sections (Abstract, Introduction, Methods, Results, Discussion)
    2. Within sections, chunk by paragraph boundaries
    3. If paragraphs are too long, split by sentence boundaries
    4. Preserve semantic context with smart overlaps
    """
    
    # Common section headers in academic papers
    # Updated patterns to handle various formatting from PDF extraction
    SECTION_PATTERNS = {
        'abstract': r'(?:^|\n)\s*abstract\s*(?:\n|$)',
        'introduction': r'(?:^|\n)\s*(?:introduction|background)\s*(?:\n|$)',
        'methods': r'(?:^|\n)\s*(?:methods?|methodology|materials?\s+and\s+methods?|experimental\s+design|study\s+design|data\s+collection)\s*(?:\n|$)',
        'results': r'(?:^|\n)\s*(?:results?|findings?)\s*(?:\n|$)',
        'discussion': r'(?:^|\n)\s*(?:discussion|interpretation)\s*(?:\n|$)',
        'conclusion': r'(?:^|\n)\s*(?:conclusion|summary|concluding\s+remarks?)\s*(?:\n|$)',
    }
    
    def __init__(
        self,
        target_chunk_size: int = 600,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1000,
        overlap_sentences: int = 1,
        additional_sections: Optional[Dict[str, str]] = None,
        fallback_to_paragraphs: bool = True
    ):
        """
        Initialize the chunker with size parameters.
        
        Args:
            target_chunk_size: Ideal chunk size in characters
            min_chunk_size: Minimum acceptable chunk size
            max_chunk_size: Maximum acceptable chunk size
            overlap_sentences: Number of sentences to overlap between chunks
            additional_sections: Dict of custom section patterns, e.g. 
                {'context': r'\n\s*research\s+in\s+the\s+context\s*\n'}
            fallback_to_paragraphs: If True, treats entire text as paragraphs when no sections found
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        self.fallback_to_paragraphs = fallback_to_paragraphs
        
        # Merge default patterns with custom ones
        self.section_patterns = self.SECTION_PATTERNS.copy()
        if additional_sections:
            self.section_patterns.update(additional_sections)
        
    def identify_sections(self, text: str) -> Dict[str, Tuple[int, int]]:
        """
        Identify major sections in the paper.
        
        Returns:
            Dict mapping section names to (start_pos, end_pos) tuples
        """
        sections = {}
        section_positions = []
        
        # Find all section headers
        for section_name, pattern in self.section_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                section_positions.append((match.start(), section_name, match.group(0)))
        
        # Sort by position
        section_positions.sort()
        
        # Create sections with start and end positions
        for i, (start_pos, section_name, header) in enumerate(section_positions):
            end_pos = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(text)
            sections[section_name] = (start_pos, end_pos)
        
        # If no sections found, use fallback strategy
        if not sections and self.fallback_to_paragraphs:
            print("⚠️  No standard sections detected. Using paragraph-based chunking for entire document.")
            sections['body'] = (0, len(text))
            
        return sections
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Paragraphs are separated by one or more blank lines.
        Uses 2+ newlines as delimiter, preserving paragraph structure.
        """
        # Split on 2+ newlines (paragraph breaks)
        # This pattern handles various whitespace combinations
        paragraphs = re.split(r'\n\s*\n+', text)
        
        # Clean up and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_semantic_chunks(self, text: str, section_name: str, start_offset: int) -> List[Chunk]:
        """
        Create semantically coherent chunks from a section of text.
        
        Strategy:
        1. Start with paragraph boundaries
        2. If paragraph < min_size, merge with next
        3. If paragraph > max_size, split by sentences
        4. Add sentence-level overlap between chunks
        """
        chunks = []
        chunk_id = 0
        
        paragraphs = self.split_into_paragraphs(text)
        
        # Handle case where no paragraph breaks exist
        if not paragraphs:
            paragraphs = [text]
        
        current_chunk_text = ""
        current_chunk_sentences = []
        current_start = start_offset
        
        for para in paragraphs:
            para_sentences = self.split_into_sentences(para)
            
            # Try adding this paragraph to current chunk
            tentative_text = current_chunk_text + "\n\n" + para if current_chunk_text else para
            
            # Decision logic
            if len(tentative_text) <= self.max_chunk_size:
                # Paragraph fits, add it
                current_chunk_text = tentative_text
                current_chunk_sentences.extend(para_sentences)
                
            elif len(current_chunk_text) < self.min_chunk_size:
                # Current chunk too small, must add this paragraph
                current_chunk_text = tentative_text
                current_chunk_sentences.extend(para_sentences)
                
            else:
                # Current chunk is good size, save it and start new one
                if current_chunk_text:
                    chunk = Chunk(
                        text=current_chunk_text.strip(),
                        section=section_name,
                        chunk_id=chunk_id,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk_text),
                        sentences=len(current_chunk_sentences)
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_text = " ".join(current_chunk_sentences[-self.overlap_sentences:])
                    current_chunk_text = overlap_text + "\n\n" + para
                    current_chunk_sentences = current_chunk_sentences[-self.overlap_sentences:] + para_sentences
                    current_start = chunk.end_char - len(overlap_text)
                else:
                    current_chunk_text = para
                    current_chunk_sentences = para_sentences
        
        # Don't forget the last chunk
        if current_chunk_text:
            chunk = Chunk(
                text=current_chunk_text.strip(),
                section=section_name,
                chunk_id=chunk_id,
                start_char=current_start,
                end_char=current_start + len(current_chunk_text),
                sentences=len(current_chunk_sentences)
            )
            chunks.append(chunk)
        
        # Handle oversized chunks by splitting on sentence boundaries
        final_chunks = []
        for chunk in chunks:
            if len(chunk.text) > self.max_chunk_size:
                # Split this chunk by sentences
                sub_chunks = self._split_large_chunk(chunk, section_name)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_large_chunk(self, chunk: Chunk, section_name: str) -> List[Chunk]:
        """Split an oversized chunk by sentence boundaries."""
        sentences = self.split_into_sentences(chunk.text)
        sub_chunks = []
        
        current_text = ""
        current_sentences = []
        chunk_id_base = chunk.chunk_id
        
        for sentence in sentences:
            tentative_text = current_text + " " + sentence if current_text else sentence
            
            if len(tentative_text) <= self.max_chunk_size:
                current_text = tentative_text
                current_sentences.append(sentence)
            else:
                # Save current chunk
                if current_text:
                    sub_chunk = Chunk(
                        text=current_text.strip(),
                        section=section_name,
                        chunk_id=f"{chunk_id_base}.{len(sub_chunks)}",
                        start_char=chunk.start_char,
                        end_char=chunk.start_char + len(current_text),
                        sentences=len(current_sentences)
                    )
                    sub_chunks.append(sub_chunk)
                
                # Start new with overlap
                overlap_text = " ".join(current_sentences[-self.overlap_sentences:])
                current_text = overlap_text + " " + sentence if overlap_text else sentence
                current_sentences = current_sentences[-self.overlap_sentences:] + [sentence]
        
        # Last sub-chunk
        if current_text:
            sub_chunk = Chunk(
                text=current_text.strip(),
                section=section_name,
                chunk_id=f"{chunk_id_base}.{len(sub_chunks)}",
                start_char=chunk.start_char,
                end_char=chunk.start_char + len(current_text),
                sentences=len(current_sentences)
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def chunk_paper(self, text: str, preserve_section_info: bool = True) -> List[Chunk]:
        """
        Main method: chunk an entire academic paper adaptively.
        
        Args:
            text: The full text of the paper
            preserve_section_info: If True, keeps section metadata with chunks
            
        Returns:
            List of Chunk objects with metadata
        """
        sections = self.identify_sections(text)
        all_chunks = []
        global_chunk_id = 0
        
        # Sort sections by position in document
        sorted_sections = sorted(sections.items(), key=lambda x: x[1][0])
        
        for section_name, (start_pos, end_pos) in sorted_sections:
            section_text = text[start_pos:end_pos]
            
            # Remove section header from text (if it matches a known pattern)
            pattern = self.section_patterns.get(section_name)
            if pattern:
                section_text = re.sub(
                    pattern,
                    '\n',
                    section_text,
                    count=1,
                    flags=re.IGNORECASE
                ).strip()
            
            if not section_text:
                continue
            
            # Create chunks for this section
            section_chunks = self.create_semantic_chunks(section_text, section_name, start_pos)
            
            # Update global chunk IDs
            for chunk in section_chunks:
                chunk.chunk_id = global_chunk_id
                global_chunk_id += 1
            
            all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def chunks_to_list(self, chunks: List[Chunk]) -> List[str]:
        """Convert Chunk objects to simple list of strings (for compatibility)."""
        return [chunk.text for chunk in chunks]
    
    def chunks_with_metadata(self, chunks: List[Chunk]) -> List[Dict]:
        """Convert Chunk objects to dictionaries with full metadata."""
        return [
            {
                'id': chunk.chunk_id,
                'text': chunk.text,
                'section': chunk.section,
                'length': len(chunk.text),
                'sentences': chunk.sentences,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char
            }
            for chunk in chunks
        ]
    
    def print_statistics(self, chunks: List[Chunk]):
        """Print statistics about the chunking."""
        print(f"\n{'='*60}")
        print(f"CHUNKING STATISTICS")
        print(f"{'='*60}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Average chunk size: {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars")
        print(f"Min chunk size: {min(len(c.text) for c in chunks)} chars")
        print(f"Max chunk size: {max(len(c.text) for c in chunks)} chars")
        print(f"Average sentences per chunk: {sum(c.sentences for c in chunks) / len(chunks):.1f}")
        
        # By section
        sections = {}
        for chunk in chunks:
            if chunk.section not in sections:
                sections[chunk.section] = []
            sections[chunk.section].append(chunk)
        
        print(f"\nChunks by section:")
        for section, section_chunks in sections.items():
            print(f"  {section.capitalize()}: {len(section_chunks)} chunks")
        print(f"{'='*60}\n")


# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    
    sample_academic_text = """
    Abstract
    
    This study examines the impact of climate change on agricultural productivity.
    We analyzed data from 500 farms over 10 years. Results show significant decline
    in yields due to temperature increases.
    
    Introduction
    
    Climate change poses significant challenges to global food security. Previous 
    research has documented temperature increases across agricultural regions.
    However, few studies have examined long-term impacts on specific crops.
    
    This research fills that gap by analyzing decade-long datasets from diverse
    farming operations. We focus on wheat and corn production in temperate zones.
    
    Methods
    
    We collected data from 500 farms across three countries from 2010 to 2020.
    Temperature and precipitation data came from local weather stations. Yield
    data was self-reported by farmers and verified through agricultural records.
    
    Statistical analysis employed mixed-effects models to account for farm-level
    variation and temporal autocorrelation. Climate variables were standardized
    before analysis.
    
    Results
    
    Average temperatures increased by 1.2°C over the study period. This correlated
    with a 15% decline in wheat yields and 12% decline in corn yields.
    
    The effect was more pronounced in regions with less irrigation infrastructure.
    Farms with adaptive practices showed smaller yield declines.
    
    Discussion
    
    Our findings confirm that climate change is already impacting agricultural
    productivity. The magnitude of effects exceeds previous projections based on
    shorter time series.
    
    Policy implications are significant. Investment in irrigation and crop
    breeding programs should be prioritized.
    """
    
    # Initialize chunker
    chunker = AcademicPaperChunker(
        target_chunk_size=400,
        min_chunk_size=150,
        max_chunk_size=800,
        overlap_sentences=1
    )
    
    # Create chunks
    chunks = chunker.chunk_paper(sample_academic_text)
    
    # Print statistics
    chunker.print_statistics(chunks)
    
    # Print first few chunks
    print("SAMPLE CHUNKS:\n")
    for i, chunk in enumerate(chunks[:5]):
        print(f"{chunk}")
        print(f"Text: {chunk.text[:100]}...")
        print()
    
    # Get simple list (compatible with your existing code)
    simple_chunks = chunker.chunks_to_list(chunks)
    print(f"\nSimple list has {len(simple_chunks)} text strings")
    
    # Get full metadata (useful for debugging/analysis)
    metadata_chunks = chunker.chunks_with_metadata(chunks)
    print(f"Metadata list has {len(metadata_chunks)} dictionaries")
    print(f"Example metadata: {metadata_chunks[0]}")