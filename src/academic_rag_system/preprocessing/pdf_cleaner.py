# PDF Cleaner module
import re

def clean_academic_text(text):
    """
    Remove common non-content sections from academic papers.
    
    This function strips out boilerplate sections that don't contribute to
    answering research questions: references, author metadata, funding,
    ethics statements, publisher footers, etc.
    
    Also fixes PDF extraction artifacts like hyphenated line breaks.
    
    Args:
        text (str): Raw text extracted from academic PDF
        
    Returns:
        str: Cleaned text with boilerplate removed
    """
    
    # ========== FIX PDF EXTRACTION ARTIFACTS (DO THIS FIRST!) ==========
    
    # Step 1: Remove hyphenation at line breaks: "method-\nology" -> "methodology"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # Step 2: Fix soft hyphens (Unicode character U+00AD)
    text = text.replace('\u00ad', '')
    
    # Step 3: Remove single newlines ONLY within sentences (not all single newlines!)
    # Only remove \n when it's clearly mid-sentence:
    # - Preceded by lowercase letter or comma (not end of sentence)
    # - Followed by lowercase letter (continuation of sentence)
    # This preserves section headers and paragraph structure
    text = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', text)
    
    # Step 4: Clean up multiple spaces (but don't touch newlines yet)
    text = re.sub(r' {2,}', ' ', text)
    
    # ========== REFERENCES SECTION ==========
    # Match various headings for bibliography/references
    # Common patterns: "References", "Bibliography", "Works Cited", "Literature Cited"
    references_patterns = [
        r'\n\s*References?\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*Works?\s+Cited\s*\n',
        r'\n\s*Literature\s+Cited\s*\n',
        r'\n\s*Citations?\s*\n',
        r'\n\s*Reference\s+List\s*\n',
        r'\n\s*List\s+of\s+References?\s*\n',
    ]
    
    for pattern in references_patterns:
        parts = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[0]
            break
    
    # ========== AUTHOR CONTRIBUTIONS ==========
    # Match author/contributor metadata sections
    author_patterns = [
        r'Authors?\s+[Cc]ontributions?',
        r'Author\s+[Ii]nformation',
        r'Contributors?',
        r'Authorship\s+[Cc]ontributions?',
        r'[Cc]ontribution\s+of\s+[Aa]uthors?',
        r'[Ww]ho\s+[Dd]id\s+[Ww]hat',
    ]
    
    for pattern in author_patterns:
        text = re.sub(
            pattern + r'.*?(?=\n\s*[A-Z][a-z]+\s+[a-z]+|\Z)',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
    
    # ========== ETHICS & CONSENT ==========
    ethics_patterns = [
        r'Ethics\s+[Aa]pproval',
        r'Ethical\s+[Aa]pproval',
        r'Ethics\s+[Ss]tatement',
        r'Institutional\s+[Rr]eview\s+[Bb]oard',
        r'IRB\s+[Aa]pproval',
        r'[Cc]onsent\s+(for\s+)?[Pp]ublication',
        r'[Cc]onsent\s+to\s+[Pp]articipate',
        r'[Pp]atient\s+[Cc]onsent',
        r'Informed\s+[Cc]onsent',
        r'Human\s+[Ss]ubjects?\s+[Aa]pproval',
        r'Animal\s+[Ee]thics',
    ]
    
    for pattern in ethics_patterns:
        text = re.sub(
            pattern + r'.*?(?=\n\s*[A-Z][a-z]+\s+[a-z]+|\Z)',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
    
    # ========== COMPETING INTERESTS / CONFLICTS ==========
    competing_patterns = [
        r'Competing\s+[Ii]nterests?',
        r'Conflict\s+of\s+[Ii]nterests?',
        r'Conflicts?\s+of\s+[Ii]nterests?',
        r'Declaration\s+of\s+[Ii]nterests?',
        r'Declaration\s+of\s+[Cc]ompeting\s+[Ii]nterests?',
        r'Disclosure\s+of\s+[Ii]nterests?',
        r'Financial\s+[Dd]isclosures?',
        r'[Nn]o\s+[Cc]ompeting\s+[Ii]nterests?',
    ]
    
    for pattern in competing_patterns:
        text = re.sub(
            pattern + r'.*?(?=\n\s*[A-Z][a-z]+\s+[a-z]+|\Z)',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
    
    # ========== FUNDING / ACKNOWLEDGMENTS ==========
    funding_patterns = [
        r'Funding',
        r'Financial\s+[Ss]upport',
        r'Grant\s+[Ss]upport',
        r'Acknowledgements?',
        r'Acknowledgments?',
        r'Sources?\s+of\s+[Ff]unding',
        r'Sponsor',
        r'Financial\s+[Ss]upport\s+and\s+[Ss]ponsorship',
    ]
    
    for pattern in funding_patterns:
        text = re.sub(
            pattern + r'.*?(?=\n\s*[A-Z][a-z]+\s+[a-z]+|\Z)',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
    
    # ========== DATA AVAILABILITY ==========
    data_availability_patterns = [
        r'Data\s+[Aa]vailability',
        r'Availability\s+of\s+[Dd]ata',
        r'Code\s+[Aa]vailability',
        r'Materials?\s+[Aa]vailability',
        r'Access\s+to\s+[Dd]ata',
        r'Supporting\s+[Ii]nformation',
        r'Supplementary\s+[Mm]aterials?',
    ]
    
    for pattern in data_availability_patterns:
        text = re.sub(
            pattern + r'.*?(?=\n\s*[A-Z][a-z]+\s+[a-z]+|\Z)',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
    
    # ========== PUBLISHER FOOTERS ==========
    publisher_patterns = [
        r'Submit\s+your\s+(next\s+)?manuscript',
        r'Publish\s+with\s+us',
        r'For\s+submission\s+guidelines',
        r'www\.[a-z]+central\.com',
        r'Springer\s+Nature',
        r'Inclusion\s+in\s+PubMed',
        r'Maximum\s+visibility\s+for\s+your\s+research',
        r'Open\s+[Aa]ccess\s+This\s+article',
        r'licensed\s+under\s+a\s+Creative\s+Commons',
    ]
    
    for pattern in publisher_patterns:
        text = re.sub(
            pattern + r'.*?(?=\n\s*[A-Z][a-z]+|\Z)',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
    
    # ========== PAGE HEADERS/FOOTERS ==========
    # Common pattern: "Author et al. Journal Name Year"
    # or "Journal Name (Year) Vol:Pages"
    header_footer_patterns = [
        r'\n.*?et\s+al\..*?\d{4}.*?\n',
        r'\n.*?Page\s+\d+\s+of\s+\d+.*?\n',
        r'\n\s*\d+\s*\n',  # Standalone page numbers
        r'\n.*?\(\d{4}\)\s+\d+:\d+.*?\n',  # Journal (2018) 12:17 format
    ]
    
    for pattern in header_footer_patterns:
        text = re.sub(pattern, '\n', text)
    
    # ========== COPYRIGHT & LICENSE ==========
    copyright_patterns = [
        r'©\s*\d{4}.*?(?=\n\n|\Z)',
        r'Copyright\s+©?\s*\d{4}.*?(?=\n\n|\Z)',
        r'All\s+rights\s+reserved.*?(?=\n\n|\Z)',
        r'This\s+is\s+an\s+[Oo]pen\s+[Aa]ccess\s+article.*?(?=\n\n|\Z)',
        r'Received:.*?Accepted:.*?Published:.*?\n',  # Date stamps
    ]
    
    for pattern in copyright_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # ========== CORRESPONDENCE INFO ==========
    correspondence_patterns = [
        r'Correspondence:.*?(?=\n\s*[A-Z][a-z]+\s+[a-z]+|\Z)',
        r'\*\s*Correspondence:.*?(?=\n\n|\Z)',
        r'Contact:.*?(?=\n\n|\Z)',
        r'Email:.*?(?=\n\n|\Z)',
        r'[Aa]ddress\s+for\s+correspondence:.*?(?=\n\n|\Z)',
    ]
    
    for pattern in correspondence_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # ========== ABBREVIATIONS ==========
    abbreviations_patterns = [
        r'Abbreviations?.*?(?=\n\s*[A-Z][a-z]+\s+[a-z]+|\Z)',
        r'List\s+of\s+[Aa]bbreviations?.*?(?=\n\s*[A-Z][a-z]+\s+[a-z]+|\Z)',
        r'Acronyms?.*?(?=\n\s*[A-Z][a-z]+\s+[a-z]+|\Z)',
    ]
    
    for pattern in abbreviations_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # ========== PEER REVIEW INFO ==========
    peer_review_patterns = [
        r'Peer\s+[Rr]eview\s+[Ii]nformation',
        r'Reviewer\s+[Cc]omments?',
        r'Editorial\s+[Nn]ote',
        r'Editors?:\s+.*?(?=\n\n|\Z)',
    ]
    
    for pattern in peer_review_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # ========== CLEAN UP WHITESPACE ==========
    # Remove excessive line breaks (more than 2 consecutive)
    # IMPORTANT: Keep double newlines for paragraph detection!
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove trailing/leading whitespace
    text = text.strip()
    
    # Remove lines that are just special characters or numbers
    text = re.sub(r'\n[^a-zA-Z]*\n', '\n', text)
    
    return text


# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    # Example usage
    sample_text = """
    This is the main content of the paper.
    
    Methods
    We collected data from 100 participants.
    
    Results
    The findings show significant effects.
    
    References
    1. Smith et al. (2020)
    2. Jones et al. (2019)
    
    Authors' contributions
    MS completed analysis and writing.
    
    Ethics approval and consent to participate
    IRB approved this study.
    
    Competing interests
    The authors declare no competing interests.
    
    Submit your next manuscript to BioMed Central
    """
    
    cleaned = clean_academic_text(sample_text)
    print("CLEANED TEXT:")
    print(cleaned)
    print("\n" + "="*50)
    print(f"Original length: {len(sample_text)} chars")
    print(f"Cleaned length: {len(cleaned)} chars")
    print(f"Removed: {len(sample_text) - len(cleaned)} chars ({100*(len(sample_text) - len(cleaned))/len(sample_text):.1f}%)")